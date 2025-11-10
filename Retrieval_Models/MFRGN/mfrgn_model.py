# -*- coding: utf-8 -*-
import os
import math
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import timm

# 从 common 模块导入 MFRGN 组件
from Retrieval_Models.MFRGN.common import (
    scTransformerLayer,
    scTransformerEncoder,
    PositionEncodingSine,
    PSP,
)

"""
ConvNeXt 模型名称映射:
{
  'convnext_tiny_in22ft1k':        'convnext_tiny.fb_in22k_ft_in1k',
  'convnext_small_in22ft1k':       'convnext_small.fb_in22k_ft_in1k',
  'convnext_base_in22ft1k':        'convnext_base.fb_in22k_ft_in1k',
  'convnext_large_in22ft1k':       'convnext_large.fb_in22k_ft_in1k',
  'convnext_xlarge_in22ft1k':      'convnext_xlarge.fb_in22k_ft_in1k',
  'convnext_tiny_384_in22ft1k':    'convnext_tiny.fb_in22k_ft_in1k_384',
  'convnext_small_384_in22ft1k':   'convnext_small.fb_in22k_ft_in1k_384',
  'convnext_base_384_in22ft1k':    'convnext_base.fb_in22k_ft_in1k_384',
  'convnext_large_384_in22ft1k':   'convnext_large.fb_in22k_ft_in1k_384',
  'convnext_xlarge_384_in22ft1k':  'convnext_xlarge.fb_in22k_ft_in1k_384',
  'convnext_tiny_in22k':           'convnext_tiny.fb_in22k',
  'convnext_small_in22k':          'convnext_small.fb_in22k',
  'convnext_base_in22k':           'convnext_base.fb_in22k',
  'convnext_large_in22k':          'convnext_large.fb_in22k',
  'convnext_xlarge_in22k':         'convnext_xlarge.fb_in22k',
  'convnextv2_tiny_22k_224_ema':   'convnextv2_tiny.fcmae_ft_in22k_in1k',
  'convnextv2_tiny_22k_384_ema':   'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
}
"""


def weights_init_kaiming(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if getattr(m, 'affine', False):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# ----------------------------- Backbone ----------------------------- #
class Backbone(nn.Module):
    """
    统一封装 ResNet / ConvNeXt 主干，支持返回中间层特征
    """
    def __init__(self,
                 model_name: str,
                 bk_checkpoint: Union[str, None],
                 return_interm_layers: bool,
                 img_size=(122, 671),
                 pretrained: bool = True):
        super().__init__()
        self.name = model_name
        self.data_config = None

        if 'resnet' in self.name.lower():
            # 仅示例支持 18/34/50
            name_upper = self.name.replace('resnet', 'ResNet').title()  # resnet50 -> Resnet50
            assert name_upper in ('Resnet18', 'Resnet34', 'Resnet50'), "number of channels are hard coded"
            ctor = getattr(torchvision.models, self.name.lower())
            if pretrained:
                if name_upper == 'Resnet50':
                    backbone = ctor(weights=ResNet50_Weights.IMAGENET1K_V1)
                elif name_upper == 'Resnet34':
                    backbone = ctor(weights=ResNet34_Weights.IMAGENET1K_V1)
                else:
                    backbone = ctor(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = ctor(weights=None)

            if return_interm_layers:
                # 使用 layer2/3/4
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                if name_upper == 'Resnet50':
                    self.num_channels = [512, 1024, 2048]
                else:
                    self.num_channels = [128, 256, 512]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                if name_upper == 'Resnet50':
                    self.num_channels = [2048]
                else:
                    self.num_channels = [512]

            self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        elif 'convnext' in self.name.lower():
            # timm ConvNeXt
            if pretrained:
                # 通过 overlay 指定本地权重文件
                self.backbone = timm.create_model(
                    self.name, pretrained=True, num_classes=0,
                    pretrained_cfg_overlay=dict(file=bk_checkpoint)
                )
            else:
                self.backbone = timm.create_model(self.name, pretrained=False, num_classes=0)

            # 记录数据配置
            try:
                self.data_config = timm.data.resolve_model_data_config(self.backbone)
            except Exception:
                self.data_config = None

            if return_interm_layers:
                self.strides = [8, 16, 32]
                if 'base' in self.name.lower():
                    self.num_channels = [256, 512, 1024]
                elif 'tiny' in self.name.lower():
                    self.num_channels = [192, 384, 768]
                else:
                    # 默认按 base
                    self.num_channels = [256, 512, 1024]
            else:
                self.strides = [32]
                self.num_channels = [1024]
        else:
            raise RuntimeError(f'error model_name (expect resnet* or convnext*), got {self.name}')

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if 'resnet' in self.name.lower():
            features = self.backbone(x)  # OrderedDict -> list
            return [feat for _, feat in features.items()]

        elif 'convnext' in self.name.lower():
            # ConvNeXt: 手动取各个 stage 输出
            x = self.backbone.stem(x)
            x0 = self.backbone.stages[0](x)      # stride 4
            x1 = self.backbone.stages[1](x0)     # stride 8
            x2 = self.backbone.stages[2](x1)     # stride 16
            x3 = self.backbone.stages[3](x2)     # stride 32
            return [x1, x2, x3]

        return []


# --------------------------- BackboneEmbed -------------------------- #
class BackboneEmbed(nn.Module):
    """
    将主干输出的特征图映射到 d_model，并叠加正弦位置编码；
    若需要中间层特征，则对最后一层再做一次 stride=2 的额外下采样作为第 4 个尺度。
    """
    def __init__(self, d_model, backbone_strides, backbone_num_channels, return_interm_layers: bool):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.d_model = d_model
        self.pos_embed = PositionEncodingSine(d_model=self.d_model)

        if self.return_interm_layers:
            num_outs = len(backbone_strides) + 1  # 多一层额外下采样
            input_proj_list = []
            for n in range(num_outs):
                if n == num_outs - 1:
                    in_channels = backbone_num_channels[n - 1]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    ))
                else:
                    in_channels = backbone_num_channels[n]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ])

    def forward(self, features: list[torch.Tensor]):
        feats_embed = []
        srcs = []
        for i, feat in enumerate(features):
            src = self.input_proj[i](feat)
            srcs.append(src)
            pos = self.pos_embed(src)
            feats_embed.append(pos)

        if self.return_interm_layers:
            # 最后一层额外下采样一次
            src = self.input_proj[-1](features[-1])
            srcs.append(src)
            pos = self.pos_embed(src)
            feats_embed.append(pos)

        return feats_embed, srcs


# ------------------------------ TimmModel --------------------------- #
class TimmModel(nn.Module):
    """
    旧版双分支占位（卫星/地面各自一套），University-1652 实际用的是 TimmModel_u。
    这里保留构造逻辑与权重解析以与旧脚本兼容；forward 可按需自行实现。
    """
    def __init__(self, model_name, sat_size, grd_size, psm=True, is_polar=False, pretrained=True):
        super().__init__()
        self.is_polar = is_polar
        self.backbone_name = model_name

        # 模型超参
        self.d_model = 128
        self.nheads = 4
        self.nlayers = 2
        self.ffn_dim = 1024
        self.dropout = 0.3
        self.em_dim = 2048
        self.activation = nn.GELU()
        self.single_features = False
        self.sat_size = sat_size
        self.grd_size = grd_size
        self.sample = psm

        # 仅作为占位，实际项目使用 TimmModel_u
        # 可参考 TimmModel_u 的 __init__/forward 按需补齐
        self.model = nn.Identity()

    def forward(self, x1, x2=None, input_id=1):
        return self.model(x1), (self.model(x2) if x2 is not None else None)


# ----------------------------- Utilities ---------------------------- #
def _append_half_scale(H: list[int], W: list[int]):
    """在列表末尾追加一次 /2 的下采样尺寸（保证最小为 1）。"""
    H2 = max(1, int(math.floor(H[-1] / 2)))
    W2 = max(1, int(math.floor(W[-1] / 2)))
    H.append(H2)
    W.append(W2)


# ------------------------------ TimmModel_u ------------------------- #
# 针对 University-1652（无人机 ↔ 卫星），两条分支共享整个网络：
class TimmModel_u(nn.Module):
    def __init__(self, model_name, img_size, psm=True, is_polar=False, pretrained=True):
        super().__init__()
        self.is_polar = is_polar
        self.backbone_name = model_name

        # 默认输入尺寸（University-1652 使用正方形 384x384）
        self.img_size = (img_size, img_size)

        # 模型超参
        self.sample = psm
        self.d_model = 128
        self.nheads = 4
        self.nlayers = 2
        self.ffn_dim = 1024
        self.dropout = 0.3
        self.em_dim = 2048
        self.activation = nn.GELU()
        self.single_features = False  # 使用多尺度特征

        # ---- 预训练权重文件名解析（基于 model_name） ---- #
        pretrained_dir = os.path.join(os.path.dirname(__file__), "pretrained")
        model_name = self.backbone_name  # 便于书写
        weight_file = None

        if model_name.lower().startswith("convnext"):
            # ConvNeXt V1: 22k→1k
            if "fb_in22k_ft_in1k" in model_name or "in22ft1k" in model_name:
                base_name = model_name.split('.')[0]
                if model_name.endswith("_384") or "384" in model_name:
                    weight_file = f"{base_name}_22k_1k_384.pth"
                else:
                    weight_file = f"{base_name}_22k_1k_224.pth"

            # ConvNeXt V1: 仅 22k 预训练
            elif "fb_in22k" in model_name and "ft_in1k" not in model_name:
                base_name = model_name.split('.')[0]
                weight_file = f"{base_name}_22k_224.pth"

            # ConvNeXt V2: fcmae 微调
            elif "fcmae_ft_in22k_in1k" in model_name:
                base_name = model_name.split('.')[0]
                if model_name.endswith("_384") or "384" in model_name:
                    weight_file = f"{base_name}_22k_384_ema.pt"
                else:
                    weight_file = f"{base_name}_22k_224_ema.pt"

            # 其他 v2
            elif model_name.lower().startswith("convnextv2"):
                weight_file = f"{model_name}.pt"

            else:
                # 兜底：按型号猜测
                ml = model_name.lower()
                if "xlarge" in ml:
                    weight_file = "convnext_xlarge_22k_1k_224.pth"
                elif "large" in ml:
                    weight_file = "convnext_large_22k_1k_224.pth"
                elif "base" in ml:
                    weight_file = "convnext_base_1k_224.pth"
                elif "small" in ml:
                    weight_file = "convnext_small_1k_224.pth"
                elif "tiny" in ml:
                    weight_file = "convnext_tiny_1k_224.pth"
                else:
                    weight_file = f"{model_name}.pth"
        else:
            # 非 ConvNeXt：直接同名
            weight_file = f"{model_name}.pth"

        self.bk_checkpoint = os.path.join(pretrained_dir, weight_file)
        if not os.path.isfile(self.bk_checkpoint):
            raise FileNotFoundError(
                f"未找到模型 '{model_name}' 的预训练权重文件: {self.bk_checkpoint}\n"
                f"请将权重文件 {weight_file} 放入 {pretrained_dir} 目录下。"
            )

        # ---- PSM（位置特定采样模块） ---- #
        if self.sample:
            self.norm1 = nn.LayerNorm(self.d_model)
            self.sample_L = PSP(sizes=[(1, 1), (6, 6), (12, 12), (21, 21)], dimension=2)
            self.in_dim_L = 622  # 由 PSP 配置决定

        # ---- 主干与嵌入 ---- #
        self.backbone = Backbone(
            self.backbone_name,
            self.bk_checkpoint,
            return_interm_layers=not self.single_features,
            pretrained=pretrained,
        )
        self.embed = BackboneEmbed(
            self.d_model,
            self.backbone.strides,
            self.backbone.num_channels,
            return_interm_layers=not self.single_features,
        )

        # ---- 多尺度自交叉注意力（共享权重） ---- #
        layer_H = scTransformerLayer(
            self.d_model, self.nheads, self.ffn_dim, self.dropout,
            activation=self.activation, is_ffn=True
        )
        self.transformer_H = scTransformerEncoder(layer_H, num_layers=2)

        layer_L = scTransformerLayer(
            self.d_model, self.nheads, self.ffn_dim, self.dropout,
            activation=self.activation, is_ffn=True, q_low=True
        )
        self.transformer_L = scTransformerEncoder(layer_L, num_layers=1)

        out_dim_g = 14

        # ---- 依据输入尺寸推导各尺度空间形状 ---- #
        self.feat_dim, self.H, self.W = self._dim(
            self.backbone_name, self.backbone.strides, img_size=self.img_size
        )

        # 生成投影维度（全局）
        if self.sample:
            in_dim = sum(self.feat_dim[1:]) + self.in_dim_L
        else:
            in_dim = sum(self.feat_dim)
        self.proj = nn.Linear(in_dim, out_dim_g)

        # ---- 局部分支（Geo-APB） ---- #
        self.num_channles = self.backbone.num_channels[:]   # 原拼写保持一致
        self.num_channles.append(self.d_model)              # 额外一层（来自 Transformer 的 d_model）
        ratio = 1

        proj_gl = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model * ratio,
                          kernel_size=self.k_size(self.d_model),
                          padding=(self.k_size(self.d_model) - 1) // 2),
                nn.BatchNorm1d(self.d_model * ratio),
                nn.Conv1d(self.d_model * ratio, self.num_channles[i],
                          kernel_size=self.k_size(self.d_model * ratio),
                          padding=(self.k_size(self.d_model * ratio) - 1) // 2),
                nn.GELU(),
                nn.BatchNorm1d(self.num_channles[i]),
            )
            for i in range(len(self.num_channles))
        )
        proj_gl.apply(weights_init_kaiming)
        self.proj_gl = proj_gl

        # 通道注意力（各尺度）
        ch = [nn.Conv2d(self.num_channles[i], self.num_channles[i], kernel_size=1)
              for i in range(len(self.H))]
        self.ch = nn.Sequential(*ch)

        # 空间注意力（各尺度），首层在启用 PSM 时使用“展开核”
        k = [9, 7, 5, 3]
        sp = [nn.Conv2d(1, 1, kernel_size=k[i], padding=(k[i] - 1) // 2)
              for i in range(len(self.num_channles))]
        if self.sample:
            sp[0] = nn.Conv2d(1, 1, kernel_size=(k[0] * k[0], 1),
                              padding=((k[0] * k[0] - 1) // 2, 0))
        self.sp = nn.Sequential(*sp)

        # 池化/激活与局部投影
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        out_dim_l = 256
        self.proj_local = nn.Linear(sum(self.num_channles), out_dim_l)

        # CLIP 风格温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # --------------------------- helpers --------------------------- #
    def get_config(self):
        return getattr(self.backbone, "data_config", None)

    def set_grad_checkpointing(self, enable: bool = True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)

    def k_size(self, in_dim: int) -> int:
        # 基于特征维度确定 Geo-APB 分支的一维卷积核大小
        t = int(abs((math.log(in_dim, 2) + 1) / 2))
        return t if (t % 2) else (t + 1)

    # --------------------------- forward --------------------------- #
    def forward(self, img_sat: torch.Tensor, img_grd: Union[torch.Tensor, None] = None, input_id: int = 1):
        """
        约定：
        - 训练/配对：img_sat（卫星/上视）作为第1个输入，img_grd（无人机/地面）作为第2个输入；
        - 单输入：用于检索特征抽取。
        """
        if img_grd is not None:
            # ----- 双输入（训练/评测） -----
            bs_sat = img_sat.shape[0]
            bs_grd = img_grd.shape[0]

            sat_feats = self.backbone(img_sat)
            grd_feats = self.backbone(img_grd)

            sat_e, sat_src = self.embed(sat_feats)
            grd_e, grd_src = self.embed(grd_feats)

            # 展平特征 -> Transformer 输入 (B, HW, C)
            sat_embed = [x.flatten(2).transpose(1, 2) for x in sat_e]
            grd_embed = [x.flatten(2).transpose(1, 2) for x in grd_e]

            # 低层 L（PSM）
            if self.sample:
                L_sat_embed = self.sample_L(sat_e[0])
                L_sat_embed = self.norm1(L_sat_embed.flatten(2).transpose(1, 2))
                L_grd_embed = self.sample_L(grd_e[0])
                L_grd_embed = self.norm1(L_grd_embed.flatten(2).transpose(1, 2))
                # 同步给局部分支的最浅层特征
                sat_feats[0] = self.sample_L(sat_feats[0])
                grd_feats[0] = self.sample_L(grd_feats[0])
            else:
                L_sat_embed = sat_embed[0]
                L_grd_embed = grd_embed[0]

            # 高层 H（拼接除第 1 个尺度外的其余尺度）
            H_sat_embed = torch.cat(sat_embed[1:], dim=1)
            H_grd_embed = torch.cat(grd_embed[1:], dim=1)

            # 自交叉注意力（两条分支共享）
            sat_H, sat_L = self.transformer_H(H_sat_embed, L_sat_embed)
            sat_L, sat_H = self.transformer_L(sat_L, sat_H)
            grd_H, grd_L = self.transformer_H(H_grd_embed, L_grd_embed)
            grd_L, grd_H = self.transformer_L(grd_L, grd_H)

            # 全局描述子
            sat_cat = torch.cat([sat_L, sat_H], dim=1)
            grd_cat = torch.cat([grd_L, grd_H], dim=1)
            sat_global = self.proj(sat_cat.transpose(1, 2)).view(bs_sat, -1)
            grd_global = self.proj(grd_cat.transpose(1, 2)).view(bs_grd, -1)

            # 将 H 拆回 3 段
            sat_h1, sat_h2, sat_h3 = self._reshape_feat(sat_H, self.H[1:], self.W[1:])
            grd_h1, grd_h2, grd_h3 = self._reshape_feat(grd_H, self.H[1:], self.W[1:])

            # 附加位置（来自最后一层 src）
            sat_feats.append(sat_src[-1])
            grd_feats.append(grd_src[-1])

            # Geo-APB 局部描述子
            sat_local = self._gpab(
                sat_feats, [L_sat_embed, sat_h1, sat_h2, sat_h3],
                proj=self.proj_gl, ch_att=self.ch, sp_att=self.sp, h=self.H, w=self.W
            )
            grd_local = self._gpab(
                grd_feats, [L_grd_embed, grd_h1, grd_h2, grd_h3],
                proj=self.proj_gl, ch_att=self.ch, sp_att=self.sp, h=self.H, w=self.W
            )
            sat_local = self.proj_local(sat_local)
            grd_local = self.proj_local(grd_local)

            # 拼接全局/局部并归一化
            desc_sat = torch.cat([sat_global, sat_local], dim=1)
            desc_grd = torch.cat([grd_global, grd_local], dim=1)
            desc_sat = F.normalize(desc_sat, p=2, dim=1)
            desc_grd = F.normalize(desc_grd, p=2, dim=1)
            return desc_sat, desc_grd

        # ----- 单输入（检索阶段抽特征） -----
        B = img_sat.shape[0]
        feats = self.backbone(img_sat)
        feats_e, feats_src = self.embed(feats)
        feats_embed = [x.flatten(2).transpose(1, 2) for x in feats_e]

        if self.sample:
            L_feat = self.sample_L(feats_e[0])
            L_feat = self.norm1(L_feat.flatten(2).transpose(1, 2))
            feats[0] = self.sample_L(feats[0])
        else:
            L_feat = feats_embed[0]

        H_feat = torch.cat(feats_embed[1:], dim=1)
        H_out, L_out = self.transformer_H(H_feat, L_feat)
        L_out, H_out = self.transformer_L(L_out, H_out)

        combined = torch.cat([L_out, H_out], dim=1)
        global_desc = self.proj(combined.transpose(1, 2)).view(B, -1)

        h1, h2, h3 = self._reshape_feat(H_out, self.H[1:], self.W[1:])
        feats.append(feats_src[-1])
        local_desc = self._gpab(
            feats, [L_out, h1, h2, h3],
            proj=self.proj_gl, ch_att=self.ch, sp_att=self.sp, h=self.H, w=self.W
        )
        local_desc = self.proj_local(local_desc)

        desc = torch.cat([global_desc, local_desc], dim=1)
        desc = F.normalize(desc, p=2, dim=1)
        return desc

    # ----------------------- inner functions ----------------------- #
    def _reshape_feat(self, feat_H: torch.Tensor, H: list[int], W: list[int]):
        """
        将拼接后的高层序列按空间比例切回三段（对应三个尺度）。
        H/W 传入的是从第 2 个尺度开始的尺寸列表（长度为 3）。
        """
        p1 = H[0] * W[0]
        p2 = H[-1] * W[-1]
        feat_h1 = feat_H[:, :p1, :].contiguous()
        feat_h2 = feat_H[:, p1:-p2, :].contiguous()
        feat_h3 = feat_H[:, -p2:, :].contiguous()
        return [feat_h1, feat_h2, feat_h3]

    def _gpab(self,
              local_feats: list[torch.Tensor],
              global_feats: list[torch.Tensor],
              proj: nn.ModuleList,
              ch_att: nn.Sequential,
              sp_att: nn.Sequential,
              h: list[int], w: list[int]):
        """
        Geo-Attention Pooling for local descriptors.
        local_feats: 各尺度局部特征图（含最浅层 + 最后一层 src）
        global_feats: [L, H1, H2, H3] 的序列表征（B, S, C）
        """
        geo_att = []
        for i, feat in enumerate(local_feats):
            # (B, C, S) <- 先把 (B, S, C) 转为 (B, C, S)
            global_feat = proj[i](global_feats[i].transpose(1, 2).contiguous())
            b, c, _ = global_feat.shape

            if self.sample and i == 0:
                # PSM 输出按序列处理
                feat = feat.unsqueeze(-1)
                global_feat = global_feat.unsqueeze(-1)
            else:
                # 其余尺度还原为空间网格
                global_feat = global_feat.reshape(b, c, h[i], w[i])

            # 通道注意力
            avg_out = self.avg_pool(global_feat)
            att_ch = ch_att[i](avg_out)

            # 空间注意力
            max_out, _ = torch.max(global_feat, dim=1, keepdim=True)
            att_sp = sp_att[i](max_out)

            # 施加注意力
            m = feat * self.sigmoid(att_ch) * self.sigmoid(att_sp)
            m = feat + m
            m = self.avg_pool(m)
            geo_att.append(m.view(b, -1))

        results = torch.cat(geo_att, dim=-1).contiguous()
        return results

    def _dim(self, model_name: str, strides: list[int], img_size=(122, 671)):
        """
        根据输入尺寸与主干步幅计算各尺度的空间大小（同时为多一层“额外下采样”预留尺寸）。
        """
        if 'convnext' in model_name.lower():
            H = [int(math.floor(img_size[0] / r)) for r in strides]
            W = [int(math.floor(img_size[1] / r)) for r in strides]
            # 为与 BackboneEmbed 的“额外下采样”对齐，再追加一层（=最后一层的 /2）
            _append_half_scale(H, W)
            feat_dim = [H[i] * W[i] for i in range(len(H))]
        elif 'resnet' in model_name.lower():
            H = [int(math.ceil(img_size[0] / r)) for r in strides]
            W = [int(math.ceil(img_size[1] / r)) for r in strides]
            _append_half_scale(H, W)
            feat_dim = [H[i] * W[i] for i in range(len(H))]
        else:
            raise RuntimeError("Unknown backbone for _dim")

        return feat_dim, H, W
