# -*- coding: utf-8 -*-
# Retrieval_Models/MFRGN/common.py

import math
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


# -----------------------------
# Pyramid Pooling (PSP) modules
# -----------------------------
class PSPModule(nn.Module):
    """经典 PSPNet 风格的金字塔池化，将不同池化尺寸的特征拼接到通道维度之外的“点数”维。"""
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            return nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            return nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            return nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        else:
            raise ValueError("Unsupported dimension for PSPModule.")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (N, C, H, W)
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]  # list of (N, C, Ni)
        center = torch.cat(priors, dim=-1)  # (N, C, sum(Ni))
        return center


class PSP(nn.Module):
    """和上面 PSPModule 类似，但允许 (h, w) 形式的池化尺寸列表。"""
    def __init__(self, sizes=[(1, 1), (3, 3), (6, 6), (8, 8)], dimension=2):
        super(PSP, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            return nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            return nn.AdaptiveAvgPool2d(output_size=(size[0], size[1]))
        elif dimension == 3:
            return nn.AdaptiveAvgPool3d(output_size=(size[0], size[1], size[2]))
        else:
            raise ValueError("Unsupported dimension for PSP.")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (N, C, H, W)
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]  # list of (N, C, Ni)
        center = torch.cat(priors, dim=-1)  # (N, C, sum(Ni))
        return center


# ---------------------------------
# Self-Cross Transformer (LoFTR风格)
# ---------------------------------
class scTransformerLayer(nn.Module):
    """
    单层 Self-Cross Transformer：
      - 输入 q_src: (N, L, C)
      - 输入 kv_src: (N, S, C)
    """
    def __init__(
        self, d_model, nheads, dim_feedforward, dropout,
        is_ffn=True, qk_cat=True, q_low=False,
        activation=nn.ReLU(inplace=True), mode='linear'
    ):
        super().__init__()
        self.is_ffn = is_ffn
        self.qk_cat = qk_cat
        self.q_low = q_low
        self.nheads = nheads
        self.dim = d_model // nheads
        self.mode = mode

        assert self.mode in ('linear', 'full', 'conv1'), "scTransformerLayer mode 必须是 'linear' / 'full' / 'conv1'"

        # 线性投影
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.merge      = nn.Linear(d_model, d_model, bias=False)

        # 注意力实现
        self.sc_attn = LinearAttention() if mode == 'linear' else FullAttention()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = activation

        # FFN
        if self.is_ffn:
            if mode in ('linear', 'full'):
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, dim_feedforward, bias=False),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model, bias=False)
                )
            else:  # conv1
                t1 = int(abs((math.log(d_model, 2) + 1) / 2))
                k1_size = t1 if t1 % 2 else t1 + 1
                t2 = int(abs((math.log(dim_feedforward, 2) + 1) / 2))
                k2_size = t2 if t2 % 2 else t2 + 1
                self.mlp = nn.Sequential(
                    nn.Conv1d(d_model, dim_feedforward, kernel_size=k1_size, padding=(k1_size - 1) // 2, bias=False),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Conv1d(dim_feedforward, d_model, kernel_size=k2_size, padding=(k2_size - 1) // 2, bias=False),
                )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query_proj.weight.data); constant_(self.query_proj.bias.data, 0.)
        xavier_uniform_(self.key_proj.weight.data);   constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data); constant_(self.value_proj.bias.data, 0.)

    def forward(self, q_src, kv_src, q_mask=None, kv_mask=None):
        """
        q_src: (N, L, C)
        kv_src: (N, S, C)
        """
        b = q_src.shape[0]
        # 线性变换并拆分多头
        query = self.query_proj(q_src).view(b, -1, self.nheads, self.dim)  # (N, L, H, D)

        # 根据配置拼接 K/V 的来源
        if self.qk_cat:
            if not self.q_low:
                key = value = torch.cat([kv_src, q_src], dim=1)
            else:
                key = value = torch.cat([q_src, kv_src], dim=1)
        else:
            key = value = kv_src

        key   = self.key_proj(key).view(b, -1, self.nheads, self.dim)     # (N, S', H, D)
        value = self.value_proj(value).view(b, -1, self.nheads, self.dim)  # (N, S', H, D)

        att = self.sc_attn(query, key, value, q_mask, kv_mask)             # (N, L, H, D)
        att = self.merge(att.view(b, -1, self.nheads * self.dim))          # (N, L, C)

        # 残差 + LN
        att = self.norm1(q_src + self.dropout1(att))

        # FFN
        if self.is_ffn:
            if self.mode == 'conv1':
                att = att.permute(0, 2, 1)
            att = att + self.dropout2(self.mlp(att))
            if self.mode == 'conv1':
                att = att.permute(0, 2, 1)
            att = self.norm2(att)

        return att, kv_src


class scTransformerEncoder(nn.Module):
    """将 scTransformerLayer 堆叠若干层。"""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, q_src, kv_src, q_mask=None, kv_mask=None):
        for layer_block in self.layers:
            q_src, kv_src = layer_block(q_src, kv_src, q_mask, kv_mask)
        return q_src, kv_src


# --------------------------------------------
# 动态正弦位置编码（按输入尺寸与通道数实时生成）
# --------------------------------------------
class PositionEncodingSine(nn.Module):
    """
    2D 正弦/余弦位置编码。
    关键：
      - 不再预先固定 shape，而是在 forward 时根据输入 (B, C, H, W) 动态生成；
      - 输出与输入同型，并与输入相加：  x + pe ；
      - 通道数以输入的 C 为准（即使构造时 d_model 传错也能自适配）。
    """
    def __init__(self, d_model: int = None, normalize: bool = False, scale: float = 2 * math.pi, temp_bug_fix: bool = True):
        super().__init__()
        self.d_model = d_model
        self.normalize = normalize
        self.scale = scale
        self.temp_bug_fix = temp_bug_fix

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: x + PE  (同形)
        """
        assert x.dim() == 4, "PositionEncodingSine 期望输入为 4D 张量 (B, C, H, W)"
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # 网格坐标
        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)  # (H, W)
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)  # (H, W)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed / (H + eps)) * self.scale
            x_embed = (x_embed / (W + eps)) * self.scale

        # 频率项（与 LoFTR 的实现一致）
        half = max(C // 2, 1)     # 注意：按输入通道 C 的一半来定义频率带数
        denom = float(half) if self.temp_bug_fix else float(C)
        div_term = torch.exp(torch.arange(0, half, 2, device=device, dtype=dtype) * (-math.log(10000.0) / denom))
        # 目标位置编码张量
        pe = torch.zeros((C, H, W), device=device, dtype=dtype)

        # 通道分配：每个频率产生 4 个通道：sin(x), cos(x), sin(y), cos(y)
        # 处理 C 不是 4 的倍数的鲁棒情况（截断/不满的部分保持 0）
        # 可向量化实现（当 C % 4 == 0 时对齐最好），这里做安全切片：
        num_slots = min(div_term.shape[0], pe[0::4].shape[0])  # 实际能填充的“组数”
        if num_slots > 0:
            pe[0:4*num_slots:4, :, :] = torch.sin(x_embed * div_term[:num_slots, None, None])  # sin x
            pe[1:4*num_slots:4, :, :] = torch.cos(x_embed * div_term[:num_slots, None, None])  # cos x
            pe[2:4*num_slots:4, :, :] = torch.sin(y_embed * div_term[:num_slots, None, None])  # sin y
            pe[3:4*num_slots:4, :, :] = torch.cos(y_embed * div_term[:num_slots, None, None])  # cos y

        return x + pe  # 与输入特征相加，形状不变


# --------------------
# 工具 / 注意力实现
# --------------------
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def elu_feature_map(x):
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """LoFTR 中的线性注意力实现。"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """
        queries: (N, L, H, D)
        keys:    (N, S, H, D)
        values:  (N, S, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # mask
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / max(v_length, 1)

        KV = torch.einsum("nshd,nshv->nhdv", K, values)                       # (N, H, D, V)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)   # (N, L, H)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * max(v_length, 1)
        return queried_values.contiguous()


class FullAttention(nn.Module):
    """标准 Scaled Dot-Product Attention 实现。"""
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """
        queries: (N, L, H, D)
        keys:    (N, S, H, D)
        values:  (N, S, H, D)
        """
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)  # (N, L, S, H)

        if kv_mask is not None and q_mask is not None:
            # 广播到 (N, L, S, 1)
            mask = q_mask[:, :, None, None] * kv_mask[:, None, :, None]
            QK.masked_fill_(~mask, float('-inf'))

        softmax_temp = 1.0 / (queries.size(-1) ** 0.5)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        out = torch.einsum("nlsh,nshd->nlhd", A, values)
        return out.contiguous()
