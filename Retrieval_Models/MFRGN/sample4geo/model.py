import torch
import timm
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from timm.models import convnext


class TimmModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, img_size: int = 384):
        super(TimmModel, self).__init__()
        self.img_size = img_size
        # 根据模型名称选择对应的预训练ConvNeXt权重路径
        if 'tiny' in model_name:
            bk_checkpoint = 'pretrained/convnext_tiny_22k_1k_224.pth'
        elif 'base' in model_name:
            bk_checkpoint = 'pretrained/convnext_base_22k_1k_224.pth'
        else:
            bk_checkpoint = None
        if bk_checkpoint and '384' in bk_checkpoint:
            bk_checkpoint = bk_checkpoint.replace('224', '384')
        # 初始化 timm 模型（无分类头，提取特征）
        if "vit" in model_name:
            # ViT 模型需指定 img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size,
                                           pretrained_cfg_overlay={'file': bk_checkpoint} if bk_checkpoint else {})
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                           pretrained_cfg_overlay={'file': bk_checkpoint} if bk_checkpoint else {})
        # 初始化 InfoNCE 温度参数 logit_scale
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None, input_id=1):
        # 支持双输入：同时输出两张图像的特征
        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2
        else:
            image_features = self.model(img1)
            return image_features


# 仅用于模块测试的代码
if __name__ == '__main__':
    from torchstat import stat
    from ptflops import get_model_complexity_info

    model_name = 'convnext_base.fb_in22k_ft_in1k_384'
    pretrained = True
    img_size = 384
    model = TimmModel(model_name, pretrained, img_size)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (M): {params / 1e6}")
    x = torch.randn(2, 3, 384, 384)
    y1, y2 = model(x, x)
    print(f"Output shapes: {y1.shape}, {y2.shape}")
    macs, params = get_model_complexity_info(model, (3, 140, 768), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
