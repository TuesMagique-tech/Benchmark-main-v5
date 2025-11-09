import os
import torch
from PIL import Image
from torchvision import transforms
from Retrieval_Models.MFRGN.get_MFRGN import get_MFRGN

# 与训练一致，使用 384×384
IMG_SIZE = (384, 384)
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_mfrgn_model():
    """（备用示例）加载标准 MFRGN（CVUSA/CVACT 版本），当前不使用。"""
    from Retrieval_Models.MFRGN import mfrgn_model
    model = mfrgn_model.TimmModel(model_name='convnext_base', pretrained=False)
    weight_path = os.path.join(os.path.dirname(__file__), 'MFRGN', 'weights', 'MFRGN_cvact.pth')
    if os.path.isfile(weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_Model(method_name: str, config=None):
    """
    根据检索方法名返回对应模型与图像预处理变换。
    注意：MFRGN_U1652 必须传入 config（dict 或 yaml 路径），以便加载你本地训练的 ckpt。
    """
    name = method_name.strip().upper()

    if name == 'MFRGN':
        # 旧的标准 MFRGN（CVUSA/CVACT）备用，不建议用
        model = load_mfrgn_model()
        img_transform = IMG_TRANSFORM

    elif name == 'MFRGN_U1652':
        if config is None:
            raise ValueError("MFRGN_U1652 需要传入 config（dict 或 yaml 路径）；请从 utils.retrieval_init 传 region_config。")
        # 统一通过 get_MFRGN(config) 加载（内部会根据 retrieval_checkpoint_path 加载你本地训练的权重）
        model = get_MFRGN(config)
        model.eval()
        img_transform = IMG_TRANSFORM

    elif name == 'CAMP':
        # 还未实现 CAMP 的加载（按需再接）
        raise NotImplementedError("CAMP 模型加载尚未实现")

    else:
        raise ValueError(f"未知的检索方法: {method_name}")

    return model, img_transform
