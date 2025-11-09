# get_MFRGN.py
import yaml
from .mfrgn_model import TimmModel_u
import os
import torch


def get_MFRGN(config):
    """
    基于给定的配置创建并返回一个 MFRGN 模型（TimmModel_u）。
    `config` 可以是指向 YAML 文件的路径，或一个配置字典。
    Baseline 评测阶段：不加载 ImageNet 预训练，转而加载本地 U1652 训练 ckpt。
    """
    # 读取配置
    if isinstance(config, str):
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    # 必要字段
    model_name = cfg.get('model', 'convnext_base.fb_in22k_ft_in1k')
    img_size   = cfg.get('img_size', 384)
    psm        = cfg.get('psm', True)
    is_polar   = cfg.get('is_polar', False)

    # 1) 构造模型（不加载 ImageNet 预训练）
    model = TimmModel_u(model_name, img_size=img_size, psm=psm, is_polar=is_polar)

    # 2) 加载本地 U1652 训练权重
    # 先从 config.yaml 顶层读取 retrieval_checkpoint_path；若无则再看环境变量 CKPT_RETR
    ckpt_path = cfg.get('retrieval_checkpoint_path', None)
    if not ckpt_path:
        ckpt_path = os.environ.get('CKPT_RETR', '')

    if not ckpt_path:
        raise RuntimeError(
            "[MFRGN] 未提供检索权重路径。"
            "请在 config.yaml 顶层添加 retrieval_checkpoint_path: <你的ckpt绝对路径>，"
            "或导出环境变量 CKPT_RETR。"
        )

    # 转绝对路径（相对路径则以当前工作目录为基准）
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.getcwd(), ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[MFRGN] 未找到权重文件：{ckpt_path}")

    # 兼容多种保存格式，统一成 state_dict
    obj = torch.load(ckpt_path, map_location='cpu')
    state_dict = obj.get('state_dict', obj.get('model', obj.get('model_state_dict', obj)))
    # 去掉 DataParallel 前缀
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        if any(k.startswith('module.') for k in keys):
            for k in keys:
                if k.startswith('module.'):
                    state_dict[k.replace('module.', '', 1)] = state_dict.pop(k)
    else:
        raise RuntimeError(f"[MFRGN] 非法的 ckpt 格式：{type(state_dict)} in {ckpt_path}")

    # 加载权重（严格匹配；若提示不匹配，再临时改成 strict=False 看缺失/多余键）
    model.load_state_dict(state_dict, strict=True)
    print(f"[MFRGN] Loaded ckpt: {ckpt_path}")

    model.eval()
    return model

