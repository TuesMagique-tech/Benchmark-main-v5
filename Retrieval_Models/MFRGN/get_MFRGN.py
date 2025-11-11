# get_MFRGN.py
import os
import yaml
import torch
from .mfrgn_model import TimmModel_u


def _to_abs_path(p: str) -> str:
    """把相对路径转为以当前工作目录为基准的绝对路径。"""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(os.getcwd(), p))


def _clean_state_dict(sd):
    """去除 DataParallel 前缀 module.，并返回清洗后的 dict。"""
    if not isinstance(sd, dict):
        return sd
    keys = list(sd.keys())
    if any(k.startswith("module.") for k in keys):
        new_sd = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            new_sd[nk] = v
        return new_sd
    return sd


def get_MFRGN(config):
    """
    基于给定的配置创建并返回一个 MFRGN 模型（TimmModel_u）。
    `config` 可以是 YAML 文件路径或已解析的 dict/命名空间。

    约定（优先级从高到低）：
      1) YAML/配置中的 `MFRGN_CKPT_PATH`
      2) YAML/配置中的 `retrieval_checkpoint_path`（兼容旧键名）
      3) 环境变量 `CKPT_RETR`

    若以上都未提供，则报错；不再使用任何写死的默认 ckpt 路径。
    """
    # ---- 读取配置 ----
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    # ---- 解析模型构造参数 ----
    # 这些键与 train_university 保持一致，便于基线/训练同步
    def _get_attr(d, k, default=None):
        if isinstance(d, dict):
            return d.get(k, default)
        return getattr(d, k, default)

    model_name = _get_attr(cfg, "model", "convnext_base.fb_in22k_ft_in1k")
    img_size   = _get_attr(cfg, "img_size", 384)
    psm        = _get_attr(cfg, "psm", True)
    is_polar   = _get_attr(cfg, "is_polar", False)

    # ---- 构造模型（保持与训练时一致；是否加载 ImageNet 预训练由模型内部逻辑决定）----
    model = TimmModel_u(model_name, img_size=img_size, psm=psm, is_polar=is_polar)

    # ---- 解析 ckpt 路径（只认配置/环境，不做任何旧路径 fallback）----
    ckpt_path = None
    if isinstance(cfg, dict):
        ckpt_path = cfg.get("MFRGN_CKPT_PATH") or cfg.get("retrieval_checkpoint_path")
    else:
        ckpt_path = getattr(cfg, "MFRGN_CKPT_PATH", None) or getattr(cfg, "retrieval_checkpoint_path", None)

    if not ckpt_path:
        ckpt_path = os.environ.get("CKPT_RETR", "").strip()

    if not ckpt_path:
        raise RuntimeError(
            "[MFRGN] 未提供检索权重路径。请在 config.yaml 顶层添加\n"
            "  MFRGN_CKPT_PATH: <你的ckpt路径>\n"
            "或兼容键\n"
            "  retrieval_checkpoint_path: <你的ckpt路径>\n"
            "也可临时通过环境变量 CKPT_RETR 传入。"
        )

    ckpt_path = _to_abs_path(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[MFRGN] 未找到权重文件：{ckpt_path}")

    # ---- 兼容多种保存格式，统一成 state_dict 并清洗前缀 ----
    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = None
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict):
        # 直接就是 state_dict（键是字符串）
        state_dict = obj
    else:
        raise RuntimeError(f"[MFRGN] 非法的 ckpt 格式：{type(obj)} in {ckpt_path}")

    state_dict = _clean_state_dict(state_dict)

    # ---- 加载权重（strict=False 更稳健；打印缺失/多余键便于排查）----
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[MFRGN][WARN] load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) <= 20 and len(unexpected) <= 20:
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

    print(f"[MFRGN] Loaded ckpt: {os.path.abspath(ckpt_path)}")

    model.eval()
    return model

