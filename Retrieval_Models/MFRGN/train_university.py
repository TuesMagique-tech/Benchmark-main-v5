import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.append("/home/chunyu/workspace/Benchmark-main-v5/Retrieval_Models/MFRGN")

import cv2
import time
import shutil
import torch
import yaml  # 新增：用于读取配置文件
import math
import albumentations as A
from dataclasses import dataclass
from torch.utils.data import DataLoader

# ---- AMP GradScaler (new API first, fallback to old) ----
try:
    from torch.amp import GradScaler  # PyTorch ≥ 2.0
    _SCALER_NEW_API = True
except Exception:
    from torch.cuda.amp import GradScaler  # 兼容旧版本
    _SCALER_NEW_API = False

from transformers import (
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# 可见 GPU（不与 config.gpu_ids 冲突；DataParallel 用 config.gpu_ids 选择）
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from sample4geo.dataset.university import (
    U1652DatasetEval,
    U1652DatasetTrain,
    get_transforms,
)
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.loss import InfoNCE

# 检索主干（MFRGN 封装）
from Retrieval_Models.MFRGN.mfrgn_model import TimmModel_u

from typing import Union  # 使用 Union[str, None] 描述可选路径


@dataclass
class Configuration:
    # ==================== 模型 ====================
    net: str = 'u652-D2S'
    model: str = 'convnext_base.fb_in22k_ft_in1k'
    is_polar: bool = False
    psm: bool = True
    img_size: int = 384
    pretrained: bool = True

    # 输入尺寸（按 ImageNet 均值方差）
    image_size_sat: tuple = (img_size, img_size)
    image_size_ground: tuple = (img_size, img_size)

    # ==================== 训练 ====================
    mixed_precision: bool = True
    custom_sampling: bool = True
    seed: int = 1

    # --------- 【微调参数：已按方案改好】 ---------
    epochs: int = 15                      # 微调轮数：10~15；此处设 15
    batch_size: int = 16                  # 有效 batch = 2 * batch_size（卫星 + 无人机）
    grad_accum_steps: int = 6             # 梯度累积，保持与你之前一致
    verbose: bool = True
    gpu_ids: tuple = (0,)                 # DataParallel 的设备列表（单卡保持 (0,)）

    # ==================== 评测 ====================
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1

    # ==================== 优化器 ====================
    clip_grad: Union[float, None] = 10.0  # 梯度裁剪阈值
    decay_exclue_bias: bool = True        # AdamW 对 bias/Norm 不衰减

    # 主干梯度检查点
    grad_checkpointing: bool = True

    # ==================== 损失 ====================
    label_smoothing: float = 0.1

    # ==================== 学习率/调度 ====================
    lr: float = 3e-5                      # 【微调 LR】小步长，稳增 Recall
    scheduler: str = "polynomial"           # 【微调调度】恒定学习率（可改 "polynomial"）
    warmup_epochs: float = 0.0            # 【微调不预热】0.0
    lr_end: float = 1e-5                  # 若改用 polynomial 时的终值

    # ==================== 数据集 ====================
    dataset: str = 'U1652-D2S'
    data_folder: str = "/home/chunyu/workspace/University-Release"

    # ==================== 图像增强 ====================
    prob_flip: float = 0.5

    # ==================== 路径 ====================
    model_path: str = "checkpoints/university"

    # 训练前零样本评测
    zero_shot: bool = False

    # 【从检查点恢复】—— 已填入你的 best e6 权重路径
    checkpoint_start: Union[str, None] = (
        "checkpoints/university/convnext_base.fb_in22k_ft_in1k/"
        "u652-D2S_U1652-D2S_11-12-21-53-47/weights_e6_92.7302.pth"
    )

    # ==================== DataLoader 线程 ====================
    num_workers: int = 0

    # ==================== 设备 ====================
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


# 初始化配置
config = Configuration()

# 路径就绪
if config.dataset == 'U1652-D2S':
    config.query_folder_train = os.path.join(config.data_folder, 'train/satellite')
    config.gallery_folder_train = os.path.join(config.data_folder, 'train/drone')
    config.query_folder_test = os.path.join(config.data_folder, 'test/query_drone')
    config.gallery_folder_test = os.path.join(config.data_folder, 'test/gallery_satellite')
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = os.path.join(config.data_folder, 'train/satellite')
    config.gallery_folder_train = os.path.join(config.data_folder, 'train/drone')
    config.query_folder_test = os.path.join(config.data_folder, 'test/query_satellite')
    config.gallery_folder_test = os.path.join(config.data_folder, 'test/gallery_drone')


if __name__ == '__main__':
    # 输出目录与日志
    model_path = "{}/{}/{}_{}_{}".format(
        config.model_path, config.model, config.net, config.dataset, time.strftime("%m-%d-%H-%M-%S")
    )
    os.makedirs(model_path, exist_ok=True)

    # 备份当前训练脚本
    src_file = os.path.abspath(__file__)
    dst_file = os.path.join(model_path, "train.py")
    shutil.copyfile(src_file, dst_file)

    # 日志：同时写控制台与文件
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    # 随机种子、线程等
    setup_system(vars(config))

    # CuDNN 开关
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.cudnn_deterministic

    # -----------------------------------------------------------------------------
    # 模型
    # -----------------------------------------------------------------------------
    print(f"\nModel: {config.model}")

    # （读取全局 config.yaml 指定的 ConvNeXt 预训练权重）
    pretrained_backbone_path = None
    cfg_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "config.yaml"))
    if os.path.isfile(cfg_file):
        with open(cfg_file, 'r') as f:
            cfg_all = yaml.safe_load(f)
        convnext_pretrain = cfg_all.get("convnext_pretrain", "")
        if convnext_pretrain:
            if not os.path.isabs(convnext_pretrain):
                convnext_pretrain = os.path.join(os.path.dirname(cfg_file), convnext_pretrain)
            pretrained_backbone_path = convnext_pretrain
            print(f"[Info] Using ConvNeXt pretrained weights: {pretrained_backbone_path}")

    # 初始化 MFRGN 模型（指定预训练骨干权重路径）
    model = TimmModel_u(
        config.model,
        config.img_size,
        psm=config.psm,
        is_polar=config.is_polar,
        pretrained=config.pretrained,
        pretrained_backbone_path=pretrained_backbone_path
    )

    # 打开梯度检查点
    if config.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        try:
            model.set_grad_checkpointing(True)
            print("Grad checkpointing: ENABLED")
        except Exception as e:
            print("Grad checkpointing set failed, continue without:", e)

    # 打印 Data Config（若实现）
    if hasattr(model, "get_config"):
        data_config = model.get_config()
        print(f"Model Data Config: {data_config}")

    # 归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size_sat = config.image_size_sat
    image_size_ground = config.image_size_ground

    # ---------- 从检查点恢复（已存在的逻辑，保持不变） ----------
    if config.checkpoint_start is not None:
        print("Starting from checkpoint:", config.checkpoint_start)
        checkpoint = torch.load(config.checkpoint_start, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(config.gpu_ids))
    model = model.to(config.device)

    print(f"\nImage Size Satellite: {image_size_sat}")
    print(f"Image Size Ground: {image_size_ground}")
    print(f"Mean: {mean}")
    print(f"Std: {std}\n")

    # -----------------------------------------------------------------------------
    # 数据
    # -----------------------------------------------------------------------------
    img_size_tuple = (config.img_size, config.img_size)
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
        img_size_tuple, mean=mean, std=std
    )

    # 训练集（卫星=query，无人机=gallery）
    train_dataset = U1652DatasetTrain(
        query_folder=config.query_folder_train,
        gallery_folder=config.gallery_folder_train,
        transforms_query=train_sat_transforms,
        transforms_gallery=train_drone_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=not config.custom_sampling,
        pin_memory=True,
    )

    # 测试/评测
    query_dataset_test = U1652DatasetEval(
        data_folder=config.query_folder_test, mode="query", transforms=val_transforms
    )
    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    gallery_dataset_test = U1652DatasetEval(
        data_folder=config.gallery_folder_test,
        mode="gallery",
        transforms=val_transforms,
        sample_ids=query_dataset_test.get_sample_ids(),
        gallery_n=config.eval_gallery_n,
    )
    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Query Images (Test set): {len(query_dataset_test)}")
    print(f"Gallery Images (Test set): {len(gallery_dataset_test)}")

    # -----------------------------------------------------------------------------
    # 损失（InfoNCE 包含 CrossEntropy + label smoothing）
    # -----------------------------------------------------------------------------
    base_loss = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=base_loss, device=config.device)

    # 混合精度 GradScaler
    scaler = GradScaler("cuda") if (config.mixed_precision and _SCALER_NEW_API) else (
             GradScaler(enabled=config.mixed_precision) )

    # -----------------------------------------------------------------------------
    # 优化器
    # -----------------------------------------------------------------------------
    if config.decay_exclue_bias:
        param_groups = list(model.named_parameters())
        no_decay = ("bias", "LayerNorm.bias")
        optimizer_parameters = [
            {
                "params": [p for n, p in param_groups if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_groups if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # -----------------------------------------------------------------------------
    # 学习率调度（微调：constant / 可切 polynomial）
    # -----------------------------------------------------------------------------
    steps_per_epoch = math.ceil(len(train_dataloader) / config.grad_accum_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(steps_per_epoch * config.warmup_epochs)

    if config.scheduler == "polynomial":
        print(f"\nScheduler: polynomial – max LR: {config.lr} – end LR: {config.lr_end}")
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=total_steps,
            lr_end=config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )
    elif config.scheduler == "cosine":
        print(f"\nScheduler: cosine – max LR: {config.lr}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "constant":
        print(f"\nScheduler: constant – max LR: {config.lr}")
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    else:
        scheduler = None

    print(f"Warmup Epochs: {config.warmup_epochs} – Warmup Steps: {warmup_steps}")
    print(f"Train Epochs: {config.epochs} – Total Train Steps: {total_steps}")

    # -----------------------------------------------------------------------------
    # （可选）训练前零样本评测
    # -----------------------------------------------------------------------------
    if config.zero_shot:
        print(f"\n{'-'*30}[Zero Shot]{'-'*30}")
        _ = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True,
        )

    # -----------------------------------------------------------------------------
    # 训练循环
    # -----------------------------------------------------------------------------
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    best_score = 0.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'-'*30}[Epoch: {epoch}]{'-'*30}")

        train_loss = train(
            config,
            model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Train Loss = {train_loss:.3f}, LR = {current_lr:.6f}")

        # 周期性评测
        if (epoch % config.eval_every_n_epoch == 0) or (epoch == config.epochs):
            print(f"\n{'-'*30}[Evaluate]{'-'*30}")
            r1_test = evaluate(
                config=config,
                model=model,
                query_loader=query_dataloader_test,
                gallery_loader=gallery_dataloader_test,
                ranks=[1, 5, 10],
                step_size=1000,
                cleanup=True,
            )

            # 保存 best
            if r1_test > best_score:
                best_score = r1_test
                best_epoch = epoch
                save_path = f"{model_path}/weights_e{epoch}_{r1_test:.4f}.pth"
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)

        # 每个 epoch 重新打乱（自定义采样时）
        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    # 最终权重
    final_path = f"{model_path}/weights_end.pth"
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)

    # 打印路径
    if best_epoch > 0:
        print(f"\nBest model (Recall@1 = {best_score:.4f}) saved at: {model_path}/weights_e{best_epoch}_{best_score:.4f}.pth")
    else:
        print("\nNo improvement in Recall@1 during training; no best model checkpoint saved.")
    print(f"Final model weights saved at: {final_path}")
