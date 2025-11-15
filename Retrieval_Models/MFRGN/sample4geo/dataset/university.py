# -*- coding: utf-8 -*-
"""
University-1652 数据集读取与增广定义（卫星/无人机）
按需求点完成如下改动：
1) 移除卫星端 RandomRotate90(p=1.0)
2) 无人机端保留视角扰动：RandomScale(±20%) + Rotate(±10°)
3) 天气扰动 OneOf 概率统一为 p=0.10（仅无人机端）
4) 局部遮挡 OneOf 概率为 p=0.20，GridDropout ratio=0.30
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Set

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UniversityDataset(Dataset):
    """
    通用 University-1652 读取器。
    需要传入：
        images: List[str]            - 图像路径列表
        sample_ids: List[int]        - 与 images 等长的样本ID（用于匹配/评测）
        mode: str                    - "sat" 或 "drone"，仅用于少量分支逻辑与统计
        transforms: albumentations.Compose
        given_sample_ids: Optional[Set[int]] - 若提供，则不在集合中的样本将被标记为 label=-1
    """
    def __init__(
        self,
        images: List[str],
        sample_ids: List[int],
        mode: str,
        transforms: Optional[A.Compose] = None,
        given_sample_ids: Optional[Set[int]] = None,
    ):
        assert len(images) == len(sample_ids), "images 与 sample_ids 长度必须一致"
        self.images = images
        self.sample_ids = sample_ids
        self.mode = mode
        self.transforms = transforms
        self.given_sample_ids = given_sample_ids

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Fail to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === 如果你曾经在 sat 分支做过四象限拼接旋转的数据增强，这里保持注释状态 ===
        # if self.mode == "sat":
        #     img90  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #     img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #     img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
        #     img_0_90    = np.concatenate([img, img90], axis=1)
        #     img_180_270 = np.concatenate([img180, img270], axis=1)
        #     img = np.concatenate([img_0_90, img_180_270], axis=0)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        label = int(sample_id)
        if self.given_sample_ids is not None and (sample_id not in self.given_sample_ids):
            label = -1

        return img, label

    def get_sample_ids(self) -> Set[int]:
        return set(self.sample_ids)


def get_transforms(
    img_size: Tuple[int, int],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
):
    """
    返回 (val_transforms, train_sat_transforms, train_drone_transforms)

    重要改动：
    - 卫星端不再做 RandomRotate90(p=1.0)，以避免与配对 UAV 的方向先验相冲突
    - UAV 端保留轻量视角扰动（RandomScale + Rotate）
    - 天气扰动仅在 UAV 端，OneOf 概率为 0.10
    - 局部遮挡 OneOf 概率为 0.20（强度偏轻，减少学习难度）
    """

    # ===== 验证/评测：只做 Resize + Normalize =====
    val_transforms = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # ===== 训练：卫星（不再旋转）=====
    train_sat_transforms = A.Compose(
        [
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),

            # 颜色抖动（适中强度）
            A.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5
            ),

            # 锐化/模糊 二选一
            A.OneOf(
                [
                    A.AdvancedBlur(p=1.0),
                    A.Sharpen(p=1.0),
                ],
                p=0.30,
            ),

            # 局部遮挡 二选一（强度偏轻）
            A.OneOf(
                [
                    A.GridDropout(ratio=0.30, p=1.0),
                    A.CoarseDropout(
                        max_holes=25,
                        max_height=int(0.20 * img_size[0]),
                        max_width=int(0.20 * img_size[1]),
                        min_holes=10,
                        min_height=int(0.10 * img_size[0]),
                        min_width=int(0.10 * img_size[1]),
                        p=1.0,
                    ),
                ],
                p=0.20,
            ),

            # 关键变更：**删除** RandomRotate90(p=1.0)
            # A.RandomRotate90(p=1.0),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # ===== 训练：无人机（视角 + 环境）=====
    iaa_weather_list = [
        A.RandomFog(fog_coef_lower=0.10, fog_coef_upper=0.30, p=1.0),
        A.RandomRain(brightness_coefficient=0.90, drop_width=1, blur_value=3, p=1.0),
        A.RandomSnow(snow_point_lower=0.10, snow_point_upper=0.30, brightness_coeff=2.0, p=1.0),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1.0),
    ]

    train_drone_transforms = A.Compose(
        [
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),

            # 视角扰动（放在 Resize 之前，避免越界）
            A.RandomScale(scale_limit=0.20, p=0.50),   # ±20% 缩放
            A.Rotate(limit=10, p=0.50),                # ±10° 旋转

            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),

            # 颜色抖动（适中强度）
            A.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5
            ),

            # 锐化/模糊 二选一
            A.OneOf(
                [
                    A.AdvancedBlur(p=1.0),
                    A.Sharpen(p=1.0),
                ],
                p=0.30,
            ),

            # 局部遮挡 二选一（强度偏轻）
            A.OneOf(
                [
                    A.GridDropout(ratio=0.30, p=1.0),
                    A.CoarseDropout(
                        max_holes=25,
                        max_height=int(0.20 * img_size[0]),
                        max_width=int(0.20 * img_size[1]),
                        min_holes=10,
                        min_height=int(0.10 * img_size[0]),
                        min_width=int(0.10 * img_size[1]),
                        p=1.0,
                    ),
                ],
                p=0.20,
            ),

            # 环境扰动（**总体概率 0.10**）
            A.OneOf(iaa_weather_list, p=0.10),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return val_transforms, train_sat_transforms, train_drone_transforms
