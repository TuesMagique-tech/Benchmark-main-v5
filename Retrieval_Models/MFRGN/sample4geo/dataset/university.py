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

import copy
import random
import time
from tqdm import tqdm

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

def get_data(path: str):
    """
    遍历给定路径下的文件夹，收集每个类别文件夹内的文件名列表。
    返回一个字典：{class_id: {"path": class_path, "files": [file1, file2, ...]}}
    """
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            class_path = os.path.join(root, name)
            # 收集该类别文件夹中的所有文件名
            file_list = []
            for _, _, files_in_dir in os.walk(class_path, topdown=False):
                file_list.extend(files_in_dir)
                break  # 只需要当前文件夹下的文件列表，不递归子目录
            data[name] = {"path": class_path, "files": file_list}
    return data

class U1652DatasetTrain(Dataset):
    def __init__(
        self,
        query_folder: str,
        gallery_folder: str,
        transforms_query: Optional[A.Compose] = None,
        transforms_gallery: Optional[A.Compose] = None,
        prob_flip: float = 0.5,
        shuffle_batch_size: int = 128,
    ):
        super().__init__()
        # 加载查询集（卫星）和检索集（无人机）的图片文件
        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        # 仅保留同时存在于 query 和 gallery 的类别
        self.ids = sorted(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        # 准备所有 (class_id, query_img_path, gallery_img_path) 对
        self.pairs = []
        for idx in self.ids:
            # Query 集合每个类别假设只有一张图像（取文件夹中的第一张）
            if len(self.query_dict[idx]["files"]) == 0:
                continue
            query_img_path = os.path.join(self.query_dict[idx]["path"], self.query_dict[idx]["files"][0])
            # Gallery 集合使用该类别文件夹中的所有图像
            gallery_path = self.gallery_dict[idx]["path"]
            for g_file in self.gallery_dict[idx]["files"]:
                gallery_img_path = os.path.join(gallery_path, g_file)
                self.pairs.append((idx, query_img_path, gallery_img_path))
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        # 初始化 samples 列表（稍后可调用 shuffle() 打乱）
        self.samples = copy.deepcopy(self.pairs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx, query_img_path, gallery_img_path = self.samples[index]
        # 读取图像
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB) if query_img is not None else None
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB) if gallery_img is not None else None
        if query_img is None or gallery_img is None:
            missing_path = query_img_path if query_img is None else gallery_img_path
            raise FileNotFoundError(f"Fail to read image: {missing_path}")
        # 随机水平翻转（同时应用于卫星和无人机图像）
        if np.random.rand() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)
        # 应用增广变换
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        # 返回图像张量和类别标签
        label = int(idx) if isinstance(idx, str) else idx
        return query_img, gallery_img, label

    def shuffle(self):
        """
        自定义 shuffle，使每个 batch 内不重复采样相同 class_id。
        """
        print("\nShuffle Dataset:")
        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)
        pairs_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []
        break_counter = 0
        pbar = tqdm()
        while True:
            if pair_pool:
                class_id, q_path, g_path = pair_pool.pop(0)
            else:
                break
            if class_id not in idx_batch and (class_id, q_path, g_path) not in pairs_epoch:
                idx_batch.add(class_id)
                current_batch.append((class_id, q_path, g_path))
                pairs_epoch.add((class_id, q_path, g_path))
                break_counter = 0
            else:
                if (class_id, q_path, g_path) not in pairs_epoch:
                    pair_pool.append((class_id, q_path, g_path))
                break_counter += 1
            if break_counter >= 512:
                break
            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch.clear()
                current_batch = []
            pbar.update(1)
        pbar.close()
        time.sleep(0.3)
        self.samples = batches
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        if len(self.samples) > 0:
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

class U1652DatasetEval(UniversityDataset):
    def __init__(
        self,
        data_folder: str,
        mode: str,
        transforms: Optional[A.Compose] = None,
        sample_ids: Optional[Set[int]] = None,
        gallery_n: int = -1,
    ):
        # 收集 data_folder 下所有图像路径和对应的类别 ID
        data_dict = get_data(data_folder)
        class_ids = list(data_dict.keys())
        images = []
        ids = []
        for cid in class_ids:
            files = data_dict[cid]["files"]
            if gallery_n is not None and gallery_n > 0 and mode == "gallery":
                # 若设置了 gallery_n，则每个类别最多只保留 gallery_n 张图像
                if len(files) > gallery_n:
                    files = sorted(files)
                    files = files[:gallery_n]
            for fname in files:
                images.append(os.path.join(data_dict[cid]["path"], fname))
                ids.append(int(cid) if cid.isdigit() else cid)
        # 根据路径推断 domain，用于选择增广方案：包含 "drone" 则视为无人机图像，否则视为卫星图像
        domain = "drone" if "drone" in data_folder.lower() else "sat"
        super().__init__(images=images, sample_ids=ids, mode=domain, transforms=transforms, given_sample_ids=sample_ids)




