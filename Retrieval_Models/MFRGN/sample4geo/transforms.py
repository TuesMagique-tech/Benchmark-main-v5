import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms_train(image_size_sat, img_size_ground, ground_cutting=0, is_polar=False):
    """定义训练阶段的增广策略（分别针对卫星视角和地面/无人机视角图像）。"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 卫星视角图像的训练增广流水线
    sat_transforms_list = [
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
        A.OneOf([A.AdvancedBlur(p=1.0), A.Sharpen(p=1.0)], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25, max_height=int(0.2 * image_size_sat[0]), max_width=int(0.2 * image_size_sat[1]),
                            min_holes=10, min_height=int(0.1 * image_size_sat[0]), min_width=int(0.1 * image_size_sat[1]), p=1.0)
        ], p=0.3),
        A.RandomRotate90(p=1.0),  # 随机90度旋转（仅针对卫星图像，使模型对方位不敏感）
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    # 地面/无人机视角图像的训练增广流水线
    ground_transforms_list = [
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3, p=0.5),
        A.OneOf([A.AdvancedBlur(p=1.0), A.Sharpen(p=1.0)], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25, max_height=int(0.2 * img_size_ground[0]), max_width=int(0.2 * img_size_ground[1]),
                            min_holes=10, min_height=int(0.1 * img_size_ground[0]), min_width=int(0.1 * img_size_ground[1]), p=1.0)
        ], p=0.3),
        # 不对地面图像进行旋转增广（保留朝向信息）
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    sat_train_tf = A.Compose(sat_transforms_list)
    ground_train_tf = A.Compose(ground_transforms_list)
    return sat_train_tf, ground_train_tf

def get_transforms_val(image_size_sat, img_size_ground, ground_cutting=0, is_polar=False):
    """定义验证/测试阶段的增广策略（分别针对卫星视角和地面视角图像）。"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_list_sat = []
    transforms_list_ground = []
    # 如果需要裁剪地面图像顶部（如有天空部分），在前处理
    if ground_cutting > 0 and not is_polar:
        transforms_list_ground.append(A.CropAndPad(px_top=-ground_cutting, keep_size=False, p=1.0))
    # 调整图像尺寸到目标大小
    transforms_list_sat.append(A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0))
    transforms_list_ground.append(A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0))
    # 归一化和张量变换
    transforms_list_sat.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    transforms_list_ground.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    sat_val_tf = A.Compose(transforms_list_sat)
    ground_val_tf = A.Compose(transforms_list_ground)
    return sat_val_tf, ground_val_tf

def get_transforms_all(config: dict):
    """
    根据配置字典返回训练查询集增广、训练库集增广和验证集增广策略。
    """
    # 1. 提取图像尺寸相关配置
    if 'IMG_SIZE_SAT' in config and 'IMG_SIZE_GROUND' in config:
        image_size_sat = config['IMG_SIZE_SAT']  # 卫星图像目标尺寸 (H, W)
        img_size_ground = config['IMG_SIZE_GROUND']  # 地面/无人机图像目标尺寸 (H, W)
    elif 'IMG_SIZE' in config:
        size = config['IMG_SIZE']
        if isinstance(size, (list, tuple)):
            image_size_sat = tuple(size)
            img_size_ground = tuple(size)
        else:
            image_size_sat = (size, size)
            img_size_ground = (size, size)
    else:
        raise KeyError("配置缺少 IMG_SIZE 或 IMG_SIZE_SAT/IMG_SIZE_GROUND 参数")

    # 2. 提取其他增广参数，设置默认值避免缺失
    ground_cutting = config.get('GROUND_CUTTING', 0)  # 地面图像裁剪顶部像素（默认为0不裁剪）
    is_polar = config.get('IS_POLAR', False)           # 地面图像是否为极坐标展开（默认为False）
    # prob_flip 在 Albumentations 增广中不直接使用，由数据集类内部处理

    # 3. 获取训练集增广策略（分别针对卫星和地面/无人机图像）
    sat_train_tf, ground_train_tf = get_transforms_train(
        image_size_sat=image_size_sat,
        img_size_ground=img_size_ground,
        ground_cutting=ground_cutting,
        is_polar=is_polar
    )
    # 4. 获取验证集增广策略（分别针对卫星和地面图像）
    sat_val_tf, ground_val_tf = get_transforms_val(
        image_size_sat=image_size_sat,
        img_size_ground=img_size_ground,
        ground_cutting=ground_cutting,
        is_polar=is_polar
    )
    # **注意**：通常 U1652 数据集中 ground_cutting=0，因此 sat_val_tf 与 ground_val_tf 等效

    # 5. 根据任务返回所需的三个增广配置
    # 对于 University-1652 的 D2S（无人机视角查询 -> 卫星视角库）任务：
    train_query_transform = ground_train_tf    # 查询集（无人机视角）训练增广
    train_gallery_transform = sat_train_tf     # 库集（卫星视角）训练增广
    # 验证阶段统一使用卫星视角的增广（若 ground_cutting 不为0，可根据需要区分）
    val_transform = sat_val_tf

    return train_query_transform, train_gallery_transform, val_transform
