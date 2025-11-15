# Retrieval_Models/MFRGN/sample4geo/trainer.py

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 说明：去掉了对 .utils 的模块级导入，避免 evaluate <-> trainer 循环依赖
monitor = {}

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    """
    训练 1 个 epoch（支持梯度累积与 AMP）。
    约束：
      - 仅在“到达累积边界/最后一个 batch”时执行：clip -> optimizer.step() -> scheduler.step()
      - 其它小步只做 backward 累积
      - 修正最后一个未满累积窗口的平均 loss 计算
    """
    model.train()
    device = train_config.device

    # ===== 累积相关 =====
    accum_steps = int(getattr(train_config, "grad_accum_steps", 1))
    accum_steps = max(accum_steps, 1)

    running_loss = 0.0     # 当前累积窗口内的原始 loss 累加（未除以 accum）
    epoch_loss_sum = 0.0   # 按“每次权重更新”的平均 loss 进行累计
    num_updates = 0        # 本 epoch 参数更新次数

    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(dataloader, ncols=120) if getattr(train_config, "verbose", True) else dataloader

    for batch_idx, (query, reference, ids) in enumerate(bar):
        query = query.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)

        # DataParallel 兼容：logit_scale 在 module 上
        if torch.cuda.device_count() > 1 and len(getattr(train_config, "gpu_ids", (0,))) > 1:
            logit_scale = model.module.logit_scale.exp()
        else:
            logit_scale = model.logit_scale.exp()

        # ===== 前向 & 反向（支持 AMP）=====
        if scaler is not None:
            with torch.autocast("cuda"):
                feat1, feat2 = model(query, reference)
                loss = loss_function(feat1, feat2, logit_scale)
            scaler.scale(loss / accum_steps).backward()
        else:
            feat1, feat2 = model(query, reference)
            loss = loss_function(feat1, feat2, logit_scale)
            (loss / accum_steps).backward()

        running_loss += float(loss.item())

        # ===== 到达一次“真实更新”的边界 =====
        is_last_batch = (batch_idx + 1) == len(dataloader)
        is_update_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch
        if is_update_step:
            if scaler is not None:
                if getattr(train_config, "clip_grad", None) is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if getattr(train_config, "clip_grad", None) is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # 只在真实 step 后更新 scheduler（修复顺序告警）
            if scheduler is not None:
                scheduler.step()

            # 统计当前“更新窗口”的平均 loss（最后未满窗口用实际步数）
            steps_in_window = accum_steps if ((batch_idx + 1) % accum_steps == 0) else ((batch_idx % accum_steps) + 1)
            avg_loss_this_update = running_loss / max(1, steps_in_window)
            epoch_loss_sum += avg_loss_this_update
            num_updates += 1
            running_loss = 0.0

            if getattr(train_config, "verbose", True):
                bar.set_postfix(ordered_dict={
                    "loss": f"{avg_loss_this_update:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                })

    if getattr(train_config, "verbose", True):
        try:
            bar.close()
        except Exception:
            pass

    epoch_avg_loss = epoch_loss_sum / max(1, num_updates)
    return float(epoch_avg_loss)


def predict(train_config, model, dataloader, is_autocast=True, input_id=1):
    model.eval()
    time.sleep(0.1)
    bar = tqdm(dataloader, total=len(dataloader), ncols=100, position=0, leave=True) \
        if getattr(train_config, "verbose", True) else dataloader

    img_features_list, ids_list = [], []
    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)
            if is_autocast:
                with torch.autocast("cuda"):
                    img = img.to(train_config.device)
                    img_feature = model(img, input_id=input_id)
            else:
                img = img.to(train_config.device)
                img_feature = model(img, input_id=input_id)

            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features_list.append(img_feature.to(torch.float32))

        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if getattr(train_config, "verbose", True):
        bar.close()

    return img_features, ids_list


def predict_dual(train_config, model, query_dataloader, reference_dataloader, is_autocast=True):
    model.eval()
    time.sleep(0.1)
    bar = tqdm(query_dataloader, total=len(query_dataloader), ncols=100, position=0, leave=True) \
        if getattr(train_config, "verbose", True) else query_dataloader

    img_features_list1, img_features_list2, ids_list = [], [], []
    reference_iter = iter(reference_dataloader)
    with torch.no_grad():
        for query, ids in bar:
            ids_list.append(ids)
            reference = next(reference_iter)[0]

            if is_autocast:
                with torch.autocast("cuda"):
                    query = query.to(train_config.device)
                    reference = reference.to(train_config.device)
                    img_feature1, img_feature2 = model(query, reference)
            else:
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                img_feature1, img_feature2 = model(query, reference)

            if train_config.normalize_features:
                img_feature1 = F.normalize(img_feature1, dim=-1)
                img_feature2 = F.normalize(img_feature2, dim=-1)

            img_features_list1.append(img_feature1.to(torch.float32))
            img_features_list2.append(img_feature2.to(torch.float32))

        img_features1 = torch.cat(img_features_list1, dim=0)
        img_features2 = torch.cat(img_features_list2, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if getattr(train_config, "verbose", True):
        bar.close()

    return img_features1, img_features2, ids_list, ids_list
