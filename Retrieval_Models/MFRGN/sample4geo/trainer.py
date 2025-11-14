# -*- coding: utf-8 -*-
import os
import time
import torch
# from torch.amp import autocast
from tqdm import tqdm
from .utils import AverageMeter
# from torch.cuda.amp import autocast
import torch.nn.functional as F
monitor = {}

# def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

#     # set model train mode
#     model.train()
    
#     losses = AverageMeter()
    
#     # wait before starting progress bar
#     time.sleep(0.1)
    
#     # Zero gradients for first step
#     optimizer.zero_grad(set_to_none=True)
    
#     step = 1
    
#     if train_config.verbose:
#         # bar = tqdm(dataloader, total=len(dataloader))
#         bar = tqdm(dataloader, total=len(dataloader), ncols=150, position=0, leave=True)
#     else:
#         bar = dataloader
    
#     # for loop over one epoch
#     for query, reference, ids in bar:
               
        # if scaler is not None:  # 混合精度模式
        #     with torch.cuda.amp.autocast():
        #         query = query.to(train_config.device)
        #         reference = reference.to(train_config.device)
        #         features1, features2 = model(query, reference)
        #         # 计算损失（DataParallel时需用 model.module 调用属性）
        #         # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
        #         #     loss = loss_function(features1, features2, model.module.logit_scale.exp())
        #         # else:
        #         #     loss = loss_function(features1, features2, model.logit_scale.exp())

        #         if len(train_config.gpu_ids) > 1:  # Using multiple GPUs (DataParallel)
        #             logit_scale_val = model.module.logit_scale.exp()
        #         else:  # Single GPU (no DataParallel)
        #             # loss = loss_function(features1, features2, model.logit_scale.exp())
        #             logit_scale_val = model.logit_scale.exp()
        #             # Compute InfoNCE loss in full precision
        #         loss = loss_function(features1.float(), features2.float(), logit_scale_val.float())

        #     losses.update(loss.item())  # Update running average of loss

        #     # 缩放后的反向传播
        #     scaler.scale(loss).backward()
        #     # （可选）梯度裁剪，在缩放梯度还原后进行
        #     if train_config.clip_grad:
        #         scaler.unscale_(optimizer)  # 将梯度恢复为未缩放状态
        #         torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
        #     # 用缩放后的梯度进行参数更新
        #     scaler.step(optimizer)
        #     scaler.update()           # 更新缩放因子
        #     optimizer.zero_grad()     # 清梯度，为下一次迭代做准备
        #     if scheduler is not None:
        #         scheduler.step()      # 更新学习率（如果使用逐步更新的调度器）
        # else:  # 常规精度模式
        #     query = query.to(train_config.device)
        #     reference = reference.to(train_config.device)
        #     features1, features2 = model(query, reference)
        #     if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
        #         loss = loss_function(features1, features2, model.module.logit_scale.exp())
        #     else:
        #         loss = loss_function(features1, features2, model.logit_scale.exp())
        #     loss.backward()
        #     if train_config.clip_grad:
        #         torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     if scheduler is not None:
        #         scheduler.step()
        
    #     if train_config.verbose:
    #         if len(optimizer.param_groups) > 1:
    #             monitor = {"loss": "{:.4f}".format(loss.item()),
    #                     "loss_avg": "{:.4f}".format(losses.avg),
    #                     "lr1" : "{:.6e}".format(optimizer.param_groups[0]['lr']),
    #                     "lr2" : "{:.6e}".format(optimizer.param_groups[1]['lr'])}
    #         else:
    #             monitor = {"loss": "{:.4f}".format(loss.item()),
    #                     "loss_avg": "{:.4f}".format(losses.avg),
    #                     "lr" : "{:.6e}".format(optimizer.param_groups[0]['lr']),}
            
    #         bar.set_postfix(ordered_dict=monitor)
        
    #     step += 1

    # if train_config.verbose:
    #     bar.close()

    # return losses.avg


def train(train_config, model, dataloader, loss_function, optimizer, scheduler, scaler):
    """
    训练 1 个 epoch（支持梯度累积与 AMP）。
    - 有效 batch = 2 * batch_size（卫星+无人机），再乘以 grad_accum_steps（梯度累积）
    - 仅在真正 optimizer.step() 时调用 scheduler.step()（且顺序在 optimizer 之后）
    """
    model.train()
    device = train_config.device

    accum_steps = int(getattr(train_config, "grad_accum_steps", 1))
    accum_steps = max(accum_steps, 1)

    running_loss = 0.0
    epoch_loss_sum = 0.0
    num_updates = 0
    monitor = {}

    # 第一次迭代前清梯度
    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(dataloader, ncols=120)
    for batch_idx, (query, reference, ids) in enumerate(bar):
        query = query.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)

        # ---- 前向 + InfoNCE（取 logit_scale，兼容 DataParallel）----
        if torch.cuda.device_count() > 1 and len(getattr(train_config, "gpu_ids", (0,))) > 1:
            logit_scale = model.module.logit_scale.exp()
        else:
            logit_scale = model.logit_scale.exp()

        if scaler is not None:
            with torch.autocast("cuda"):
                feat1, feat2 = model(query, reference)
                loss = loss_function(feat1, feat2, logit_scale)
            (loss / accum_steps).backward()
        else:
            feat1, feat2 = model(query, reference)
            loss = loss_function(feat1, feat2, logit_scale)
            (loss / accum_steps).backward()

        running_loss += float(loss.item())

        # ---- 到达一次“真实更新”的边界：做一次权重更新 ----
        is_update_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
        if not is_update_step:
            continue

        # 1) 梯度裁剪 + optimizer.step()
        if scaler is not None:
            if getattr(train_config, "clip_grad", None) is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
            scaler.step(optimizer)   # <<< 先 optimizer.step()
            scaler.update()
        else:
            if getattr(train_config, "clip_grad", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad)
            optimizer.step()         # <<< 先 optimizer.step()

        # 2) 再 scheduler.step()   （官方建议的正确顺序）
        if scheduler is not None:
            scheduler.step()

        # 3) 最后再清梯度，进入下一个累积窗口
        optimizer.zero_grad(set_to_none=True)

        # 统计 & 进度条
        avg_loss_this_update = running_loss / accum_steps
        epoch_loss_sum += avg_loss_this_update
        num_updates += 1
        running_loss = 0.0
        monitor = {
            "loss": f"{avg_loss_this_update:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
        }
        bar.set_postfix(ordered_dict=monitor)

    # 返回“每次真实更新窗口”的平均 loss
    return float(epoch_loss_sum / max(1, num_updates))


def predict(train_config, model, dataloader, is_autocast=True, input_id=1):
    model.eval()
    time.sleep(0.1)
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader), ncols=100, position=0, leave=True)
    else:
        bar = dataloader

    img_features_list = []
    ids_list = []
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

    if train_config.verbose:
        bar.close()
    return img_features, ids_list


def predict_dual(train_config, model, query_dataloader, reference_dataloader, is_autocast=True):
    model.eval()
    time.sleep(0.1)
    if train_config.verbose:
        bar = tqdm(query_dataloader, total=len(query_dataloader), ncols=100, position=0, leave=True)
    else:
        bar = query_dataloader

    img_features_list1, img_features_list2 = [], []
    ids_list = []
    reference_iter = iter(reference_dataloader)
    with torch.no_grad():
        for query, ids in bar:
            ids_list.append(ids)
            if is_autocast:
                with torch.autocast("cuda"):
                    query = query.to(train_config.device)
                    reference = next(reference_iter)[0].to(train_config.device)
                    img_feature1, img_feature2 = model(query, reference)
            else:
                query = query.to(train_config.device)
                reference = next(reference_iter)[0].to(train_config.device)
                img_feature1, img_feature2 = model(query, reference)

            if train_config.normalize_features:
                img_feature1 = F.normalize(img_feature1, dim=-1)
                img_feature2 = F.normalize(img_feature2, dim=-1)

            img_features_list1.append(img_feature1.to(torch.float32))
            img_features_list2.append(img_feature2.to(torch.float32))

        img_features1 = torch.cat(img_features_list1, dim=0)
        img_features2 = torch.cat(img_features_list2, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()
    return img_features1, img_features2, ids_list, ids_list