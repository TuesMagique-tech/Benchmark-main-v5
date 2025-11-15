def train(train_config, model, dataloader, loss_function, optimizer, scheduler, scaler):
    """
    训练 1 个 epoch（支持梯度累积与 AMP）。
    关键点：
      - 仅在“真实更新步”（满足 grad_accum_steps 或最后一个 batch）才做 optimizer.step()
      - scheduler.step() 严格放在 optimizer.step() 之后且同样只在真实更新步调用
      - 可选梯度裁剪，与 AMP 联动（先 unscale_ 再 clip）
    """
    import torch
    from tqdm import tqdm
    import torch.nn.functional as F

    model.train()
    device = train_config.device

    accum_steps = int(getattr(train_config, "grad_accum_steps", 1))
    accum_steps = max(accum_steps, 1)

    running_loss = 0.0      # 一个累积窗口内的损失和（未除）
    epoch_loss_sum = 0.0    # 统计“每次权重更新”的平均损失之和
    num_updates = 0         # 本 epoch 共发生了多少次参数更新

    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(dataloader, ncols=120)
    for batch_idx, (query, reference, ids) in enumerate(bar):
        query = query.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)

        # DataParallel 兼容：logit_scale 在 module 上
        if torch.cuda.device_count() > 1 and len(getattr(train_config, "gpu_ids", (0,))) > 1:
            logit_scale = model.module.logit_scale.exp()
        else:
            logit_scale = model.logit_scale.exp()

        if scaler is not None:
            # AMP
            with torch.autocast("cuda"):
                feat1, feat2 = model(query, reference)
                loss = loss_function(feat1, feat2, logit_scale)
            (scaler.scale(loss) / accum_steps).backward()
        else:
            feat1, feat2 = model(query, reference)
            loss = loss_function(feat1, feat2, logit_scale)
            (loss / accum_steps).backward()

        running_loss += float(loss.item())

        # —— 是否到达一次“真实更新步”：进行 step/clip/scheduler/统计 ——
        is_update_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
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

            # 学习率调度器：必须在 optimizer.step() 之后；仅真实更新步调用一次
            if scheduler is not None:
                scheduler.step()

            # “一次权重更新”的平均损失（把累积窗口中的 loss 均摊）
            avg_loss_this_update = running_loss / accum_steps
            epoch_loss_sum += avg_loss_this_update
            num_updates += 1
            running_loss = 0.0

            # 进度条显示
            bar.set_postfix(ordered_dict={
                "loss": f"{avg_loss_this_update:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
            })

    # 返回“每次权重更新”的平均损失
    epoch_avg_loss = epoch_loss_sum / max(1, num_updates)
    return float(epoch_avg_loss)
