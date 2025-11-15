# -*- coding: utf-8 -*-
"""
InfoNCE with (optional) DHML-style memory bank and (optional) ICEL.

满足需求：
1) 在 __init__ 中建立两路特征记忆库（memory_bank1 / memory_bank2），用于提供长/短期难负样本；
2) 在 forward 中，每个 batch 的特征会追加进记忆库，并在相似度计算时把“当前对端特征 + 记忆库特征”拼接为负样本池；
3) 计算交叉熵时，只把前 N 列（当前 batch 的真实正样本列）作为正类，记忆库附加列仅作为负类参与 softmax；
4) 提供 reset_memory()/clear_memory() 供外部在每个 epoch 开始时清空记忆库（建议启用）；
5) ICEL 相关逻辑完全受 use_icel / lambda_icel 控制；默认关闭，不会影响 InfoNCE 主项。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(
        self,
        loss_function: nn.Module,
        device: torch.device,
        use_memory: bool = False,
        memory_size: int = 700,
        use_icel: bool = False,
        lambda_icel: float = 0.0,
        icel_threshold: float = 0.7,
        normalize: bool = True,
    ):
        """
        Args:
            loss_function: 一般为 nn.CrossEntropyLoss(label_smoothing=...)
            device       : 运行设备
            use_memory   : 是否启用记忆库（DHML）
            memory_size  : 记忆库最大条目数（滑动窗口）
            use_icel     : 是否启用 ICEL（默认关闭）
            lambda_icel  : ICEL 损失权重
            icel_threshold:ICEL 邻域阈值
            normalize    : 是否在 loss 内部做 L2 归一化
        """
        super().__init__()
        self.loss_function = loss_function
        self.device = device

        # DHML memory-bank 参数
        self.use_memory = bool(use_memory)
        self.memory_size = int(memory_size)

        # ICEL 参数（默认关闭）
        self.use_icel = bool(use_icel)
        self.lambda_icel = float(lambda_icel)
        self.icel_threshold = float(icel_threshold)

        self.normalize = bool(normalize)

        # 用 buffer 保存以便随模型迁移到 GPU；首次使用时会替换为正确维度
        self.register_buffer("memory_bank1", torch.zeros(0, 1), persistent=False)  # view1
        self.register_buffer("memory_bank2", torch.zeros(0, 1), persistent=False)  # view2

    # ====== 可选：在每个 epoch 开始时调用，以清空记忆库 ======
    @torch.no_grad()
    def reset_memory(self):
        if self.memory_bank1.numel():
            self.memory_bank1 = self.memory_bank1[:0]
        if self.memory_bank2.numel():
            self.memory_bank2 = self.memory_bank2[:0]

    # 别名，便于外部调用
    clear_memory = reset_memory

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) if self.normalize else x

    def _append_to_memory(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """
        将当前 batch 的两路特征追加到记忆库，并做滑动截断。
        feats1/feats2: [N, C]
        """
        with torch.no_grad():
            if self.memory_bank1.numel() == 0:
                # 首次：直接用当前 batch 初始化
                self.memory_bank1 = feats1.detach().clone()
                self.memory_bank2 = feats2.detach().clone()
            else:
                new_mem1 = torch.cat([self.memory_bank1, feats1.detach()], dim=0)
                new_mem2 = torch.cat([self.memory_bank2, feats2.detach()], dim=0)
                # 仅保留最近 memory_size 条（FIFO 队列）
                if new_mem1.size(0) > self.memory_size:
                    new_mem1 = new_mem1[-self.memory_size:]
                    new_mem2 = new_mem2[-self.memory_size:]
                self.memory_bank1 = new_mem1
                self.memory_bank2 = new_mem2

    def forward(
        self,
        image_features1: torch.Tensor,
        image_features2: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features1: [N, C]（例如：drone）
            image_features2: [N, C]（例如：satellite）
            logit_scale    : 标量张量或 float（CLIP 风格温度缩放）

        Returns:
            total_loss     : 标量 loss（InfoNCE 主项 + 可选 ICEL 项）
        """
        assert image_features1.dim() == 2 and image_features2.dim() == 2, \
            "image_features must be 2D (N, C)."
        N1, C1 = image_features1.shape
        N2, C2 = image_features2.shape
        assert C1 == C2, f"Feature dim mismatch: {C1} vs {C2}"
        assert N1 == N2, f"Batch size mismatch: {N1} vs {N2}"
        N = N1

        # 归一化（若未在模型中完成）
        f1 = self._maybe_normalize(image_features1)
        f2 = self._maybe_normalize(image_features2)

        # 维护/更新记忆库（DHML）
        if self.use_memory:
            self._append_to_memory(f1, f2)

        # 组装“对端当前批 + 记忆库”作为候选集
        if self.use_memory and self.memory_bank2.numel() > 0:
            feats2_all = torch.cat([f2, self.memory_bank2], dim=0)  # for 1->2
        else:
            feats2_all = f2

        if self.use_memory and self.memory_bank1.numel() > 0:
            feats1_all = torch.cat([f1, self.memory_bank1], dim=0)  # for 2->1
        else:
            feats1_all = f1

        # 相似度（温度缩放）
        if torch.is_tensor(logit_scale):
            scale = logit_scale.view(-1)[0]
        else:
            scale = torch.tensor(float(logit_scale), device=f1.device)

        # logits_12: [N, (N + M2)], logits_21: [N, (N + M1)]
        logits_12 = scale * (f1 @ feats2_all.t())
        logits_21 = scale * (f2 @ feats1_all.t())

        # 只监督前 N 列（真实正样本列）；附加的记忆库列仅作为负样本进入 softmax
        targets = torch.arange(N, device=f1.device, dtype=torch.long)
        loss_12 = self.loss_function(logits_12, targets)  # drone->sat
        loss_21 = self.loss_function(logits_21, targets)  # sat->drone
        info_nce_loss = 0.5 * (loss_12 + loss_21)

        # ====== (可选) ICEL —— 完全受 use_icel / lambda_icel 控制，默认不计算 ======
        icel_loss = torch.tensor(0.0, device=f1.device)
        if self.use_icel and self.lambda_icel > 0.0:
            # 如需启用，请在此处实现邻域一致性损失（根据阈值 self.icel_threshold）
            # 当前按你的策略：禁用或在训练脚本里分段调度启用
            pass

        total_loss = info_nce_loss + self.lambda_icel * icel_loss
        return total_loss
