# Retrieval_Models/MFRGN/sample4geo/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn  # 保留以兼容现有环境

class InfoNCE(nn.Module):
    """
    InfoNCE with Dual-View Memory Banks (DHML-style) + ICEL (cross-view neighbor consistency).
    - 为两个模态各维护一个 FIFO memory bank：memory_bank1 (view-1/UAV), memory_bank2 (view-2/SAT)
    - 计算对比相似度时，将历史特征拼接为额外负样本，增强难负、延缓过快收敛
    - 叠加 ICEL：利用对端 memory bank 中的最近邻建立跨模态一致性约束，进一步对齐分布
    接口保持与原版一致：forward(image_features1, image_features2, logit_scale) -> 标量 loss
    """

    def __init__(
        self,
        loss_function,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_memory: bool = True,
        memory_size: int = 1024,
        use_icel: bool = True,
        lambda_icel: float = 0.5,
        icel_threshold: float = 0.5,
    ):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

        # ---- DHML / Memory Bank 开关与容量 ----
        self.use_memory = bool(use_memory)
        self.memory_size = int(memory_size)

        # ---- ICEL 设置 ----
        self.use_icel = bool(use_icel)
        self.lambda_icel = float(lambda_icel)
        self.icel_threshold = float(icel_threshold)

        # 延迟初始化（在第一次 forward 时，根据特征维度与设备创建）
        self.memory_bank1 = None  # [M, C] for view-1
        self.memory_bank2 = None  # [M, C] for view-2
        self.mem_ptr1 = 0
        self.mem_ptr2 = 0
        self.curr_size1 = 0
        self.curr_size2 = 0

    @torch.no_grad()
    def _ensure_banks(self, C1: int, C2: int, device1: torch.device, device2: torch.device, dtype1, dtype2):
        if self.memory_bank1 is None:
            self.memory_bank1 = torch.zeros((self.memory_size, C1), device=device1, dtype=dtype1)
            self.mem_ptr1 = 0
            self.curr_size1 = 0
        if self.memory_bank2 is None:
            self.memory_bank2 = torch.zeros((self.memory_size, C2), device=device2, dtype=dtype2)
            self.mem_ptr2 = 0
            self.curr_size2 = 0

    @torch.no_grad()
    def _enqueue(self, bank: torch.Tensor, feats: torch.Tensor, ptr: int, curr_size: int):
        """
        将当前 batch 特征写入环形缓冲队列（FIFO）。
        bank: [M, C]；feats: [B, C] (已归一化, 不回传梯度写入)
        返回：ptr, curr_size
        """
        B = feats.size(0)
        M = self.memory_size
        if B >= M:
            bank.copy_(feats[-M:].detach())
            return 0, M

        end = (ptr + B) % M
        if curr_size < M:
            # 未写满
            length = min(B, M - ptr)
            bank[ptr:ptr + length] = feats[:length].detach()
            if length < B:
                remain = B - length
                bank[0:remain] = feats[length:].detach()
                end = remain
            curr_size = min(M, curr_size + B)
        else:
            # 已满，循环覆盖
            length = min(B, M - ptr)
            bank[ptr:ptr + length] = feats[:length].detach()
            if length < B:
                remain = B - length
                bank[0:remain] = feats[length:].detach()
                end = remain
        return end, curr_size

    def forward(self, image_features1: torch.Tensor,
                image_features2: torch.Tensor,
                logit_scale: torch.Tensor) -> torch.Tensor:
        """
        image_features1: [B, C1]  (e.g., UAV view)
        image_features2: [B, C2]  (e.g., SAT view)
        logit_scale:     标量或 [1]
        """
        # 1) 特征归一化
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # 2) 内存队列准备（不影响计算图）
        if self.use_memory:
            with torch.no_grad():
                self._ensure_banks(
                    image_features1.shape[1], image_features2.shape[1],
                    image_features1.device, image_features2.device,
                    image_features1.dtype, image_features2.dtype
                )

        # 3) 组装用于相似度计算的对端特征（当前批 + 历史负样本）
        if self.use_memory and self.curr_size2 > 0:
            all_feats2 = torch.cat([image_features2, self.memory_bank2[:self.curr_size2]], dim=0)
        else:
            all_feats2 = image_features2

        if self.use_memory and self.curr_size1 > 0:
            all_feats1 = torch.cat([image_features1, self.memory_bank1[:self.curr_size1]], dim=0)
        else:
            all_feats1 = image_features1

        # 4) 两个方向的相似度 logits（正样本仍在列 0..B-1）
        logits_12 = logit_scale * (image_features1 @ all_feats2.T)   # view1 -> view2(+mem2)
        logits_21 = logit_scale * (image_features2 @ all_feats1.T)   # view2 -> view1(+mem1)

        B = image_features1.size(0)
        labels = torch.arange(B, dtype=torch.long, device=self.device)

        # 5) 对称 InfoNCE（与原实现一致，取两向平均）
        loss12 = self.loss_function(logits_12, labels)
        loss21 = self.loss_function(logits_21, labels)
        nce_loss = (loss12 + loss21) / 2.0

        # 6) ICEL：跨模态最近邻一致性（使用历史库作为“稳定邻域”）
        icel_loss = torch.zeros((), device=image_features1.device, dtype=image_features1.dtype)
        if self.use_memory and self.use_icel and self.curr_size1 > 0 and self.curr_size2 > 0:
            # view1 -> (mem of view2)
            sim_1 = image_features1 @ self.memory_bank2[:self.curr_size2].T  # [B, K2]
            max_sim_1, nn_idx_1 = sim_1.max(dim=1)
            mask1 = max_sim_1 > self.icel_threshold
            if mask1.any():
                nn_feats2 = self.memory_bank2[nn_idx_1[mask1]]
                icel_loss_1 = F.mse_loss(image_features1[mask1], nn_feats2)
            else:
                icel_loss_1 = torch.zeros((), device=image_features1.device, dtype=image_features1.dtype)

            # view2 -> (mem of view1)
            sim_2 = image_features2 @ self.memory_bank1[:self.curr_size1].T  # [B, K1]
            max_sim_2, nn_idx_2 = sim_2.max(dim=1)
            mask2 = max_sim_2 > self.icel_threshold
            if mask2.any():
                nn_feats1 = self.memory_bank1[nn_idx_2[mask2]]
                icel_loss_2 = F.mse_loss(image_features2[mask2], nn_feats1)
            else:
                icel_loss_2 = torch.zeros((), device=image_features2.device, dtype=image_features2.dtype)

            icel_loss = (icel_loss_1 + icel_loss_2) / 2.0

        total_loss = nce_loss + self.lambda_icel * icel_loss

        # 7) 本批计算完后再把当前特征写入 memory（避免把“本批正样本列”混入扩展负样本）
        if self.use_memory:
            with torch.no_grad():
                self.mem_ptr1, self.curr_size1 = self._enqueue(self.memory_bank1, image_features1,
                                                               self.mem_ptr1, self.curr_size1)
                self.mem_ptr2, self.curr_size2 = self._enqueue(self.memory_bank2, image_features2,
                                                               self.mem_ptr2, self.curr_size2)

        return total_loss
