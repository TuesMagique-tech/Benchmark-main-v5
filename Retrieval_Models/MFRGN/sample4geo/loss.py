# -*- coding: utf-8 -*-
"""
InfoNCE with (optional) DHML-style memory bank and (optional) ICEL.
本实现满足以下要求：
1) 在 __init__ 中建立两路特征的记忆库（memory_bank1 / memory_bank2），用于提供长/短期难负样本；
2) 在 forward 中，每个 batch 的特征会追加进记忆库，并在计算相似度时把“当前对端特征 + 记忆库特征”拼接为负样本池；
3) 计算 CE 时，只把前 N 列（当前 batch 的真实正样本列）作为正类，记忆库附加的列仅作为负类参与 softmax；
4) 提供 reset_memory()/clear_memory() 供外部在每个 epoch 开始时清空记忆库（可选，建议启用）；
5) ICEL 相关逻辑可通过 use_icel=False 完全关闭；若开启也只在该分支中生效，不影响 InfoNCE 主项。
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(...):
        ...
        # 记忆库初始化
        self.use_memory = bool(use_memory)
        self.memory_size = int(memory_size)
        # 新增：存储记忆库条目的标签ID（用于负样本过滤）
        self.register_buffer("memory_labels", torch.zeros(0, dtype=torch.long), persistent=False)
        ...
    @torch.no_grad()
    def reset_memory(self):
        # 清空记忆库以及对应标签
        if self.memory_bank1.numel():
            self.memory_bank1 = self.memory_bank1[:0]
            self.memory_labels = self.memory_labels[:0]
        if self.memory_bank2.numel():
            self.memory_bank2 = self.memory_bank2[:0]
    ...
    def _append_to_memory(self, feats1: torch.Tensor, feats2: torch.Tensor, labels: torch.LongTensor):
        """将当前batch的两路特征及标签追加到记忆库（滑动窗口更新）"""
        with torch.no_grad():
            if self.memory_bank1.numel() == 0:
                # 首次：用当前batch初始化
                self.memory_bank1 = feats1.detach().clone()
                self.memory_bank2 = feats2.detach().clone()
                self.memory_labels = labels.detach().clone()
            else:
                new_mem1 = torch.cat([self.memory_bank1, feats1.detach()], dim=0)
                new_mem2 = torch.cat([self.memory_bank2, feats2.detach()], dim=0)
                new_labels = torch.cat([self.memory_labels, labels.detach()], dim=0)
                # 若超过容量，裁掉最旧样本（保持最新 memory_size 条）
                if new_mem1.size(0) > self.memory_size:
                    new_mem1 = new_mem1[-self.memory_size:]
                    new_mem2 = new_mem2[-self.memory_size:]
                    new_labels = new_labels[-self.memory_size:]
                self.memory_bank1, self.memory_bank2 = new_mem1, new_mem2
                self.memory_labels = new_labels
    def forward(self, image_features1: torch.Tensor, image_features2: torch.Tensor,
                logit_scale: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        # 输入: f1 (无人机特征), f2 (卫星特征), logit_scale (温度参数), labels (类别ID)
        N = image_features1.size(0)
        # （已在模型中归一化，这里冗余归一化以防万一）
        f1 = self._maybe_normalize(image_features1)
        f2 = self._maybe_normalize(image_features2)
        # **组装负样本池**：包含当前对端样本 + 历史记忆样本（排除同类）
        if self.use_memory and self.memory_bank2.numel() > 0:
            # 筛除记忆库中与当前batch标签重复的样本
            mem_mask = ~torch.isin(self.memory_labels, labels).to(f1.device)
            feats2_mem = self.memory_bank2[mem_mask]       # 仅保留异类样本
            feats2_all = torch.cat([f2, feats2_mem], dim=0)  # 当前卫星 + 记忆卫星
        else:
            feats2_all = f2
        if self.use_memory and self.memory_bank1.numel() > 0:
            mem_mask = ~torch.isin(self.memory_labels, labels).to(f1.device)
            feats1_mem = self.memory_bank1[mem_mask]
            feats1_all = torch.cat([f1, feats1_mem], dim=0)
        else:
            feats1_all = f1
        # 计算相似度 logits （温度缩放）
        scale = logit_scale if torch.is_tensor(logit_scale) else torch.tensor(logit_scale, device=f1.device)
        logits_12 = scale * (f1 @ feats2_all.t())  # [N, N + M2']
        logits_21 = scale * (f2 @ feats1_all.t())  # [N, N + M1']
        # 构造CrossEntropy目标：仅监督前N列（当前batch一一对应正样本）
        targets = torch.arange(N, device=f1.device, dtype=torch.long)
        loss_12 = self.loss_function(logits_12, targets)
        loss_21 = self.loss_function(logits_21, targets)
        info_nce_loss = 0.5 * (loss_12 + loss_21)
        # **(新增) ICEL邻域一致性损失**：
        icel_loss = torch.tensor(0.0, device=f1.device)
        if self.use_icel and self.lambda_icel > 0.0:
            # 计算跨域余弦相似度矩阵（不含温度缩放）
            sim_matrix = f1 @ feats2_all.t()  # [N, N + M2']
            sim_matrix = sim_matrix.detach()  # 若需防止与InfoNCE梯度干扰，可选detach特征用于邻域判断
            # 寻找互为最近邻的样本对且相似度超过阈值
            # 对每个无人机样本i，在卫星集合中找最近邻j
            vals_i, idx_j = sim_matrix.max(dim=1)        # idx_j: [N] 每个i的最佳邻居索引
            # 对每个卫星样本j，在无人机集合中找最近邻i'
            vals_j, idx_i = sim_matrix.max(dim=0)        # idx_i: [N + M2'] 每个j的最佳邻居（可能含memory）
            # 收集互为近邻且相似度≥阈值的索引对
            neighbor_pairs = []
            for i in range(N):
                j = int(idx_j[i])
                if j < sim_matrix.size(1):  # j有效
                    # 检查对称
                    if idx_i[j] == i and vals_i[i] >= self.icel_threshold and vals_j[j] >= self.icel_threshold:
                        neighbor_pairs.append((i, j))
            # 计算一致性损失（均值）
            if neighbor_pairs:
                sim_vals = [ (f1[i] * feats2_all[j]).sum() for (i, j) in neighbor_pairs ]
                # icel_loss = 平均(1 - cos_sim)  （所有邻居对）
                icel_loss = torch.stack([1 - s for s in sim_vals]).mean()
        # InfoNCE主损失 + ICEL加权损失
        total_loss = info_nce_loss + self.lambda_icel * icel_loss
        # **延后更新记忆库**：在计算完loss后再追加当前特征
        if self.use_memory:
            self._append_to_memory(f1, f2, labels)
        return total_loss

