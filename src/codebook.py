"""★ codebook.py 六十四卦态势码本（必需）—— PRINCIPLES §5（L3）/ ROADMAP E2（A1）。

- 卦 = 码字：K 个 d_h 维码字（K=64 ↔ 6-bit，恰好一轮卦数）
- 爻变 = 转移：训练时统计码字间的转移计数（EMA/累计）→ 行随机转移图
- 取象 = 量化：h → 最近码字（VQ；训练用 commitment 损失把潜态拉向卦空间）
- 卦德 = 统计：每卦累计出现次数与经过奖励 → 供图上价值迭代（planner.py）

卦象编码：**先天卦序**（邵雍二进制序，低位=初爻）——不做 King Wen 名号标注，
避免显示层错误标签（正确名表留 ◇ 显示层再做）。

◇ 可扩展：可学习转移网络（代替计数）、EMA 衰减、用九/用六特殊码语义、
  King Wen 显示层、卦码检索记忆库（卦爻辞=按码字索引的技能先验）
"""
from __future__ import annotations

import torch
from torch import nn


class HexagramCodebook(nn.Module):
    """VQ 态势码本 + 转移图统计。挂在 WorldModel 上（可选）即激活 L3。"""

    def __init__(self, d_h: int, k: int = 64, beta: float = 0.25) -> None:
        super().__init__()
        self.k = k
        self.d_h = d_h
        self.beta = beta
        self.codes = nn.Parameter(torch.randn(k, d_h) * 0.1)
        self.register_buffer("counts", torch.zeros(k, k))      # 转移计数
        self.register_buffer("visits", torch.zeros(k))          # 卦出现次数
        self.register_buffer("reward_sum", torch.zeros(k))      # 卦内累计奖励
        self.register_buffer("usage", torch.zeros(k))           # 使用率 EMA（死码检测）

    # ---- 取象：量化 ----
    def assign(self, h: torch.Tensor) -> torch.Tensor:
        """h (B,d_h) → 最近码字索引 (B,)。"""
        d = torch.cdist(h, self.codes)
        return d.argmin(dim=1)

    def vq_loss(self, h: torch.Tensor):
        """VQ 损失（commitment + codebook），返回 (loss, idx)。"""
        idx = self.assign(h)
        code = self.codes[idx]
        commit = torch.mean((h - code.detach()) ** 2)
        codebook = torch.mean((h.detach() - code) ** 2)
        return commit + self.beta * codebook, idx

    # ---- 观象：统计更新（仅训练态调用） ----
    @torch.no_grad()
    def observe(self, idx_seq: torch.Tensor, rew_seq: torch.Tensor = None) -> None:
        """idx_seq (B,T)；rew_seq (B,T) 与 idx 对齐（到达该卦时的奖励）。"""
        B, T = idx_seq.shape
        if T >= 2:
            pairs = (idx_seq[:, :-1] * self.k + idx_seq[:, 1:]).flatten()
            self.counts.view(-1).scatter_add_(
                0, pairs, torch.ones_like(pairs, dtype=self.counts.dtype))
        self.visits.scatter_add_(0, idx_seq.flatten(),
                                 torch.ones(B * T, dtype=self.visits.dtype))
        if rew_seq is not None:
            self.reward_sum.scatter_add_(0, idx_seq.flatten(), rew_seq.flatten())
        bc = torch.bincount(idx_seq.flatten(), minlength=self.k).float()
        self.usage.mul_(0.99).add_(0.01 * bc / bc.sum().clamp(min=1))

    @torch.no_grad()
    def reseed_dead(self, h_pool: torch.Tensor, thresh: float = 0.002) -> int:
        """死码重生（VQ 标准 trick）：长期未被赋值的码字重置为当前潜态样本。

        返回重置数量。转移/奖励统计不清零（重置只改码向量）。
        """
        dead = self.usage < thresh
        n_dead = int(dead.sum().item())
        if n_dead and h_pool is not None and h_pool.shape[0] > 0:
            picks = h_pool[torch.randint(0, h_pool.shape[0], (n_dead,),
                                         device=h_pool.device)]
            self.codes.data[dead] = picks + 0.01 * torch.randn_like(picks)
        return n_dead

    # ---- 卦德：读出 ----
    def transition_probs(self) -> torch.Tensor:
        """行随机转移矩阵；从未观测的卦行全零（不传播）。"""
        T = self.counts.clone()
        row = T.sum(dim=1, keepdim=True)
        return torch.where(row > 0, T / row.clamp(min=1), torch.zeros_like(T))

    def mean_reward(self) -> torch.Tensor:
        """每卦平均奖励；未见卦为 0。"""
        visited = self.visits > 0
        mr = torch.where(visited, self.reward_sum / self.visits.clamp(min=1),
                         torch.zeros_like(self.reward_sum))
        return mr

    # ---- 卦象：先天序二进制 ----
    def to_hexagram(self, idx: int) -> str:
        """先天卦序二进制串，低位=初爻（阳=1）。K 非 2^n 时位数取 ceil(log2 K)。"""
        bits = max(1, (self.k - 1).bit_length())
        return format(int(idx), f"0{bits}b")


def value_iteration(cb: HexagramCodebook, gamma: float = 0.9, depth: int = 3) -> torch.Tensor:
    """图上慢规划的核心：V = 卦德 + γ·T·V，迭代 depth 次。返回 (K,)。"""
    mr = cb.mean_reward()
    T = cb.transition_probs()
    V = mr.clone()
    for _ in range(depth):
        V = mr + gamma * (T @ V)
    return V
