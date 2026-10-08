"""★ 阴算子（必需）—— 慢、收敛、弛豫。PRINCIPLES §3（L1 阴）。

沿共享能量头 E(h) 的梯度做收缩步：
    h ← h − lr · ∂E/∂h
把阳的发散候选收敛到能量地形的稳定邻域。不接受外部观测——阴主内。

★ 必需：step / relax
◇ 可扩展：Hopfield 闭式吸引子；DEQ 隐式微分（替代显式多步）；
  反向学习（噪声注入 + 反学习步）由 dream.py 调 substrate.inject 实现
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .substrate import LatentState, Substrate, advance


class YinMode(nn.Module):
    """阴：能量弛豫。持有共享底座引用（不复制——互根）。"""

    def __init__(self, substrate: Substrate, lr: float = 0.1) -> None:
        super().__init__()
        self.substrate = substrate
        self.lr = float(lr)

    def step(self, state: LatentState) -> LatentState:
        """阴一步：能量弛豫。

        create_graph=torch.is_grad_enabled()：训练态下梯度经弛豫步
        流回共享 energy_head（互根），推理态零开销。
        """
        h = state.h
        h_leaf = h.detach().requires_grad_(True)
        with torch.enable_grad():
            e = self.substrate.energy_of(h_leaf).sum()
            (g,) = torch.autograd.grad(
                e, h_leaf, create_graph=torch.is_grad_enabled()
            )
        # 在原 h（保留上游计算图）上施加更新——阴不切断历史
        return advance(state, h - self.lr * g)

    def relax(
        self,
        state: LatentState,
        n_steps: int = 3,
        tol: Optional[float] = None,
    ) -> LatentState:
        """多步弛豫至能量稳定邻域。tol: 单步平均能量变化阈值（早停 ◇）。"""
        if tol is None:
            for _ in range(n_steps):
                state = self.step(state)
            return state
        prev = None
        for _ in range(n_steps):
            state = self.step(state)
            with torch.no_grad():
                e = float(self.substrate.energy_of(state.h).mean())
            if prev is not None and abs(prev - e) < tol:
                break
            prev = e
        return state
