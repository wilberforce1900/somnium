"""★ 阳算子（必需）—— 快、发散、流式。PRINCIPLES §3（L1 阳）。

对共享主干正向动力学的多步快速展开：增量累积更新
    h ← h + dt · Δ(h, x)
不设收敛检查——发散倾向由阴算子的事后弛豫制衡（阴阳互济）。

★ 必需：step / stream
◇ 可扩展：展开深度课程化；并行流式；cell 线性注意力化（substrate 侧）
"""
from __future__ import annotations

import torch
from torch import nn

from .substrate import LatentState, Substrate, advance


class YangMode(nn.Module):
    """阳：感知与即时反应。持有共享底座引用（不复制——互根）。"""

    def __init__(self, substrate: Substrate, dt: float = 1.0) -> None:
        super().__init__()
        self.substrate = substrate
        self.dt = float(dt)

    def step(self, state: LatentState, x: torch.Tensor) -> LatentState:
        """阳一步：接收观测 x，流式增量更新。x: (B, d_x)。"""
        delta = self.substrate.forward_delta(state.h, x)
        return advance(state, state.h + self.dt * delta)

    def stream(self, state: LatentState, xs: torch.Tensor) -> LatentState:
        """多步发散展开。xs: (T, B, d_x)。"""
        for x in xs:
            state = self.step(state, x)
        return state
