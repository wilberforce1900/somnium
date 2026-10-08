"""★ 太极门控（必需）—— 阴阳消长、互根混合。PRINCIPLES §3。

α ∈ (0,1) 为阳配比，每步软混合两种更新（阳中有阴、阴中有阳）：
    h ← h + α·Δ阳 + (1−α)·Δ阴

gate_mode —— A2 消融开箱即用（PRINCIPLES §11）：
    "learned"    可学习门（默认）
    "fixed_yang" α ≡ 1（纯阳对照）
    "fixed_yin"  α ≡ 0（纯阴对照）
    "schedule"   外部节律 f(t)（◇ 对接梦醒宏节律：醒期偏阳、梦期偏阴）

◇ 可扩展：门控上下文加能量/卦码；宏梦醒节律调度。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from .substrate import LatentState, Substrate, advance
from .yang import YangMode
from .yin import YinMode

_GATE_MODES = ("learned", "fixed_yang", "fixed_yin", "schedule")


class TaijiGate(nn.Module):
    """太极：模式门控与软混合。持有 substrate / yang / yin 的引用。"""

    def __init__(
        self,
        substrate: Substrate,
        yang: YangMode,
        yin: YinMode,
        gate_mode: str = "learned",
        schedule: Optional[Callable[[int], float]] = None,
    ) -> None:
        super().__init__()
        if gate_mode not in _GATE_MODES:
            raise ValueError(f"gate_mode 必须是 {_GATE_MODES} 之一：{gate_mode}")
        if gate_mode == "schedule" and schedule is None:
            raise ValueError("schedule 模式必须提供 schedule=f(t) -> α")
        self.substrate = substrate
        self.yang = yang
        self.yin = yin
        self.gate_mode = gate_mode
        self.schedule = schedule
        self.gate_net = nn.Linear(substrate.d_h + substrate.d_x, 1)
        with torch.no_grad():
            # 零初始化：出生时 σ(0)=0.5，阴阳各半
            self.gate_net.weight.zero_()
            self.gate_net.bias.zero_()

    def alpha(self, state: LatentState, x: torch.Tensor) -> torch.Tensor:
        """阳配比 α，shape (B, 1)。"""
        b = state.h.shape[0]
        if self.gate_mode == "learned":
            return torch.sigmoid(self.gate_net(torch.cat([state.h, x], dim=-1)))
        if self.gate_mode == "fixed_yang":
            return state.h.new_ones((b, 1))
        if self.gate_mode == "fixed_yin":
            return state.h.new_zeros((b, 1))
        a = float(self.schedule(state.t))
        return state.h.new_full((b, 1), a)

    def mixed_step(
        self, state: LatentState, x: torch.Tensor
    ) -> tuple[LatentState, torch.Tensor]:
        """一步阴阳混合更新。返回 (新潜态, α)。"""
        a = self.alpha(state, x)
        s_yang = self.yang.step(state, x)
        s_yin = self.yin.step(state)
        dh_y = s_yang.h - state.h
        dh_i = s_yin.h - state.h
        h_new = state.h + a * dh_y + (1.0 - a) * dh_i
        return advance(state, h_new), a
