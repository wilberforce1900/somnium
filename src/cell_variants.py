"""★ cell_variants.py 门控/多层 cell 变体（迭代 #7 方向 (a)，TRM 教训警戒下小步试）。

假设链（排除链收官后）：d_h>256 病态 = 单层线性 cell 表达力天花板。
本模块提供两个**可选** cell 变体，全部参数化在 Substrate 之内（互根铁律不变）：

    GatedCell   GRU 式更新门+遗忘门（z, r）作用在阳核上：
                h' = (1-z)⊙h + z⊙tanh(W[h,r⊙h,x])——门控让"保留/写入"可分，
                宽潜态下未被写入的维可以恒定保持（对抗 tanh 饱和漂移）。
    TwoLayerCell 双层非线性 cell：h' = tanh(W2·gelu(W1[h,x]))——最朴素的表达力扩展。

接口：与 Substrate 完全同构（forward_delta(h,x) / energy_of(h)），可互换；
构造走 Substrate(cell_kind=...)，默认 "linear"（= v0，全部既有实验不受影响）。

设计红线：变体只动 cell，不动阴阳门控/能量头/潜态协议——单变量归因。
"""
from __future__ import annotations

import torch
from torch import nn


class GatedCell(nn.Module):
    """GRU 式门控阳核：输入布局 [h(m) | x(d_x)]，输出 m 维。"""

    def __init__(self, d_h: int, d_x: int) -> None:
        super().__init__()
        self.d_h, self.d_x = d_h, d_x
        in_dim = d_h + d_x
        self.Wz = nn.Linear(in_dim, d_h)   # 更新门 z
        self.Wr = nn.Linear(in_dim, d_h)   # 遗忘门 r
        self.Wh = nn.Linear(in_dim, d_h)   # 候选状态

    def forward(self, hx: torch.Tensor) -> torch.Tensor:
        h, x = hx[:, :self.d_h], hx[:, self.d_h:]
        z = torch.sigmoid(self.Wz(hx))
        r = torch.sigmoid(self.Wr(hx))
        h_cand = torch.tanh(self.Wh(torch.cat([r * h, x], dim=-1)))
        return (1 - z) * h + z * h_cand


class TwoLayerCell(nn.Module):
    """双层非线性阳核：gelu 放中间（比 tanh 堆叠梯度友好）。"""

    def __init__(self, d_h: int, d_x: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        self.d_h, self.d_x = d_h, d_x
        h2 = hidden or max(d_h, 64)
        self.L1 = nn.Linear(d_h + d_x, h2)
        self.L2 = nn.Linear(h2, d_h)

    def forward(self, hx: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.L2(torch.nn.functional.gelu(self.L1(hx))))


from typing import Optional  # noqa: E402


def build_cell(kind: str, d_h: int, d_x: int) -> nn.Module:
    """cell 工厂。kind ∈ {linear, gated, twolayer}。"""
    if kind == "linear":
        return nn.Linear(d_h + d_x, d_h)
    if kind == "gated":
        return GatedCell(d_h, d_x)
    if kind == "twolayer":
        return TwoLayerCell(d_h, d_x)
    raise ValueError(f"未知 cell kind: {kind}")
