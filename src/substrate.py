"""★ 公共底座（必需）—— 阴阳共同的本体（"一体"）。

PRINCIPLES §2（L0 混沌）/ §3（L1 阴阳）的代码化。

设计铁律（PRINCIPLES §3，吸取 TRM 教训）：
    阴阳是同一底座上的两种动力学模式，不是两个网络。
    cell（阳用）与 energy（阴用）永远挂在同一个 Substrate 实例上，
    参数互根，不随模式分离。

组件：
    LatentState  潜态束：阴阳共同作用的 h + 全局步数 t
    Substrate    共享主干：cell 正向动力学（阳）+ energy 能量地形（阴）
                 + spawn/inject（L0 混沌入口）
    NoisePrior   噪声底质：初始化（太极未分）与梦期噪声的强度

◇ 可扩展（v0 留口不实现）：
    - cell → 线性注意力 / DeltaNet 单元（E1 后升级）
    - LatentState 加卦码槽位（L3 codebook.py 接入）
    - sigma 课程化 / 可学习（对接 dream.py 梦期调度）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def default_device() -> torch.device:
    """cuda → mps → cpu 自动降级；测试一律显式 CPU。◇ 后续挪 utils.py。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LatentState:
    """阴阳共享的潜态束。v0 仅 h（主干潜态）与 t（全局步数）。

    ◇ 可扩展：卦码槽（L3）、阴阳模式轨迹（梦 log 用）、情景缓冲引用。
    """

    h: torch.Tensor
    t: int = 0


def advance(state: LatentState, h_new: torch.Tensor) -> LatentState:
    """以新潜态推进一步——所有模式算子的统一出口。"""
    return LatentState(h_new, state.t + 1)


class NoisePrior(nn.Module):
    """L0 混沌底质：噪声强度 sigma。v0 为常数。

    ◇ 可扩展：退火课程、可学习 sigma（梦期反向学习用，PRINCIPLES §4）。
    """

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = float(sigma)


class RMSNorm(nn.Module):
    """仅尺度归一化（ITER7 §2.5 (b) 线用）。

    本机 torch 2.2.2 无 nn.RMSNorm（2.4 才内置），本地实现保证双端一致。
    不除均值——保留逐样本特征均值通道（LN 无此通道，见 ITER7 §2.5 臂设计）。
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms.to(x.dtype)) * self.weight


class Substrate(nn.Module):
    """共享主干（公共底座本体）。

    - cell:        Linear(d_h + d_x → d_h)，阳算子的正向动力学核
    - energy_head: MLP(d_h → 1)，阴算子下降的能量地形
    - 近线性小初始化：出生时贴近混沌（太极未分），训练中秩序渐生
    """

    def __init__(self, d_x: int = 32, d_h: int = 64, init_scale: float = 0.1,
                 cell_kind: str = "linear", norm_kind: str = "none") -> None:
        super().__init__()
        self.d_x = d_x
        self.d_h = d_h
        self.cell_kind = cell_kind
        # 迭代#7 (b) 线：潜态归一化（ITER7 §2.5）。none=恒等（v0 行为逐位不变）。
        if norm_kind not in ("none", "ln_post", "rms_post", "ln_pre"):
            raise ValueError(f"norm_kind 未知：{norm_kind}")
        self.norm_kind = norm_kind
        if norm_kind in ("ln_post", "ln_pre"):
            self.state_norm = nn.LayerNorm(d_h)
        elif norm_kind == "rms_post":
            self.state_norm = RMSNorm(d_h)
        # cell 工厂（迭代#7 方向 a）：linear=v0 原样；gated/twolayer 见 cell_variants。
        # 变体自带合理初始化，不走下方线性小初始化（init_scale 仅作用于 linear）。
        from .cell_variants import build_cell
        self.cell = build_cell(cell_kind, d_h, d_x)
        self.energy_head = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.GELU(),
            nn.Linear(d_h, 1),
        )
        self.noise = NoisePrior()
        with torch.no_grad():
            if cell_kind == "linear":
                for lin in (self.cell, self.energy_head[0], self.energy_head[2]):
                    lin.weight.uniform_(-init_scale, init_scale)
                    lin.bias.zero_()
            else:
                for lin in (self.energy_head[0], self.energy_head[2]):
                    lin.weight.uniform_(-init_scale, init_scale)
                    lin.bias.zero_()

    # ---- 阳侧动力学核（yang.py 调用） ----
    def forward_delta(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """共享正向动力学：Δ = tanh(cell([h, x]))。

        linear cell 与变体（gated/twolayer 自带末端 tanh）统一套外层 tanh：
        变体输出再过一次 tanh 会压缩梯度——故变体分支直接返回 cell 输出。
        """
        z = self.cell(torch.cat([self.pre_cell_input(h), x], dim=-1))
        if self.cell_kind == "linear":
            return torch.tanh(z)
        return z

    # ---- 迭代#7 (b) 线：潜态归一化（ITER7 §2.5） ----
    def pre_cell_input(self, h: torch.Tensor) -> torch.Tensor:
        """cell 输入侧的 h 分量（ln_pre 臂在此归一化；其余原样）。"""
        if self.norm_kind == "ln_pre":
            return self.state_norm(h)
        return h

    def norm_state(self, h: torch.Tensor) -> torch.Tensor:
        """混合出口的潜态归一化（ln_post / rms_post 臂；其余恒等）。

        仅施加于混合出口；extra_relax 通道未纳入（(b) 线实验 extra_relax=0）。
        """
        if self.norm_kind in ("ln_post", "rms_post"):
            return self.state_norm(h)
        return h

    def cell_preact(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """cell 前置激活 z（饱和探针用；linear cell 下 |z|>2 记饱和）。"""
        return self.cell(torch.cat([self.pre_cell_input(h), x], dim=-1))

    # ---- 阴侧能量地形（yin.py 调用） ----
    def energy_of(self, h: torch.Tensor) -> torch.Tensor:
        """共享能量地形 E(h)，shape (B,)。"""
        return self.energy_head(h).squeeze(-1)

    # ---- L0 混沌入口（dream.py 将复用） ----
    def spawn(self, batch_size: int) -> LatentState:
        """从噪声先验采样初始潜态（混沌中出生）。"""
        p = next(self.cell.parameters())  # 变体 cell 无单 .weight，取任意参数定设备
        h = torch.randn(batch_size, self.d_h, device=p.device, dtype=p.dtype)
        return LatentState(h * self.noise.sigma, t=0)

    def inject(self, h: torch.Tensor, level: float) -> torch.Tensor:
        """噪声注入：梦期 rollout 起点 / 反向学习的统一入口（dream.py 用）。"""
        return h + level * torch.randn_like(h)
