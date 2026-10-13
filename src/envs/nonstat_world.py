"""★ ENV-E v2 `nonstat_world`（iter8 阶段 1）—— 策略层非平稳世界。

v1（slip 双 regime）pilot 教训（ENV-E/P0，2026-10-12/13）：slip-only 在预测层
对健康模型太温和——模型预切换即泛化（m1@switch≈收敛值）且零干扰，48-run
矩阵被否决。v2 按 ENV-E-PROPOSAL 评审刀③⑤转向**策略层分歧**：

    两个特殊格 A/B 的角色随 regime 翻转：
        ρ0：A=goal(+1)，B=hazard(−1)   → 最优策略=奔 A 避 B
        ρ1：A=hazard(−1)，B=goal(+1)   → 最优策略=奔 B 避 A（反向导航）
    奖励语义翻转 → 预测层（奖励头）与策略层同时被迫重学，不可预泛化。

    regime 是**跨 episode 的全局持续过程**（市场的诚实抽象；v1 评审刀④）：
    每步以概率 lam 切换；**不进观测**（部分可观测）；info["switched"] 只进日志。
    lam=0 退化为纯静态单 regime（矩阵锚点）。

观测 schema 与 LatentGrid 同构（8 维：位置/A 增量/B 增量/时钟/1）——仪器连续性；
slip 两 regime 相同（与 v1 的 slip 轴解耦）。
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from .latent_grid import DIRS


class NonStatWorld:
    """策略层非平稳网格世界（ENV-E v2 载具）。"""

    def __init__(self, grid: int = 8, slip: float = 0.1, lam: float = 0.0,
                 horizon: int = 50, seed: Optional[int] = None,
                 regime: int = 0) -> None:
        if regime not in (0, 1):
            raise ValueError("regime 必须 0/1")
        self.G = grid
        self.slip = slip
        self.lam = lam
        self.H = horizon
        self.regime = regime
        self._rng = random.Random(seed)
        self.d_obs = 8
        self.n_actions = 4
        self._agent = (0, 0)
        self._a = (0, 0)   # 特殊格 A
        self._b = (0, 0)   # 特殊格 B
        self._t = 0
        self.switches = 0  # 累计切换数（统计校验用）

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self._rng = random.Random(seed)
        cells = []
        while len(cells) < 3:
            c = (self._rng.randrange(self.G), self._rng.randrange(self.G))
            if c not in cells:
                cells.append(c)
        self._agent, self._a, self._b = cells
        self._t = 0
        return self._obs()

    def _obs(self) -> torch.Tensor:
        ax, ay = self._agent
        (ax_, ay_), (bx_, by_) = self._a, self._b
        G, H = self.G, self.H
        return torch.tensor(
            [ax / G, ay / G,
             (ax_ - ax) / G, (ay_ - ay) / G,      # A 增量（语义随 regime 翻转）
             (bx_ - ax) / G, (by_ - ay) / G,      # B 增量
             self._t / H, 1.0],
            dtype=torch.float32,
        )

    def step(self, action: int):
        """返回 (obs, reward, done, info)；info 含 rare 与 switched（仅日志）。"""
        rng = self._rng
        switched = False
        if rng.random() < self.lam:
            self.regime = 1 - self.regime
            self.switches += 1
            switched = True
        if rng.random() < self.slip:
            action = rng.randrange(4)
        dx, dy = DIRS[action]
        ax, ay = self._agent
        self._agent = (min(max(ax + dx, 0), self.G - 1),
                       min(max(ay + dy, 0), self.G - 1))
        self._t += 1
        goal, hazard = ((self._a, self._b) if self.regime == 0
                        else (self._b, self._a))
        done, reward = False, -0.01
        if self._agent == goal:
            reward, done = 1.0, True
        elif self._agent == hazard:
            reward, done = -1.0, True
        elif self._t >= self.H:
            done = True
        return self._obs(), reward, done, {"rare": switched, "switched": switched}

    # ---- 兼容 random_rollout/guided 采集接口 ----
    @property
    def rng(self):
        return self._rng
