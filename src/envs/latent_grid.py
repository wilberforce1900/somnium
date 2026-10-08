"""★ ENV-A `latent_grid`（必需）—— E0/E3 实验环境。ROADMAP §1.0。

2D 网格世界：可控转移动力学（slip）+ 稀有威胁事件（hazard 随机迁移）。

- obs (d_obs=8, float32): [ax/G, ay/G, gdx/G, gdy/G, hdx/G, hdy/G, t/H, 1]
- action: int 0..3 = 上/右/下/左
- reward: goal +1（终局）/ hazard -1（终局）/ 每步 -0.01 / 超时 0
- rare 事件：hazard 迁移步 info["rare"]=True —— 排练(rehearsal)与梦 log 的锚

◇ 可扩展：局部网格视野、多 hazard、连续动作、固定起点课程（已留 fixed_* 参数）
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # 上右下左


class LatentGrid:
    """确定性种子 + 可控动力学的网格世界。"""

    def __init__(
        self,
        grid: int = 8,
        slip: float = 0.1,
        p_threat_move: float = 0.02,
        horizon: int = 50,
        seed: Optional[int] = None,
        fixed_start: Optional[Tuple[int, int]] = None,
        fixed_goal: Optional[Tuple[int, int]] = None,
        fixed_hazard: Optional[Tuple[int, int]] = None,
        with_hazard: bool = True,
    ) -> None:
        if (fixed_start is None) != (fixed_goal is None):
            raise ValueError("fixed_start 与 fixed_goal 必须同时提供或同时缺省")
        self.G = grid
        self.slip = slip
        self.p_threat = p_threat_move
        self.H = horizon
        self.fixed_start = fixed_start
        self.fixed_goal = fixed_goal
        self.fixed_hazard = fixed_hazard
        self.with_hazard = with_hazard
        self.rng = random.Random(seed)
        self.d_obs = 8
        self.n_actions = 4
        self._agent = (0, 0)
        self._goal = (0, 0)
        self._hazard: Optional[Tuple[int, int]] = None
        self._t = 0

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self.rng = random.Random(seed)
        if self.fixed_start is not None:
            self._agent = tuple(self.fixed_start)
            self._goal = tuple(self.fixed_goal)
            if not self.with_hazard:
                self._hazard = None
            elif self.fixed_hazard is not None:
                self._hazard = tuple(self.fixed_hazard)
            else:
                empties = [
                    (x, y)
                    for x in range(self.G)
                    for y in range(self.G)
                    if (x, y) not in (self._agent, self._goal)
                ]
                self._hazard = empties[self.rng.randrange(len(empties))]
        else:
            n_cells = 3 if self.with_hazard else 2
            cells: List[Tuple[int, int]] = []
            while len(cells) < n_cells:
                c = (self.rng.randrange(self.G), self.rng.randrange(self.G))
                if c not in cells:
                    cells.append(c)
            self._agent, self._goal = cells[0], cells[1]
            self._hazard = cells[2] if self.with_hazard else None
        self._t = 0
        return self._obs()

    def _obs(self) -> torch.Tensor:
        ax, ay = self._agent
        gx, gy = self._goal
        hx, hy = self._hazard if self._hazard is not None else self._agent
        G, H = self.G, self.H
        return torch.tensor(
            [
                ax / G,
                ay / G,
                (gx - ax) / G,
                (gy - ay) / G,
                (hx - ax) / G,
                (hy - ay) / G,
                self._t / H,
                1.0,
            ],
            dtype=torch.float32,
        )

    def step(self, action: int):
        """执行动作。返回 (obs, reward, done, info)，info={"rare": bool}。"""
        if self.rng.random() < self.slip:
            action = self.rng.randrange(4)
        dx, dy = DIRS[action]
        ax, ay = self._agent
        nx = min(max(ax + dx, 0), self.G - 1)
        ny = min(max(ay + dy, 0), self.G - 1)
        self._agent = (nx, ny)
        self._t += 1
        # 稀有威胁事件：hazard 迁移到随机空格
        rare = False
        if self._hazard is not None and self.rng.random() < self.p_threat:
            empties = [
                (x, y)
                for x in range(self.G)
                for y in range(self.G)
                if (x, y) != self._agent and (x, y) != self._goal
            ]
            self._hazard = empties[self.rng.randrange(len(empties))]
            rare = True
        done, reward = False, -0.01
        if self._agent == self._goal:
            reward, done = 1.0, True
        elif self._hazard is not None and self._agent == self._hazard:
            reward, done = -1.0, True
        elif self._t >= self.H:
            done = True
        return self._obs(), reward, done, {"rare": rare}


def random_rollout(env: LatentGrid):
    """随机策略跑一整局。返回 (obs(T+1,d_obs), act(T,)long, rew(T,), rare(T,)bool)。"""
    obs0 = env.reset()
    obs_list, act_list, rew_list, rare_list = [obs0], [], [], []
    done = False
    while not done:
        a = env.rng.randrange(env.n_actions)
        o, r, done, info = env.step(a)
        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        rare_list.append(info["rare"])
    return (
        torch.stack(obs_list),
        torch.tensor(act_list, dtype=torch.long),
        torch.tensor(rew_list, dtype=torch.float32),
        torch.tensor(rare_list, dtype=torch.bool),
    )


def guided_rollout(env: LatentGrid, p_greedy: float = 0.8):
    """贪心引导策略 rollout（P0-1 课程）：以 p_greedy 沿目标增量方向走，其余随机。

    解决稀疏奖励：随机策略在对角长程任务上几乎摸不到 +1，奖励头/卦德图会失明。
    引导局仍是真实环境交互（A3 的"真实步数"口径不变）。依赖 obs 的目标增量
    分量（LatentGrid/Maze 的 obs[2:4] 均为此格式）。
    """
    obs0 = env.reset()
    obs_list, act_list, rew_list, rare_list = [obs0], [], [], []
    done = False
    cur = obs0
    while not done:
        if env.rng.random() < p_greedy:
            gdx, gdy = cur[2].item() * env.G, cur[3].item() * env.G
            if abs(gdx) >= abs(gdy) and abs(gdx) > 1e-6:
                a = 1 if gdx > 0 else 3
            elif abs(gdy) > 1e-6:
                a = 0 if gdy > 0 else 2
            else:
                a = env.rng.randrange(env.n_actions)
        else:
            a = env.rng.randrange(env.n_actions)
        cur, r, done, info = env.step(a)
        obs_list.append(cur)
        act_list.append(a)
        rew_list.append(r)
        rare_list.append(info["rare"])
    return (
        torch.stack(obs_list),
        torch.tensor(act_list, dtype=torch.long),
        torch.tensor(rew_list, dtype=torch.float32),
        torch.tensor(rare_list, dtype=torch.bool),
    )
