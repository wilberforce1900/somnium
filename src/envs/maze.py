"""★ ENV-C `maze`（必需）—— E1 门控消融任务。ROADMAP §1.0 / §1.2。

完美迷宫（DFS 回溯 carving 生成，每局新迷宫）：必须感知局部墙壁结构
才能规划长路径——与 ENV-A 的"平坦动力学+稀有事件"形成任务对比。

- obs (d_obs=10): [ax/G, ay/G, gdx/G, gdy/G, 墙↑, 墙→, 墙↓, 墙←, t/H, 1]
- action 0..3（上右下左）；撞墙原地不动
- reward: goal +1（终局）/ 每步 -0.01 / 超时 0；无 slip、无 hazard → rare 恒 False
- 难度分析锚：info["dist"] = 当前位置到出口的 BFS 步数（不入 obs，
  仅供 E1 的 α-难度相关性分析）

◇ 可扩展：固定迷宫池（课程）、多出口、钥匙门、轨迹观测
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

import torch

from .latent_grid import DIRS  # 上右下左，与 action 编号一致


class Maze:
    """每局重新生成完美迷宫；入口 (0,0)，出口 (G-1,G-1)。"""

    def __init__(
        self,
        size: int = 8,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.G = size
        self.H = horizon if horizon is not None else 2 * size * size
        self.rng = random.Random(seed)
        self.d_obs = 10
        self.n_actions = 4
        self._open: List[List[int]] = []
        self._dist: List[List[int]] = []
        self._agent = (0, 0)
        self._goal = (size - 1, size - 1)
        self._t = 0
        self.maze_id = 0

    # ---- 生成 ----
    def _gen_maze(self) -> List[List[int]]:
        """DFS 回溯 carving：返回每格的开墙位掩码（bit d = DIRS[d] 方向开）。"""
        G = self.G
        open_mask = [[0] * G for _ in range(G)]
        visited = [[False] * G for _ in range(G)]
        stack = [(0, 0)]
        visited[0][0] = True
        while stack:
            x, y = stack[-1]
            nbrs = []
            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < G and 0 <= ny < G and not visited[ny][nx]:
                    nbrs.append((d, nx, ny))
            if not nbrs:
                stack.pop()
                continue
            d, nx, ny = nbrs[self.rng.randrange(len(nbrs))]
            open_mask[y][x] |= 1 << d
            open_mask[ny][nx] |= 1 << ((d + 2) % 4)
            visited[ny][nx] = True
            stack.append((nx, ny))
        return open_mask

    def _bfs_dist(self) -> List[List[int]]:
        """从出口反向 BFS：到出口的最短步数表（难度/走解测试用）。"""
        G = self.G
        dist = [[-1] * G for _ in range(G)]
        gx, gy = self._goal
        dist[gy][gx] = 0
        dq = deque([(gx, gy)])
        while dq:
            x, y = dq.popleft()
            for d in range(4):
                if self._open[y][x] & (1 << d):
                    dx, dy = DIRS[d]
                    nx, ny = x + dx, y + dy
                    if dist[ny][nx] < 0:
                        dist[ny][nx] = dist[y][x] + 1
                        dq.append((nx, ny))
        return dist

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self.rng = random.Random(seed)
        self._open = self._gen_maze()
        self._dist = self._bfs_dist()
        self._agent = (0, 0)
        self._goal = (self.G - 1, self.G - 1)
        self._t = 0
        self.maze_id += 1
        return self._obs()

    # ---- 交互 ----
    def _obs(self) -> torch.Tensor:
        ax, ay = self._agent
        gx, gy = self._goal
        G, H = self.G, self.H
        walls = [0.0 if self._open[ay][ax] & (1 << d) else 1.0 for d in range(4)]
        return torch.tensor(
            [ax / G, ay / G, (gx - ax) / G, (gy - ay) / G,
             *walls, self._t / H, 1.0],
            dtype=torch.float32,
        )

    def step(self, action: int):
        dx, dy = DIRS[action]
        ax, ay = self._agent
        if self._open[ay][ax] & (1 << action):
            self._agent = (ax + dx, ay + dy)
        self._t += 1
        done, reward = False, -0.01
        if self._agent == self._goal:
            reward, done = 1.0, True
        elif self._t >= self.H:
            done = True
        return self._obs(), reward, done, {
            "rare": False,
            "dist": self._dist[self._agent[1]][self._agent[0]],
        }
