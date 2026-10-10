"""★ ENV-D `rich_world`（迭代 #7）—— 参数化富世界，复杂度旋钮 k=0..4。

拍板方案（2026-10-10，用户确认"文档默认 + k≥3 部分可观测"）：
    k=0  现 ENV-A 等价（d_obs=8，对照组；内部复用 LatentGrid）
    k=1  视野 3×3 + 单房间 + 钥匙-门
    k=2  视野 5×5 + 双房间 + 钥匙-门 + 多目标(2)
    k=3  视野 5×5 + 四房间 + 推箱 + 多目标(3) + 部分可观测（视野外遮蔽）
    k=4  k=3 + 门周期开闭（非平稳：让"排练/稀有事件"组件再次有活干）

观测协议（ ITER7 §1.4 约定）：
    obs[:2] = 位置归一化 (x/G, y/G)   ← ObsCoverage dims=(0,1) 继续有效
    中段    = 视野展平（每格编码值，k=0 时为 ENV-A 的目标/危险增量等）
    尾段    = [时钟 t/H, 1.0]
    d_obs 由 env.d_obs 给出；action 恒 0..3（上右下左）。

奖励：goal(全部目标完成) +1 / 撞 hazard −1 / 每步 −0.01；info["rare"] 锚定
稀有事件（k≥1: 钥匙拾取；k=4: 门开闭瞬变）。

确定性与测试：同 seed 完全可复现；连通性由生成算法保证（房间树+门）。
◇ 可扩展：更多对象类型、动态 hazard、课程化 k。
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from .latent_grid import DIRS


class RichWorld:
    """参数化富世界。k=0 委托 LatentGrid（保证对照严格可比）。"""

    def __init__(self, complexity: int = 1, horizon: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        if complexity not in (0, 1, 2, 3, 4):
            raise ValueError(f"complexity 必须 0..4，得到 {complexity}")
        self.k = complexity
        if self.k == 0:  # 对照组：ENV-A 等价（含 hazard/rare）
            from .latent_grid import LatentGrid
            self._env0 = LatentGrid(grid=6, slip=0.1, p_threat_move=0.02,
                                    horizon=horizon or 30, seed=seed)
            self.d_obs = self._env0.d_obs
            self.n_actions = 4
            self.G = self._env0.G
            self.H = self._env0.H
            self._rng = self._env0.rng
            # 元数据补齐（测试/外部检查用；k=0 无门/钥匙概念）
            self.rooms = 0
            self.view = 0
            self.n_keys = self.n_goals = 0
            self.push_boxes = self.partial_obs = self.doors_oscillate = False
            self.walls: set = set()
            self.doors: list = []
            self.keys_pos: list = []
            self.goals_pos: list = []
            self._boxes: list = []
            self.hazard = None
            self._keys_held: set = set()
            self._goals_done: set = set()
            self._doors_open: dict = {}
            self.exit_gate = False
            return
        # ---- k≥1 参数表 ----
        self.view = 3 if self.k == 1 else 5                     # 视野边长
        self.rooms = 1 if self.k == 1 else (2 if self.k == 2 else 4)
        self.n_keys = 1 if self.k <= 2 else 2
        self.n_goals = 1 if self.k == 1 else (2 if self.k == 2 else 3)
        self.push_boxes = self.k >= 3
        self.partial_obs = self.k >= 3
        self.doors_oscillate = self.k >= 4
        self.G = 10 if self.k >= 2 else 8
        self.H = horizon or (60 + 40 * self.k)
        self._rng = random.Random(seed)
        self.d_obs = 4 + self.view * self.view + (2 if self.k >= 2 else 0) + 2
        #        位置2  视野 view²          钥匙/目标进度(k≥2)  时钟1 + 常量1
        self.n_actions = 4
        self._build()

    # ---- 生成 ----
    def _build(self) -> None:
        rng = self._rng
        G = self.G
        # 房间：竖直二分/四分（k=1 单房间全开）
        self.walls = set()  # (x,y) 墙格
        self.doors: List[Tuple[int, int]] = []
        if self.rooms >= 2:
            gx = G // 2
            gap = rng.randrange(1, G - 1)
            for y in range(G):
                if y != gap:
                    self.walls.add((gx, y))
            self.doors.append((gx, gap))
        if self.rooms >= 4:
            gy = G // 2
            for door_x in (rng.randrange(0, gx), rng.randrange(gx + 1, G)):
                for x in range(G):
                    if x != door_x and (x, gy) not in self.walls:
                        self.walls.add((x, gy))
                self.doors.append((door_x, gy))
        # 对象布置（空地不与墙/门重叠）
        def empty_cells(n: int, avoid=()) -> List[Tuple[int, int]]:
            pool = [(x, y) for x in range(G) for y in range(G)
                    if (x, y) not in self.walls and (x, y) not in self.doors
                    and (x, y) not in avoid]
            rng.shuffle(pool)
            return pool[:n]

        self.start = (0, 0)
        self.exit = (G - 1, G - 1)
        avoid = {self.start, self.exit}
        self.keys_pos = empty_cells(self.n_keys, avoid)
        avoid |= set(self.keys_pos)
        self.goals_pos = empty_cells(self.n_goals, avoid)
        avoid |= set(self.goals_pos)
        self.boxes_pos = empty_cells(2, avoid) if self.push_boxes else []
        avoid |= set(self.boxes_pos)
        self.hazard = empty_cells(1, avoid)[0]
        # 钥匙-门对应：每扇门绑一把钥匙（顺序取）。
        # k=1 单房间无墙门：钥匙绑定"终点门"——最后一个目标格需持钥才能登记完成
        # （踩上无钥=不登记，奖励不发；这是 k=1 钥匙的存在意义）。
        self.door_keys = {i: self.keys_pos[i % self.n_keys]
                          for i in range(len(self.doors))}
        self.exit_gate = (self.rooms == 1)  # k=1：终点需持钥
        # 状态
        self._agent = self.start
        self._keys_held = set()
        self._goals_done = set()
        self._boxes = list(self.boxes_pos)
        self._doors_open = {i: False for i in range(len(self.doors))}
        self._t = 0
        self._gen_tick = 0

    # ---- 视野渲染 ----
    def _cell_code(self, x: int, y: int) -> float:
        if (x, y) in self.walls:
            return 1.0
        if (x, y) in self.doors:
            i = self.doors.index((x, y))
            return 0.5 if self._doors_open[i] else 0.8
        if (x, y) in self.keys_pos and (x, y) not in self._keys_held:
            return 0.3
        if (x, y) in self.goals_pos and (x, y) not in self._goals_done:
            return 0.4
        if (x, y) == self.hazard:
            return 0.6
        if (x, y) in self._boxes:
            return 0.2
        return 0.0

    def _obs(self) -> torch.Tensor:
        rng = self._rng
        ax, ay = self._agent
        G, H = self.G, self.H
        out = [ax / G, ay / G]
        if self.k >= 2:
            out.append(len(self._keys_held) / max(self.n_keys, 1))
            out.append(len(self._goals_done) / max(self.n_goals, 1))
        half = self.view // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x, y = ax + dx, ay + dy
                if self.partial_obs and not (abs(dx) <= half and abs(dy) <= half):
                    out.append(0.0)  # 部分可观测：视野外恒 0（遮蔽）
                    continue
                if 0 <= x < G and 0 <= y < G:
                    out.append(self._cell_code(x, y))
                else:
                    out.append(1.0)  # 边界当墙
        out.append(self._t / H)
        out.append(1.0)
        return torch.tensor(out, dtype=torch.float32)

    # ---- 交互 ----
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if self.k == 0:
            return self._env0.reset(seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._build()
        return self._obs()

    def step(self, action: int):
        if self.k == 0:
            return self._env0.step(action)
        rng = self._rng
        rare = False
        ax, ay = self._agent
        dx, dy = DIRS[action]
        nx, ny = ax + dx, ay + dy
        # k=4 非平稳：门周期开闭（每 20 步翻转一次状态）
        if self.doors_oscillate:
            self._gen_tick += 1
            if self._gen_tick % 20 == 0:
                for i in self._doors_open:
                    self._doors_open[i] = not self._doors_open[i]
                rare = True
        blocked = False
        if not (0 <= nx < self.G and 0 <= ny < self.G):
            blocked = True
        elif (nx, ny) in self.walls:
            blocked = True
        elif (nx, ny) in self.doors:
            i = self.doors.index((nx, ny))
            key = self.door_keys[i]
            if self._doors_open[i]:
                pass  # 开门可穿
            elif key in self._keys_held:
                self._doors_open[i] = True  # 有钥匙自动开门（穿过）
                rare = True
            else:
                blocked = True  # 锁着
        else:
            # 推箱（k≥3）：目标格须空
            if (nx, ny) in self._boxes:
                bx, by = nx + dx, ny + dy
                if (0 <= bx < self.G and 0 <= by < self.G
                        and (bx, by) not in self.walls
                        and (bx, by) not in self.doors
                        and (bx, by) not in self._boxes
                        and (bx, by) != self.hazard):
                    self._boxes[self._boxes.index((nx, ny))] = (bx, by)
                else:
                    blocked = True
        if not blocked:
            self._agent = (nx, ny)
        # 拾钥匙
        if self._agent in self.keys_pos and self._agent not in self._keys_held:
            self._keys_held.add(self._agent)
            rare = True
        self._t += 1
        done, reward = False, -0.01
        if self._agent in self.goals_pos and self._agent not in self._goals_done:
            # 终点门（k=1）：最后一个目标需持钥登记
            if not (self.exit_gate and len(self._goals_done) == self.n_goals - 1
                    and not self._keys_held):
                self._goals_done.add(self._agent)
                rare = True
        if len(self._goals_done) >= self.n_goals:
            reward, done = 1.0, True
        elif self._agent == self.hazard:
            reward, done = -1.0, True
        elif self._t >= self.H:
            done = True
        return self._obs(), reward, done, {"rare": rare,
                                           "progress": len(self._goals_done)}

    # ---- 兼容 random_rollout/guided 采集接口 ----
    @property
    def rng(self):
        return self._rng
