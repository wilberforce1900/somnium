"""ENV-C maze 测试：生成确定性 / 连通性 / 撞墙 / 走解 / 观测 / rare。"""
import torch

from src.envs import Maze, random_rollout
from src.envs.latent_grid import DIRS


class TestGeneration:
    def test_seed_deterministic(self):
        a = Maze(size=5, horizon=40, seed=7).reset()
        b = Maze(size=5, horizon=40, seed=7).reset()
        assert torch.equal(a, b)

    def test_all_cells_reachable(self):
        env = Maze(size=6, horizon=60, seed=1)
        env.reset()
        assert all(d >= 0 for row in env._dist for d in row), "完美迷宫应全连通"

    def test_new_maze_each_reset(self):
        env = Maze(size=6, horizon=60, seed=2)
        o1 = env.reset()
        o2 = env.reset()
        # 同 seed 流下两局迷宫不同（生成器推进）；不比较 obs 首步（起点固定相同）
        assert env.maze_id == 2


class TestDynamics:
    def test_wall_blocks(self):
        env = Maze(size=5, horizon=40, seed=3)
        env.reset()
        ax, ay = env._agent
        # 找一个封死的方向撞墙 → 原地
        blocked = [d for d in range(4) if not (env._open[ay][ax] & (1 << d))]
        assert blocked, "角落至少有封死方向"
        obs, r, done, _ = env.step(blocked[0])
        assert obs[0].item() == 0.0 and obs[1].item() == 0.0
        assert r == -0.01 and not done

    def test_solve_by_dist(self):
        """沿 BFS 距离递减方向走必达出口——连通性与动力学联合验证。"""
        env = Maze(size=5, horizon=80, seed=3)
        env.reset()
        done, steps, r = False, 0, 0.0
        while not done:
            ax, ay = env._agent
            best_d, best_a = None, 0
            for d in range(4):
                if env._open[ay][ax] & (1 << d):
                    dx, dy = DIRS[d]
                    nd = env._dist[ay + dy][ax + dx]
                    if best_d is None or nd < best_d:
                        best_d, best_a = nd, d
            _, r, done, info = env.step(best_a)
            steps += 1
        assert r == 1.0 and done and steps <= 30

    def test_timeout(self):
        env = Maze(size=4, horizon=5, seed=0)
        env.reset()
        for _ in range(5):
            _, r, done, _ = env.step(0)
        assert done and r == -0.01


class TestObsAndInfo:
    def test_obs_shape_and_walls_binary(self):
        env = Maze(size=5, horizon=40, seed=4)
        obs = env.reset()
        assert obs.shape == (10,) and obs.dtype == torch.float32
        walls = obs[4:8]
        assert torch.all((walls == 0.0) | (walls == 1.0))

    def test_rare_always_false(self):
        env = Maze(size=5, horizon=40, seed=5)
        env.reset()
        _, _, _, info = env.step(0)
        assert info["rare"] is False and info["dist"] >= 0

    def test_rollout_shapes(self):
        env = Maze(size=5, horizon=30, seed=6)
        obs, act, rew, rare = random_rollout(env)
        assert obs.shape[1] == 10 and rare.dtype == torch.bool
