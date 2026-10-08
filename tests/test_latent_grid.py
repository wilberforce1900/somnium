"""ENV-A latent_grid 测试：动力学 / 终局 / 稀有事件 / 确定性 / 观测有效性。"""
import torch

from src.envs import LatentGrid, guided_rollout, random_rollout


def fixed_env(**kw):
    return LatentGrid(
        grid=5, slip=0.0, p_threat_move=0.0, horizon=20, seed=0,
        fixed_start=(0, 0), fixed_goal=(4, 4), **kw,
    )


class TestDynamics:
    def test_move_right(self):
        env = fixed_env()
        env.reset()
        obs, r, done, _ = env.step(1)  # 右
        assert abs(obs[0].item() - 1 / 5) < 1e-6 and r == -0.01 and not done

    def test_wall_clamp(self):
        env = fixed_env()
        env.reset()
        obs, _, _, _ = env.step(3)  # 左撞墙
        assert obs[0].item() == 0.0 and obs[1].item() == 0.0

    def test_no_slip_deterministic(self):
        env = fixed_env()
        env.reset(seed=7)
        o1, r1, d1, _ = env.step(0)
        env.reset(seed=7)
        o2, r2, d2, _ = env.step(0)
        assert torch.equal(o1, o2) and r1 == r2 and d1 == d2

    def test_reset_seed_reproducible(self):
        a = LatentGrid(grid=6, seed=11).reset()
        b = LatentGrid(grid=6, seed=11).reset()
        assert torch.equal(a, b)


class TestTermination:
    def test_goal_reward(self):
        env = LatentGrid(grid=3, slip=0, p_threat_move=0, horizon=10, seed=0,
                         fixed_start=(0, 0), fixed_goal=(1, 0))
        env.reset()
        _, r, done, _ = env.step(1)
        assert r == 1.0 and done

    def test_hazard_hit(self):
        env = LatentGrid(grid=3, slip=0, p_threat_move=0, horizon=10, seed=0,
                         fixed_start=(0, 0), fixed_goal=(2, 2), fixed_hazard=(1, 0))
        env.reset()
        _, r, done, _ = env.step(1)
        assert r == -1.0 and done

    def test_timeout(self):
        env = LatentGrid(grid=4, slip=0, p_threat_move=0, horizon=3, seed=0,
                         fixed_start=(0, 0), fixed_goal=(3, 3))
        env.reset()
        for i in range(3):
            obs, r, done, _ = env.step(0)
        assert done and r == -0.01


class TestRareEvents:
    def test_threat_move_flags_rare(self):
        env = LatentGrid(grid=4, slip=0, p_threat_move=1.0, horizon=10, seed=0,
                         fixed_start=(0, 0), fixed_goal=(3, 3), fixed_hazard=(1, 1))
        env.reset()
        _, _, _, info = env.step(0)
        assert info["rare"] is True

    def test_no_threat_when_disabled(self):
        env = fixed_env()
        env.reset()
        _, _, _, info = env.step(0)
        assert info["rare"] is False


class TestObservation:
    def test_obs_valid(self):
        env = LatentGrid(grid=6, seed=3)
        obs = env.reset()
        assert obs.shape == (8,) and obs.dtype == torch.float32
        assert torch.all(obs[:-1] <= 1.0) and torch.all(obs[:-1] >= -1.0)
        assert obs[-1].item() == 1.0

    def test_fixed_mismatch_rejected(self):
        try:
            LatentGrid(fixed_start=(0, 0))
        except ValueError:
            pass
        else:
            raise AssertionError("单独给 fixed_start 应报错")


class TestNoHazard:
    def test_with_hazard_false(self):
        """纯导航模式：无 hazard，obs 危险分量恒 0，不可能 -1 终局。"""
        env = LatentGrid(grid=4, slip=0, p_threat_move=0, horizon=20, seed=1,
                         with_hazard=False)
        obs = env.reset()
        assert obs[4].item() == 0.0 and obs[5].item() == 0.0  # 危险增量=0
        for _ in range(20):
            o, r, done, _ = env.step(env.rng.randrange(4))
            assert r >= -0.01
            if done:
                break


class TestRollout:
    def test_random_rollout_shapes(self):
        env = LatentGrid(grid=4, slip=0.1, p_threat_move=0.05, horizon=15, seed=5)
        obs, act, rew, rare = random_rollout(env)
        T = act.shape[0]
        assert obs.shape == (T + 1, 8) and act.dtype == torch.long
        assert rew.shape == (T,) and rare.dtype == torch.bool
        assert act.min() >= 0 and act.max() <= 3

    def test_guided_rollout_reaches_goal(self):
        """P0-1 课程机制：无 slip 全贪心（p=1.0）必达目标。"""
        env = LatentGrid(grid=5, slip=0.0, p_threat_move=0.0, horizon=30, seed=0,
                         fixed_start=(0, 0), fixed_goal=(4, 4), with_hazard=False)
        obs, act, rew, rare = guided_rollout(env, p_greedy=1.0)
        assert rew[-1].item() == 1.0 and act.shape[0] <= 30

    def test_guided_rollout_shapes(self):
        env = LatentGrid(grid=4, slip=0.1, p_threat_move=0.05, horizon=15, seed=6)
        obs, act, rew, rare = guided_rollout(env, p_greedy=0.8)
        assert obs.shape[1] == 8 and act.dtype == torch.long
