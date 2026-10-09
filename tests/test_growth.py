"""growth.py 测试：函数保持扩宽（生产者复制+消费者零列）——梦长模型原语。"""
import random

import torch

from src.core import WorldModel
from src.dream import EpisodeBuffer
from src.envs import LatentGrid, random_rollout
from src.growth import widen_world_model
from src.substrate import LatentState


def small(seed=0, d_h=16):
    torch.manual_seed(seed)
    return WorldModel(d_obs=8, n_actions=4, d_h=d_h, hidden=16)


class TestFunctionPreservation:
    def test_exact_any_extension(self):
        """noise=0：任意扩展初始态（无需复制）输出逐位保持——消费者零列保证。"""
        m = small(seed=3, d_h=16)
        obs = torch.randn(6, 8)
        act = torch.randint(0, 4, (6,))
        h0 = torch.randn(6, 16)
        x_pre = m._x(obs, act)
        s_pre, a_pre = m.taiji.mixed_step(LatentState(h0), x_pre)
        r_pre = m.predict_reward(s_pre.h)
        o_pre = m.decode_obs(s_pre.h)

        widen_world_model(m, 24, rng=random.Random(42), noise=0.0)
        h0_ext = torch.cat([h0, torch.randn(6, 8)], dim=1)  # 新维任意值
        x_post = m._x(obs, act)
        s_post, a_post = m.taiji.mixed_step(LatentState(h0_ext), x_post)
        assert torch.allclose(a_pre, a_post, atol=1e-6)
        assert torch.allclose(r_pre, m.predict_reward(s_post.h), atol=1e-6)
        assert torch.allclose(o_pre, m.decode_obs(s_post.h), atol=1e-6)

    def test_multistep_trajectory_preserved(self):
        """T=6 连续 rollout：同一 spawn 种子两侧对照，输出头全程保持。"""
        m = small(seed=5, d_h=16)
        obs = torch.randn(4, 6, 8)
        act = torch.randint(0, 4, (4, 6))
        torch.manual_seed(9)                      # 控制 pre 侧 spawn
        h_pre, _ = m(obs, act)
        r_pre = m.predict_reward(h_pre)

        widen_world_model(m, 24, rng=random.Random(7), noise=0.0)
        torch.manual_seed(9)                      # 同种子重建初始 16 维
        h0 = torch.randn(4, 16)
        h = torch.cat([h0, torch.zeros(4, 8)], dim=1)
        with torch.no_grad():
            for t in range(6):
                h, _ = m.rollout_step(h, obs[:, t], act[:, t])
        assert torch.allclose(r_pre, m.predict_reward(h), atol=1e-5)

    def test_noise_is_small_perturbation(self):
        """noise=0.01：输出偏差有界（O(ε) 级，非重置）。"""
        m = small(seed=6, d_h=16)
        obs = torch.randn(6, 8)
        act = torch.randint(0, 4, (6,))
        h0 = torch.randn(6, 16)
        s_pre, _ = m.taiji.mixed_step(LatentState(h0), m._x(obs, act))
        r_pre = m.predict_reward(s_pre.h)
        widen_world_model(m, 24, rng=random.Random(8), noise=0.01)
        h0_ext = torch.cat([h0, torch.zeros(6, 8)], dim=1)
        s_post, _ = m.taiji.mixed_step(LatentState(h0_ext), m._x(obs, act))
        assert (m.predict_reward(s_post.h) - r_pre).abs().max() < 0.05


class TestShapesAndTraining:
    def test_shapes_consistent(self):
        m = small(seed=0, d_h=12)
        widen_world_model(m, 20, rng=random.Random(1), noise=0.01)
        obs = torch.randn(4, 6, 8)
        act = torch.randint(0, 4, (4, 6))
        h, alphas = m(obs, act)
        assert h.shape == (4, 20) and alphas.shape == (4, 6, 1)
        assert m.substrate.d_x == 20 + 4

    def test_new_dims_wake_up_in_training(self):
        """新维可被训练唤醒：扩宽后训 200 步，消费者新列梯度非零（影子苏醒）。"""
        env = LatentGrid(grid=4, slip=0.1, p_threat_move=0.05, horizon=12, seed=0)
        buf = EpisodeBuffer()
        for _ in range(8):
            buf.add_episode(*random_rollout(env))
        m = small(seed=1, d_h=16)
        widen_world_model(m, 24, rng=random.Random(2), noise=0.01)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for i in range(200):
            b = buf.sample_windows(8, 6, rng=random.Random(i))
            loss, _ = m.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        head_new = m.reward_head.weight.data[:, 16:]
        assert head_new.abs().sum() > 1e-6  # 消费者新列已醒来

    def test_plan_and_spawn_after_growth(self):
        m = small(seed=2, d_h=16)
        widen_world_model(m, 24, rng=random.Random(3), noise=0.01)
        obs = torch.randn(8)
        h = m.substrate.spawn(1).h[0]
        assert h.shape == (24,)
        a, _ = m.plan(obs, h, horizon=3, k=8)
        assert a in range(4)
