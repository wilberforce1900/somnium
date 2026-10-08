"""读梦（A5 重设版）测试：ObsCoverage / blind 偏置 / 反馈环接线 / var_head。"""
import random

import torch

from src.core import WorldModel
from src.dream import (DreamConfig, DreamLog, DreamScheduler,
                       EpisodeBuffer, ObsCoverage)
from src.envs import LatentGrid, random_rollout


def make_cov(seed=0):
    torch.manual_seed(seed)
    cov = ObsCoverage(d_obs=8, dims=(0, 1), bins=6)
    # 只喂上半区：obs[:,0] ∈ [0.5, 1]
    obs = torch.rand(2000, 8)
    obs[:, 0] = 0.5 + 0.5 * torch.rand(2000)
    cov.update(obs)
    return cov


class TestObsCoverage:
    def test_histogram_counts(self):
        cov = ObsCoverage(d_obs=8, bins=4)
        obs = torch.tensor([[0.1, 0.1], [0.9, 0.9]]).repeat(1, 1)
        obs = torch.stack([torch.tensor([0.1, 0.1] + [0.0] * 6),
                           torch.tensor([0.9, 0.9] + [0.0] * 6)])
        cov.update(obs)
        # bins=4：0.1→格0，0.9→格3
        assert cov.counts[0, 0].item() == 1 and cov.counts[3, 3].item() == 1
        assert cov.n == 2

    def test_marginal_stats(self):
        cov = ObsCoverage(d_obs=4, bins=4)
        obs = torch.randn(500, 4) * 2 + 1
        cov.update(obs)
        assert torch.allclose(cov.mean, obs.mean(0), atol=0.2)
        assert torch.allclose(cov.std, obs.std(0), atol=0.2)

    def test_blind_favors_unvisited(self):
        """盲区采样应集中在上半区未访问格（obs[:,0] 小的格）。"""
        cov = make_cov()
        p = cov.pseudo_obs(400)
        assert p[:, 0].mean() < 0.5  # 盲区在 x 低半区

    def test_uniform_control_no_bias(self):
        """对照臂（blind=False）：格位均匀，无偏置。"""
        cov = make_cov()
        cov.blind = False
        p = cov.pseudo_obs(400)
        assert abs(p[:, 0].mean().item() - 0.5) < 0.08

    def test_pseudo_obs_shape_and_dims(self):
        cov = make_cov()
        torch.manual_seed(7)
        p = cov.pseudo_obs(16)
        torch.manual_seed(7)
        ca, cb = cov._cells(16, cov.blind)
        assert p.shape == (16, 8)
        assert torch.equal(p[:, 0], ca) and torch.equal(p[:, 1], cb)

    def test_read_replay_oversamples_blind(self):
        """读梦 v2：覆盖逆权重使盲区 episode 被过采样。"""
        cov = ObsCoverage(d_obs=2, bins=4)
        cov.update(torch.full((50, 2), 0.9))  # 右上角格重度访问
        buf = EpisodeBuffer()
        buf.episodes = [
            {"obs": torch.full((9, 2), 0.9), "act": torch.zeros(8, dtype=torch.long),
             "rew": torch.zeros(8), "rare": torch.zeros(8, dtype=torch.bool)},
            {"obs": torch.full((9, 2), 0.1), "act": torch.zeros(8, dtype=torch.long),
             "rew": torch.zeros(8), "rare": torch.zeros(8, dtype=torch.bool)},
        ]
        wf = lambda ep: cov.episode_weight(ep["obs"])
        b0 = buf.sample_windows(200, 4, rng=random.Random(0))
        b1 = buf.sample_windows(200, 4, rng=random.Random(0), weight_fn=wf)
        frac0 = float((b0["obs"][:, 0, 0] < 0.5).float().mean())
        frac1 = float((b1["obs"][:, 0, 0] < 0.5).float().mean())
        assert frac1 > frac0 + 0.3, (frac0, frac1)


class TestFeedbackLoop:
    def test_wake_updates_coverage(self):
        """反馈环接线：醒期数据自动喂覆盖统计。"""
        torch.manual_seed(0)
        env = LatentGrid(grid=4, horizon=15, seed=0)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16)
        buf = EpisodeBuffer()
        buf.add_episode(*random_rollout(env))
        cov = ObsCoverage(d_obs=8, bins=4)
        sched = DreamScheduler(model, buf, DreamConfig(batch=4, t_win=4,
                                                        imagine_len=2),
                                log=DreamLog(), coverage=cov)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        assert cov.n == 0
        b = buf.sample_windows(4, 4, rng=random.Random(0))
        sched.wake_update(opt, b)
        assert cov.n == 4 * 4  # (B,T) 全部喂入

    def test_dream_phase_with_coverage(self):
        """读梦开启的梦期可跑，log 带 coverage 字段。"""
        torch.manual_seed(0)
        env = LatentGrid(grid=4, horizon=15, seed=0)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16)
        buf = EpisodeBuffer()
        buf.add_episode(*random_rollout(env))
        cov = ObsCoverage(d_obs=8, bins=4)
        sched = DreamScheduler(model, buf, DreamConfig(batch=4, t_win=4,
                                                        imagine_len=2),
                                log=DreamLog(), coverage=cov)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        summary = sched.dream_phase(opt)
        assert "imagination" in summary
        assert sched.log.last()["coverage"] == cov.covered_frac()

    def test_imagine_uses_coverage_when_present(self):
        """伪观测来自覆盖器（格位合法 = 在 bins 网格上），而非 N(0,1)。"""
        torch.manual_seed(0)
        env = LatentGrid(grid=4, horizon=15, seed=1)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16)
        buf = EpisodeBuffer()
        buf.add_episode(*random_rollout(env))
        cov = ObsCoverage(d_obs=8, bins=3)
        # 喂入特征数据使边际统计非标准正态
        obs = torch.rand(100, 8) * 0.01 + 0.5
        cov.update(obs)
        sched = DreamScheduler(model, buf, DreamConfig(batch=8, imagine_len=2),
                                log=DreamLog(), coverage=cov)
        loss, stats = sched.imagine_loss()
        assert torch.isfinite(loss)


class TestVarHead:
    def test_sigma_positive_and_trained(self):
        torch.manual_seed(0)
        env = LatentGrid(grid=4, horizon=12, seed=0)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16)
        buf = EpisodeBuffer()
        for _ in range(5):
            buf.add_episode(*random_rollout(env))
        b = buf.sample_windows(8, 4, rng=random.Random(0))
        loss, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        assert torch.isfinite(torch.tensor(parts["var_nll"]))
        loss.backward()
        assert model.var_head.weight.grad.abs().sum() > 0
        h = torch.randn(4, 24)
        assert torch.all(model.predict_uncertainty(h) > 0)
