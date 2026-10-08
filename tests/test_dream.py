"""dream.py 测试：缓冲采样 / 四开关 / 梦日志 / 醒期更新 / 全关不变。"""
import random
from copy import deepcopy

import torch

from src.core import WorldModel
from src.dream import (DreamConfig, DreamLog, DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, random_rollout


def make(seed=0, cfg=None):
    torch.manual_seed(seed)
    model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16)
    buf = EpisodeBuffer(capacity=100)
    env = LatentGrid(grid=4, slip=0.1, p_threat_move=0.1, horizon=15, seed=seed)
    for _ in range(10):
        buf.add_episode(*random_rollout(env))
    sched = DreamScheduler(model, buf, cfg, log=DreamLog(), rng=random.Random(seed))
    return model, buf, sched


def fake_episodes(n=10, t=12, rare_ep=0):
    """构造假 episode：obs 全 0，rare_ep 指定的 episode 全 rare。"""
    eps = []
    for i in range(n):
        rare = torch.ones(t, dtype=torch.bool) if i < rare_ep else torch.zeros(t, dtype=torch.bool)
        eps.append({
            "obs": torch.zeros(t + 1, 8),
            "act": torch.zeros(t, dtype=torch.long),
            "rew": torch.zeros(t),
            "rare": rare,
        })
    return eps


class TestBuffer:
    def test_window_shapes_and_shift(self):
        buf = EpisodeBuffer()
        ep_obs = torch.stack([torch.full((8,), float(i)) for i in range(13)])  # T=12
        buf.add_episode(ep_obs, torch.zeros(12, dtype=torch.long),
                        torch.zeros(12), torch.zeros(12, dtype=torch.bool))
        b = buf.sample_windows(4, 6, rng=random.Random(0))
        assert b["obs"].shape == (4, 6, 8) and b["act"].shape == (4, 6)
        assert b["obs_next"].shape == (4, 6, 8)
        # 窗口对齐：obs_next[t] == obs[t+1]（值为索引）
        assert torch.allclose(b["obs_next"][0, 0, 0], b["obs"][0, 1, 0])

    def test_rare_bias_oversamples(self):
        buf = EpisodeBuffer()
        for ep in fake_episodes(n=10, rare_ep=1):  # 1/10 的 episode 全 rare
            buf.episodes.append(ep)
        g0 = buf.sample_windows(300, 6, rare_bias=0.0, rng=random.Random(1))
        g5 = buf.sample_windows(300, 6, rare_bias=5.0, rng=random.Random(1))
        assert g5["rare_frac"] > g0["rare_frac"] + 0.2

    def test_only_last(self):
        buf = EpisodeBuffer()
        for i in range(5):
            buf.add_episode(torch.full((6, 8), float(i)), torch.zeros(5, dtype=torch.long),
                            torch.zeros(5), torch.zeros(5, dtype=torch.bool))
        b = buf.sample_windows(10, 4, rng=random.Random(0), only_last=1)
        vals = {round(v, 4) for v in b["obs"].flatten().tolist()}
        assert vals == {4.0}

    def test_empty_raises(self):
        buf = EpisodeBuffer()
        try:
            buf.sample_windows(4, 4)
        except IndexError:
            pass
        else:
            raise AssertionError("空 buffer 应报错")


class TestDreamPhase:
    def test_all_switches_run_and_log(self):
        model, buf, sched = make()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        summary = sched.dream_phase(opt)
        assert set(summary) == {"imagination", "consolidation", "reverse_learning", "rehearsal"}
        assert len(sched.log) == 1
        e = sched.log.last()
        assert e["hexagram"] is None and "energy_probe" in e

    def test_imagination_only_with_empty_buffer(self):
        torch.manual_seed(0)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16)
        cfg = DreamConfig(imagination=True, consolidation=False,
                          reverse_learning=False, rehearsal=False)
        sched = DreamScheduler(model, EpisodeBuffer(), cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        summary = sched.dream_phase(opt)  # buffer 空，只想象，不崩
        assert "imagination" in summary

    def test_imagination_grads_reach_substrate(self):
        model, _, sched = make()
        loss, stats = sched.imagine_loss()
        loss.backward()
        assert model.substrate.energy_head[0].weight.grad is not None
        assert model.substrate.energy_head[0].weight.grad.abs().sum() > 0

    def test_all_off_leaves_params_unchanged(self):
        model, buf, _ = make(seed=2)
        cfg = DreamConfig(imagination=False, consolidation=False,
                          reverse_learning=False, rehearsal=False)
        sched = DreamScheduler(model, buf, cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        before = deepcopy(list(model.parameters()))
        sched.dream_phase(opt)
        for p0, p1 in zip(before, model.parameters()):
            assert torch.equal(p0, p1)


class TestWakeUpdate:
    def test_overfits_single_batch(self):
        torch.manual_seed(0)
        model, buf, sched = make(seed=0)
        batch = buf.sample_windows(16, 8, rng=random.Random(0))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = sched.wake_update(opt, batch)["pred"]
        for _ in range(25):
            parts = sched.wake_update(opt, batch)
        assert parts["pred"] < first


class TestDreamLog:
    def test_save_and_clear(self, tmp_path):
        log = DreamLog()
        log.append({"t": 0, "components": {}})
        log.append({"t": 1, "components": {}})
        p = tmp_path / "sub" / "dream.jsonl"
        log.save(p)
        assert len(log) == 0
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2 and '"t": 1' in lines[1]
