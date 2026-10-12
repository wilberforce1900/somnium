"""混合回放测试（iter8 阶段 0）：批内新鲜/全缓冲各半、recent 窗口约束、
极端 recent_frac、小缓冲、rare_frac 聚合、确定性。episode 身份用恒定奖励编码。"""
import random

import torch

from src.dream import EpisodeBuffer


def _ep(idx, T=12, d_obs=4):
    obs = torch.randn(T + 1, d_obs)
    act = torch.randint(0, 2, (T,))
    rew = torch.full((T,), float(idx))  # 窗口奖励均值 == episode 身份
    rare = torch.zeros(T, dtype=torch.bool)
    return obs, act, rew, rare


def _buf(n, cap=100):
    buf = EpisodeBuffer(capacity=cap)
    for i in range(n):
        buf.add_episode(*_ep(i))
    return buf


class TestMixedReplay:
    def test_split_counts_and_recent_window(self):
        buf = _buf(30)
        b = buf.sample_windows_mixed(32, 8, rng=random.Random(0),
                                     only_last=5, recent_frac=0.5)
        assert b["obs"].shape == (32, 8, 4)
        # obs_next 严格右移一位（对齐长度比较）
        assert torch.equal(b["obs_next"][:, :-1], b["obs"][:, 1:])
        rews = b["rew"].mean(dim=1)
        # 前 16 个窗口（round(32×0.5)）来自最新 5 局（idx 25..29）
        assert (rews[:16] >= 24.5).all(), "新鲜半必须全部来自 only_last 窗口"
        # 后 16 个来自全缓冲——必含旧局
        assert (rews[16:] < 24.5).any(), "全缓冲半应覆盖旧 episode"

    def test_recent_frac_extremes(self):
        buf = _buf(12)
        b = buf.sample_windows_mixed(8, 8, rng=random.Random(1),
                                     only_last=4, recent_frac=1.0)
        assert (b["rew"].mean(dim=1) >= 7.5).all()  # 全部来自 idx 8..11
        b2 = buf.sample_windows_mixed(8, 8, rng=random.Random(1),
                                      only_last=4, recent_frac=0.0)
        assert b2["obs"].shape[0] == 8

    def test_small_buffer_no_crash(self):
        buf = EpisodeBuffer()
        buf.add_episode(*_ep(0))
        b = buf.sample_windows_mixed(4, 8, rng=random.Random(2),
                                     only_last=8, recent_frac=0.5)
        assert b["obs"].shape == (4, 8, 4)

    def test_rare_frac_weighted_aggregate(self):
        buf = _buf(10, cap=50)
        for ep in buf.episodes[-2:]:
            ep["rare"] = torch.ones_like(ep["rare"])
        b = buf.sample_windows_mixed(16, 8, rng=random.Random(3),
                                     only_last=3, recent_frac=0.5)
        assert 0.0 <= b["rare_frac"] <= 1.0

    def test_deterministic_given_rng(self):
        buf = _buf(20)
        a = buf.sample_windows_mixed(8, 8, rng=random.Random(7),
                                     only_last=4, recent_frac=0.5)
        c = buf.sample_windows_mixed(8, 8, rng=random.Random(7),
                                     only_last=4, recent_frac=0.5)
        assert torch.equal(a["obs"], c["obs"])

    def test_short_recent_episodes_width_align(self):
        """P0-A 实崩复现：最近 only_last 条全部短于 t_win 时，新鲜半被
        sample_windows 降级为更短窗口 → 两部宽度不一致，cat 前必须对齐。"""
        buf = EpisodeBuffer(capacity=50)
        for i in range(20):
            buf.add_episode(*_ep(i, T=20))    # 旧局长
        for i in range(20, 28):
            buf.add_episode(*_ep(i, T=5))     # 最新 8 条全短（< t_win=8）
        b = buf.sample_windows_mixed(16, 8, rng=random.Random(5),
                                     only_last=8, recent_frac=0.5)
        w = b["obs"].shape[1]
        assert w == 5, "应全部对齐到短窗宽（sample_windows 降级行为）"
        assert torch.equal(b["obs_next"][:, :-1], b["obs"][:, 1:])
        # 新鲜半（前 8 行）仍来自最新 8 局（idx 20..27）
        assert (b["rew"].mean(dim=1)[:8] >= 19.5).all()
        # 全缓冲半（后 8 行）必含旧局
        assert (b["rew"].mean(dim=1)[8:] < 19.5).any()
