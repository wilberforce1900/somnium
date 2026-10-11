"""state 归一化测试（ITER7 §2.5 (b) 线）：出口 LN/RMS 数学正确 / ln_pre 只动
cell 输入 / none 逐位不回归 / 归一化下端到端可训练 / 生长守卫。"""
import random

import torch

from src.core import WorldModel
from src.dream import EpisodeBuffer
from src.envs import LatentGrid, random_rollout
from src.substrate import LatentState, RMSNorm, Substrate


def _hx(B=4, d_h=16, d_x=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, d_h), torch.randn(B, d_x)


class TestNormKinds:
    def test_unknown_kind_rejected(self):
        try:
            Substrate(d_x=8, d_h=16, norm_kind="batchnorm")
        except ValueError:
            pass
        else:
            raise AssertionError("未知 norm_kind 应拒绝")

    def test_rmsnorm_math(self):
        rms = RMSNorm(8)
        x = torch.randn(3, 8)
        out = rms(x)
        assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(3), atol=1e-5)

    def test_none_forward_bitwise_unchanged(self):
        """norm=none：forward_delta 与 v0 逐位一致、norm_state 恒等。"""
        sub = Substrate(d_x=8, d_h=16, norm_kind="none")
        h, x = _hx()
        expect = torch.tanh(sub.cell(torch.cat([h, x], dim=-1)))
        assert torch.equal(sub.forward_delta(h, x), expect)
        assert sub.norm_state(h) is h
        assert sub.pre_cell_input(h) is h


class TestPostNorm:
    def test_ln_post_standardizes_state(self):
        """ln_post：出口每样本特征零均值、单位方差（仿射初始 γ=1/β=0）。"""
        torch.manual_seed(0)
        sub = Substrate(d_x=8, d_h=16, norm_kind="ln_post")
        h = sub.spawn(4).h
        out = sub.norm_state(h)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-5)
        assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(4), atol=1e-4)

    def test_rms_post_scales_state(self):
        """rms_post：出口均方 ≈1；逐样本均值不被去除（≠ LN）。"""
        torch.manual_seed(0)
        sub = Substrate(d_x=8, d_h=16, norm_kind="rms_post")
        h = sub.spawn(4).h * 3.7
        out = sub.norm_state(h)
        assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(4), atol=1e-4)

    def test_ln_post_in_mixed_step(self):
        """端到端：经 taiji 混合步后出口已归一化、分量探针返回双步幅。"""
        torch.manual_seed(0)
        wm = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=8, norm_kind="ln_post")
        h = torch.randn(5, 16)
        obs = torch.randn(5, 8)
        act = torch.randint(0, 4, (5,))
        s, a, dh_y, dh_i = wm.taiji.mixed_step_parts(
            LatentState(h), wm._x(obs, act))
        assert torch.allclose(s.h.mean(dim=-1), torch.zeros(5), atol=1e-5)
        assert dh_y.shape == (5, 16) and dh_i.shape == (5, 16)
        assert a.shape == (5, 1)


class TestPreNorm:
    def test_ln_pre_only_affects_cell_input(self):
        """ln_pre：cell 输入侧被归一化；出口状态保持原样（norm_state 恒等）。"""
        torch.manual_seed(0)
        sub = Substrate(d_x=8, d_h=16, norm_kind="ln_pre")
        h, x = _hx()
        hn = sub.state_norm(h)
        expect = torch.tanh(sub.cell(torch.cat([hn, x], dim=-1)))
        assert torch.allclose(sub.forward_delta(h, x), expect)
        assert sub.norm_state(h) is h
        # cell_preact 与 forward_delta 同路径（未过 tanh）
        manual = torch.cat([hn, x], dim=-1) @ sub.cell.weight.T + sub.cell.bias
        assert torch.allclose(sub.cell_preact(h, x), manual, atol=1e-5)


class TestIntegration:
    def test_trainable_with_norms(self):
        """四种 norm_kind 下 WorldModel 训练一步：loss 有限、归一化参数有梯度。"""
        ebuf = EpisodeBuffer()
        for seed in range(3):
            e = LatentGrid(grid=4, horizon=10, seed=seed)
            ebuf.add_episode(*random_rollout(e))
        for nk in ("none", "ln_post", "rms_post", "ln_pre"):
            torch.manual_seed(0)
            wm = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=8, norm_kind=nk)
            opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
            b = ebuf.sample_windows(3, 6, rng=random.Random(0))
            loss, _ = wm.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert torch.isfinite(loss), nk
            if nk != "none":
                assert wm.substrate.state_norm.weight.grad is not None, nk

    def test_norm_trains_downward(self):
        """烟测：ln_post 下数十步循环 loss 末段低于初值。"""
        torch.manual_seed(0)
        ebuf = EpisodeBuffer()
        for s in range(4):
            e = LatentGrid(grid=4, horizon=10, seed=s)
            ebuf.add_episode(*random_rollout(e))
        wm = WorldModel(d_obs=8, n_actions=4, d_h=32, hidden=16, norm_kind="ln_post")
        opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        losses = []
        for i in range(30):
            b = ebuf.sample_windows(8, 6, rng=random.Random(i))
            loss, _ = wm.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        assert sum(losses[-5:]) / 5 < losses[0]

    def test_widen_rejects_norm(self):
        """生长守卫：归一化模型 widen 显式拒绝（非静默错误）。"""
        from src.growth import widen_world_model
        wm = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=8, norm_kind="ln_post")
        try:
            widen_world_model(wm, 24)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("归一化模型 widen 应显式拒绝")
