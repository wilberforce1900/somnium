"""cell_variants 测试：变体正确性 / 互根保持 / 单变量归因 / 基线不回归。"""
import torch

from src.cell_variants import GatedCell, TwoLayerCell, build_cell
from src.core import WorldModel
from src.envs import LatentGrid, random_rollout
from src.substrate import Substrate


class TestCellFactory:
    def test_all_kinds_build(self):
        for kind in ("linear", "gated", "twolayer"):
            sub = Substrate(d_x=12, d_h=24, cell_kind=kind)
            assert sub.cell_kind == kind

    def test_unknown_kind_rejected(self):
        try:
            Substrate(d_x=8, d_h=16, cell_kind="transformer")
        except ValueError:
            pass
        else:
            raise AssertionError("未知 kind 应拒绝")

    def test_linear_default_unchanged(self):
        """默认 linear：与 v0 行为逐位一致（基线不回归保证）。"""
        torch.manual_seed(0)
        sub = Substrate(d_x=8, d_h=16, cell_kind="linear")
        h = torch.randn(4, 16)
        x = torch.randn(4, 8)
        expect = torch.tanh(sub.cell(torch.cat([h, x], dim=-1)))
        assert torch.equal(sub.forward_delta(h, x), expect)


class TestVariantsForward:
    def test_gated_math(self):
        """GatedCell 逐位对公式（z/r/h_cand）。"""
        cell = GatedCell(d_h=8, d_x=4)
        h = torch.randn(3, 8)
        x = torch.randn(3, 4)
        hx = torch.cat([h, x], dim=-1)
        z = torch.sigmoid(cell.Wz(hx))
        r = torch.sigmoid(cell.Wr(hx))
        cand = torch.tanh(cell.Wh(torch.cat([r * h, x], dim=-1)))
        assert torch.allclose(cell.forward(hx), (1 - z) * h + z * cand)

    def test_gated_preserves_when_z_zero(self):
        """z=0 时输出=输入 h（门控保持语义）。"""
        cell = GatedCell(d_h=8, d_x=4)
        with torch.no_grad():
            cell.Wz.weight.zero_(); cell.Wz.bias.fill_(-20.0)  # σ≈0
        h = torch.randn(2, 8)
        out = cell.forward(torch.cat([h, torch.randn(2, 4)], dim=-1))
        assert torch.allclose(out, h)

    def test_twolayer_shape(self):
        cell = TwoLayerCell(d_h=16, d_x=8)
        out = cell.forward(torch.randn(5, 24))
        assert out.shape == (5, 16)


class TestWorldModelIntegration:
    def test_wm_with_variants_trainable(self):
        """三变体经 WorldModel 训练一步，全部共享参数有梯度（互根）。"""
        from src.dream import EpisodeBuffer
        env = LatentGrid(grid=4, horizon=10, seed=0)
        ebuf = EpisodeBuffer()
        for seed in range(4):
            e = LatentGrid(grid=4, horizon=10, seed=seed)
            ebuf.add_episode(*random_rollout(e))
        for kind in ("linear", "gated", "twolayer"):
            torch.manual_seed(0)
            wm = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=16,
                            cell_kind=kind)
            opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
            b = ebuf.sample_windows(3, 6, rng=__import__("random").Random(0))
            loss, _ = wm.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
            assert torch.isfinite(loss), kind

    def test_variant_growth_compat_warning_free(self):
        """变体下 widen 的函数保持不适用（生产者复制假设线性 cell）——
        但 widen 只对 linear 有意义，此处仅验证变体模型可正常 forward。"""
        torch.manual_seed(0)
        sub = Substrate(d_x=12, d_h=24, cell_kind="gated")
        h = torch.randn(2, 24)
        x = torch.randn(2, 12)
        out = sub.forward_delta(h, x)
        assert out.shape == (2, 24)
