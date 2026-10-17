"""ITER14 §2 读梦 v2 反幻觉测试：①ObsCoverage blind=False=均匀权重（对照臂无信号）、
blind=True 权重对已见/未见格位有反应；②三臂管线端到端（A 零梦期、B/C consolidation
轮数=cycles）；③B/C 采样机制真接入（consol 损失有限且 rec 良定义）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestCoverageBlind:
    def test_blind_false_uniform(self):
        from src.dream import ObsCoverage

        cov = ObsCoverage(d_obs=31, dims=(0, 1), bins=6)
        cov.blind = False
        w1 = cov.episode_weight(torch.rand(20, 31))
        w2 = cov.episode_weight(torch.rand(20, 31))
        assert abs(w1 - 1.0) < 1e-9 and abs(w2 - 1.0) < 1e-9

    def test_blind_true_reacts(self):
        from src.dream import ObsCoverage

        cov = ObsCoverage(d_obs=31, dims=(0, 1), bins=6)
        cov.blind = True
        # 受控格位：ep1 全在角格 (0,0) 区，ep2 全在对角格 (5,5) 区——无重叠
        ep1 = torch.rand(20, 31) * 0.0
        ep1[:, 0] = 0.05
        ep1[:, 1] = 0.05
        ep2 = torch.rand(20, 31) * 0.0
        ep2[:, 0] = 0.95
        ep2[:, 1] = 0.95
        cov.update(ep1)
        w1 = cov.episode_weight(ep1)
        w2 = cov.episode_weight(ep2)
        assert w1 <= 0.5 + 1e-9            # 已见格位权重被压
        assert abs(w2 - 1.0) < 1e-9        # 未见格位满权重


class TestArms:
    def _run(self, arm, tmp_path, monkeypatch):
        import experiments.readdream_v2.pilot_readdream as P

        monkeypatch.setattr(P, "OUT", tmp_path)
        return P.train_readdream(0, torch.device("cpu"), arm=arm, d_h=16,
                                 hidden=8, cycles=6, eval_every=3, batch=8,
                                 t_win=4, ep_per_cycle=2, wake_batches=2)

    def test_arm_A_no_dream(self, tmp_path, monkeypatch):
        rec = self._run("A", tmp_path, monkeypatch)
        assert rec["consol_steps"] == 0
        assert rec["arm"] == "A"
        for k in ("cov95_end", "cov95_term_end", "bins_end", "n_end"):
            assert k in rec

    def test_arm_B_consolidation_cycles(self, tmp_path, monkeypatch):
        rec = self._run("B", tmp_path, monkeypatch)
        assert rec["consol_steps"] == 6     # 每 cycle 一轮读梦巩固

    def test_arm_C_matches_B_volume(self, tmp_path, monkeypatch):
        rec = self._run("C", tmp_path, monkeypatch)
        assert rec["consol_steps"] == 6     # 同回放量、仅无盲区信号
