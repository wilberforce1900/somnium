"""ITER13 R2 反幻觉测试：①噪声确实施加（同环境两档 σ 的采集 obs 不同）；
②cov95 仪器良定义（覆盖 ∈[0,1]、分箱 5 段、终局子集一致）；
③管线端到端（k5 环境可跑、rec 字段齐）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestNoiseApplied:
    def test_noise_changes_obs(self):
        from experiments.cal_margin.pilot_cal_margin import noisy_obs

        torch.manual_seed(0)
        obs = torch.zeros(4, 31)
        assert torch.equal(noisy_obs(obs, 0.0), obs)
        assert not torch.allclose(noisy_obs(obs, 0.2), obs)
        n = noisy_obs(torch.zeros(200, 31), 0.2)
        assert abs(float(n.std()) - 0.2) < 0.05   # 幅度≈σ_n


class TestCov95WellDefined:
    def test_cov95_bounds_and_bins(self, tmp_path, monkeypatch):
        import experiments.cal_margin.pilot_cal_margin as P

        from src.core import WorldModel
        from src.envs.rich_world import RichWorld

        torch.manual_seed(0)
        m = WorldModel(d_obs=31, n_actions=4, d_h=16, hidden=8,
                       norm_kind="ln_post")
        ev = RichWorld(complexity=5, seed=7)
        c = P.measure_cov95(m, ev, 0.2, episodes=3)
        assert 0.0 <= c["cov95"] <= 1.0
        assert len(c["bins"]) == 5
        assert all(x is None or 0.0 <= x <= 1.0 for x in c["bins"])
        assert c["n"] > 0
        assert c["cov95_term"] is None or 0.0 <= c["cov95_term"] <= 1.0


class TestPipelineEndToEnd:
    def test_tiny_k5_run(self, tmp_path, monkeypatch):
        import experiments.cal_margin.pilot_cal_margin as P

        monkeypatch.setattr(P, "OUT", tmp_path)
        rec = P.train_cal(0, torch.device("cpu"), d_h=16, sigma_n=0.0,
                          cycles=6, eval_every=3, batch=8, t_win=4, hidden=8,
                          ep_per_cycle=2, wake_batches=2)
        rows = [json.loads(x) for x in
                open(tmp_path / "cal_k5_dh16_sig0_s0.jsonl")]
        assert rows and all("cov95" in r for r in rows)
        for k in ("cov95_end", "cov95_term_end", "bins_end", "n_end",
                  "sigma_n", "d_h"):
            assert k in rec
        assert rec["k"] == 5 and rec["sigma_n"] == 0.0
