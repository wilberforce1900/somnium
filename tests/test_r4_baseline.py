"""ITER13 R4 反幻觉测试：①参数匹配（AR vs Somnium 15{,}912 ±10%）；
②因果性（改末位 obs 不影响早位预测）；③锁步管线端到端（双模型同批次、读数良定义）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestParamMatch:
    def test_ar_matches_somnium_within_10pct(self):
        from experiments.r4_baseline.run_r4 import ARTransformer
        from src.core import WorldModel

        som = WorldModel(d_obs=8, n_actions=4, d_h=64, hidden=32,
                         norm_kind="ln_post")
        ar = ARTransformer()
        p_som = sum(p.numel() for p in som.parameters())
        p_ar = sum(p.numel() for p in ar.parameters())
        # 实测锚（论文公式 3d_h²+52d_h+296=15{,}912 为近似；ln_post/门/能量头
        # 实计入后 d_h=64 真值 16{,}048——R4 rec 记录实测值）
        assert 15000 <= p_som <= 17000
        assert abs(p_ar - p_som) / p_som <= 0.10, (p_ar, p_som)


class TestCausality:
    def test_future_obs_does_not_leak_back(self):
        from experiments.r4_baseline.run_r4 import ARTransformer

        torch.manual_seed(0)
        ar = ARTransformer()
        ar.eval()
        obs = torch.rand(2, 8, 8)
        act = torch.randint(0, 4, (2, 8))
        with torch.no_grad():
            out1 = ar(obs, act)
            obs2 = obs.clone()
            obs2[:, -1] += 5.0                # 扰动末位
            out2 = ar(obs2, act)
        assert torch.equal(out1[:, :-1], out2[:, :-1])   # 早位不变（因果）
        assert not torch.allclose(out1[:, -1], out2[:, -1])


class TestLockstepPipeline:
    def test_tiny_dual_train_end_to_end(self, tmp_path, monkeypatch):
        import experiments.r4_baseline.run_r4 as R4

        monkeypatch.setattr(R4, "OUT", tmp_path)
        rec = R4.run_r4(0, torch.device("cpu"), world="rho0", cycles=8,
                        eval_every=4, batch=8, t_win=4, hidden=8,
                        ep_per_cycle=3, wake_batches=2)
        rows = [json.loads(x) for x in open(tmp_path / "r4_rho0_s0.jsonl")]
        assert rows and all("dec_r2_som" in r and "dec_r2_ar" in r for r in rows)
        for k in ("params_som", "params_ar", "delta_end", "dec_r2_som_end",
                  "dec_r2_ar_end", "copy_r2_end"):
            assert k in rec
        assert isinstance(rec["delta_end"], float)
        assert 0 < rec["params_som"] and 0 < rec["params_ar"]
