"""ITER9 term-scale 反幻觉测试：管线端到端可跑；拆分诊断与冻结仪器在
同种子新环境下逐位等价（护仪器同源的断言）；λ_rew 确实进损失
（同权重探针：total 差恰好 = Δλ·l_rew——防"参数入了构造器但没进损失"的静默）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSplitEquivalence:
    def test_split_matches_pooled_on_fresh_env(self):
        from experiments.env_e_nonstat.pilot_terminal import terminal_acc
        from experiments.term_scale.pilot_terminal_scale import terminal_acc_split
        from src.core import WorldModel
        from src.envs.nonstat_world import NonStatWorld

        torch.manual_seed(0)
        m = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=8,
                       norm_kind="ln_post")
        e1 = NonStatWorld(grid=4, slip=0.1, lam=0.0, horizon=20, seed=77,
                          regime=0)
        e2 = NonStatWorld(grid=4, slip=0.1, lam=0.0, horizon=20, seed=77,
                          regime=0)
        acc, _mae, n = terminal_acc(m, e1, episodes=10)
        ap, an, np_, nn_ = terminal_acc_split(m, e2, episodes=10)
        assert n == np_ + nn_          # 同种子新环境 → 逐位同局
        if n == 0:
            assert acc is None
            return
        pooled = (ap * np_ + an * nn_) / (np_ + nn_)
        assert abs(pooled - acc) < 1e-12


class TestLamRewGoesIntoLoss:
    def test_lam_rew_scales_total(self):
        from src.core import WorldModel

        torch.manual_seed(0)
        m4 = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=8,
                        norm_kind="ln_post", lam_rew=4.0)
        m1 = WorldModel(d_obs=8, n_actions=4, d_h=16, hidden=8,
                        norm_kind="ln_post", lam_rew=1.0)
        m1.load_state_dict(m4.state_dict())
        obs = torch.rand(2, 6, 8)
        act = torch.randint(0, 4, (2, 6))
        obs_next = torch.rand(2, 6, 8)
        rew = torch.zeros(2, 6)
        # spawn 从噪声生（substrate.spawn）——每次前向前重置种子，两次可比
        torch.manual_seed(0)
        t4, p4 = m4.loss_on_batch(obs, act, obs_next, rew)
        torch.manual_seed(0)
        t1, p1 = m1.loss_on_batch(obs, act, obs_next, rew)
        assert abs(p4["rew"] - p1["rew"]) < 1e-7
        assert abs((float(t4) - float(t1)) - 3.0 * p1["rew"]) < 1e-6


class TestScalePipeline:
    def test_tiny_run_end_to_end(self, tmp_path, monkeypatch):
        import experiments.term_scale.pilot_terminal_scale as P
        monkeypatch.setattr(P, "OUT", tmp_path)
        rec = P.train_scale(0, torch.device("cpu"), d_h=16, lam_rew=4.0,
                            cycles=8, eval_every=4, batch=8, t_win=4,
                            hidden=8, ep_per_cycle=3, wake_batches=2)
        for key in ("acc0_at600", "acc0_end", "mae_end",
                    "split_acc_pos", "split_acc_neg"):
            assert rec[key] is None or isinstance(rec[key], float)
        assert rec["lam_rew"] == 4.0 and rec["d_h"] == 16
        rows = [json.loads(line) for line in
                open(tmp_path / "term_scale_dh16_lam4_s0.jsonl")]
        assert len(rows) == 3            # c1, c4, c8
        assert all(0.0 <= r["acc0"] <= 1.0 for r in rows)
