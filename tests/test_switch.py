"""ITER12 λ 阶梯反幻觉测试：
①wake 臂 PRE 与 iter9 冻结实现逐位等价；②dream 臂 PRE 与 iter10 逐位等价；
③切换相位端到端（sw 行带 acc1/regime/acc_sw，acc_sw=当前 regime 半，开关计数活）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestPreEquivalence:
    def test_wake_pre_bit_identical_to_iter9(self, tmp_path, monkeypatch):
        import experiments.term_scale.pilot_terminal_scale as P9
        import experiments.env_e_v3.pilot_switch as P12

        d9, d12 = tmp_path / "p9", tmp_path / "p12"
        d9.mkdir()
        d12.mkdir()
        monkeypatch.setattr(P9, "OUT", d9)
        monkeypatch.setattr(P12, "OUT", d12)
        kw = dict(d_h=16, eval_every=4, batch=8, t_win=4, hidden=8,
                  ep_per_cycle=3, wake_batches=2)
        P9.train_scale(0, torch.device("cpu"), cycles=8, **kw)
        P12.train_switch(0, torch.device("cpu"), schedule="wake", dwell=50,
                         pre=8, post=0, **kw)
        rows9 = [json.loads(x) for x in
                 open(d9 / "term_scale_dh16_lam1_s0.jsonl")]
        rows12 = [json.loads(x) for x in
                  open(d12 / "switch_wake_dwell50_s0.jsonl")]
        assert len(rows9) == len(rows12) == 3
        for x, y in zip(rows9, rows12):
            assert x["acc0"] == y["acc0"], (x["cycle"], x["acc0"], y["acc0"])
            assert x["mae0"] == y["mae0"]

    def test_dream_pre_bit_identical_to_iter10(self, tmp_path, monkeypatch):
        import experiments.term_scale.pilot_dream_semantics as P10
        import experiments.env_e_v3.pilot_switch as P12

        d10, d12 = tmp_path / "p10", tmp_path / "p12"
        d10.mkdir()
        d12.mkdir()
        monkeypatch.setattr(P10, "OUT", d10)
        monkeypatch.setattr(P12, "OUT", d12)
        kw = dict(d_h=16, eval_every=4, batch=8, t_win=4, hidden=8,
                  ep_per_cycle=3, wake_batches=2)
        P10.train_dreamsem(0, torch.device("cpu"), dream_batches=3,
                           cycles=8, **kw)
        P12.train_switch(0, torch.device("cpu"), schedule="dream",
                         dream_batches=3, dwell=50, pre=8, post=0, **kw)
        rows10 = [json.loads(x) for x in
                  open(d10 / "dream_sem_dh16_s0.jsonl")]
        rows12 = [json.loads(x) for x in
                  open(d12 / "switch_dream_dwell50_s0.jsonl")]
        assert len(rows10) == len(rows12) == 3
        for x, y in zip(rows10, rows12):
            assert x["acc0"] == y["acc0"], (x["cycle"], x["acc0"], y["acc0"])


class TestSwitchPipeline:
    def test_tiny_switch_end_to_end(self, tmp_path, monkeypatch):
        import experiments.env_e_v3.pilot_switch as P12

        monkeypatch.setattr(P12, "OUT", tmp_path)
        rec = P12.train_switch(0, torch.device("cpu"), schedule="wake",
                               dwell=3, pre=6, post=9, eval_every=3,
                               batch=8, t_win=4, hidden=8, ep_per_cycle=2,
                               wake_batches=2)
        rows = [json.loads(x) for x in
                open(tmp_path / "switch_wake_dwell3_s0.jsonl")]
        sw = [r for r in rows if r["phase"] == "sw"]
        assert sw and all("acc_sw" in r for r in sw)
        for r in sw:
            assert r["regime"] in (0, 1)
            assert r["acc_sw"] == (r["acc0"] if r["regime"] == 0 else r["acc1"])
        assert rec["dwell"] == 3
        assert abs(rec["lam"] - 1.0 / 3) < 1e-12
        assert rec["switches"] > 0            # lam>0 → 切换确实在发生
        for k in ("acc_sw_end", "acc_sw_phase", "acc0_sw_end", "acc1_sw_end",
                  "cross_pre", "acc1_at_switch"):
            assert k in rec
