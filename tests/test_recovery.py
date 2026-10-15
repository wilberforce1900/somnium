"""ITER11 单翻转恢复反幻觉测试：
①wake 臂 PRE 与 iter9 冻结实现逐位等价；②dream 臂 PRE 与 iter10 逐位等价
（护"PRE 段=归档锚点"的口径）；③翻转协议端到端（post 行带 acc1、τ 良定义）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestPreEquivalence:
    def test_wake_pre_bit_identical_to_iter9(self, tmp_path, monkeypatch):
        import experiments.term_scale.pilot_terminal_scale as P9
        import experiments.env_e_v3.pilot_recovery as P11

        d9, d11 = tmp_path / "p9", tmp_path / "p11"
        d9.mkdir()
        d11.mkdir()
        monkeypatch.setattr(P9, "OUT", d9)
        monkeypatch.setattr(P11, "OUT", d11)
        kw = dict(d_h=16, eval_every=4, batch=8, t_win=4, hidden=8,
                  ep_per_cycle=3, wake_batches=2)
        r9 = P9.train_scale(0, torch.device("cpu"), cycles=8, **kw)
        P11.train_recovery(0, torch.device("cpu"), schedule="wake",
                           pre=8, post=0, **kw)
        rows9 = [json.loads(x) for x in
                 open(d9 / "term_scale_dh16_lam1_s0.jsonl")]
        rows11 = [json.loads(x) for x in open(d11 / "recovery_wake_s0.jsonl")]
        assert len(rows9) == len(rows11) == 3          # c1, c4, c8
        for x, y in zip(rows9, rows11):
            assert x["acc0"] == y["acc0"], (x["cycle"], x["acc0"], y["acc0"])
            assert x["mae0"] == y["mae0"]
        assert r9["acc0_at600"] == P9._mean_last(rows11, "acc0")

    def test_dream_pre_bit_identical_to_iter10(self, tmp_path, monkeypatch):
        import experiments.term_scale.pilot_dream_semantics as P10
        import experiments.env_e_v3.pilot_recovery as P11

        d10, d11 = tmp_path / "p10", tmp_path / "p11"
        d10.mkdir()
        d11.mkdir()
        monkeypatch.setattr(P10, "OUT", d10)
        monkeypatch.setattr(P11, "OUT", d11)
        kw = dict(d_h=16, eval_every=4, batch=8, t_win=4, hidden=8,
                  ep_per_cycle=3, wake_batches=2)
        P10.train_dreamsem(0, torch.device("cpu"), dream_batches=3,
                           cycles=8, **kw)
        P11.train_recovery(0, torch.device("cpu"), schedule="dream",
                           dream_batches=3, pre=8, post=0, **kw)
        rows10 = [json.loads(x) for x in
                  open(d10 / "dream_sem_dh16_s0.jsonl")]
        rows11 = [json.loads(x) for x in open(d11 / "recovery_dream_s0.jsonl")]
        assert len(rows10) == len(rows11) == 3
        for x, y in zip(rows10, rows11):
            assert x["acc0"] == y["acc0"], (x["cycle"], x["acc0"], y["acc0"])


class TestFlipPipeline:
    def test_tiny_flip_end_to_end(self, tmp_path, monkeypatch):
        import experiments.env_e_v3.pilot_recovery as P11

        monkeypatch.setattr(P11, "OUT", tmp_path)
        rec = P11.train_recovery(0, torch.device("cpu"), schedule="wake",
                                 pre=6, post=6, eval_every=3, batch=8,
                                 t_win=4, hidden=8, ep_per_cycle=2,
                                 wake_batches=2)
        rows = [json.loads(x) for x in open(tmp_path / "recovery_wake_s0.jsonl")]
        assert {r["phase"] for r in rows} == {"pre", "post"}
        flip = next(r for r in rows if r["cycle"] == 6)
        assert "acc1" in flip                          # 翻转点必评 acc1
        post_rows = [r for r in rows if r["phase"] == "post"]
        assert post_rows and all("acc1" in r for r in post_rows)
        assert rec["seed"] == 0 and rec["schedule"] == "wake"
        for k in ("acc1_flip", "acc1_end", "tau50", "tau90", "cross_pre",
                  "acc0_post_end"):
            assert k in rec


class TestReturnRespectsProtocol:
    def test_dream_schedule_has_dream_log(self, tmp_path, monkeypatch):
        import experiments.env_e_v3.pilot_recovery as P11

        monkeypatch.setattr(P11, "OUT", tmp_path)
        P11.train_recovery(0, torch.device("cpu"), schedule="dream",
                           dream_batches=2, pre=5, post=5, eval_every=5,
                           batch=8, t_win=4, hidden=8, ep_per_cycle=2,
                           wake_batches=2)
        dlog = [json.loads(x) for x in
                open(tmp_path / "recovery_dream_s0.dream.jsonl")]
        assert len(dlog) == 2
