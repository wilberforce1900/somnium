"""ITER10 梦调度因果反幻觉测试：
①无梦路径与 iter9 冻结实现逐位等价（护"醒-only 对照=iter9 归档"的口径根基）；
②梦管线端到端可跑（前奏真执行、梦 log 落盘、rec 字段良定义）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = ("acc0_at600", "acc0_at1200", "acc0_end", "delta_1200_end", "mae_end",
        "n_end", "split_acc_pos", "split_acc_neg", "split_n_pos", "split_n_neg")


class TestNoDreamEquivalence:
    def test_no_dream_bit_identical_to_iter9(self, tmp_path, monkeypatch):
        """无梦路径 ≡ pilot_terminal_scale.train_scale（逐位）——对照=归档的守门。"""
        import experiments.term_scale.pilot_terminal_scale as P9
        import experiments.term_scale.pilot_dream_semantics as P10

        d9, d10 = tmp_path / "p9", tmp_path / "p10"
        d9.mkdir()
        d10.mkdir()
        monkeypatch.setattr(P9, "OUT", d9)
        monkeypatch.setattr(P10, "OUT", d10)
        kw = dict(d_h=16, cycles=8, eval_every=4, batch=8, t_win=4, hidden=8,
                  ep_per_cycle=3, wake_batches=2)
        r9 = P9.train_scale(0, torch.device("cpu"), **kw)
        r10 = P10.train_dreamsem(0, torch.device("cpu"), dream_batches=0, **kw)
        for k in KEYS:
            assert r9[k] == r10[k], (k, r9[k], r10[k])


class TestDreamPipeline:
    def test_tiny_dream_run(self, tmp_path, monkeypatch):
        import experiments.term_scale.pilot_dream_semantics as P10

        monkeypatch.setattr(P10, "OUT", tmp_path)
        rec = P10.train_dreamsem(0, torch.device("cpu"), d_h=16,
                                 dream_batches=3, cycles=6, eval_every=3,
                                 batch=8, t_win=4, hidden=8,
                                 ep_per_cycle=2, wake_batches=2)
        assert rec["dream_batches"] == 3
        assert rec["schedule"] == "dream_first"
        for k in KEYS:
            assert rec[k] is None or isinstance(rec[k], (int, float))
        dlog = [json.loads(line) for line in
                open(tmp_path / "dream_sem_dh16_s0.dream.jsonl")]
        assert len(dlog) == 3          # 前奏 3 步梦 log
        assert all("energy_probe" in e for e in dlog)
        rows = [json.loads(line) for line in
                open(tmp_path / "dream_sem_dh16_s0.jsonl")]
        assert len(rows) == 3          # c1, c3, c6
