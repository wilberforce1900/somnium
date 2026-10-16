"""ITER13 R3 反幻觉测试：①镜像数学（x 场翻转、其余不动）；
②管线端到端（flip/end 两行、retention 良定义）；
③replay_old 臂标记与输出齐全。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestMirror:
    def test_mirror_flips_x_fields_only(self):
        from experiments.forget_construct.pilot_forget import mirror_obs

        obs = torch.tensor([0.25, 0.5, 0.1, 0.2, -0.3, 0.4, 0.6, 1.0])
        m = mirror_obs(obs)
        assert torch.equal(m[[0, 2, 4]], torch.tensor([0.75, -0.1, 0.3]))
        assert torch.equal(m[[1, 3, 5, 6, 7]], obs[[1, 3, 5, 6, 7]])

    def test_mirror_2d_episode_and_short_rows(self):
        """2D episode 张量按列镜像；短局（3 步=4 行）不越界。"""
        from experiments.forget_construct.pilot_forget import mirror_obs

        obs = torch.rand(4, 8)          # T+1=4 的短局
        m = mirror_obs(obs)
        assert torch.allclose(m[:, 0], 1.0 - obs[:, 0])
        assert torch.allclose(m[:, 2], -obs[:, 2])
        assert torch.allclose(m[:, 4], -obs[:, 4])
        assert torch.equal(m[:, [1, 3, 5, 6, 7]], obs[:, [1, 3, 5, 6, 7]])


class TestForgetPipeline:
    def test_tiny_wake_run(self, tmp_path, monkeypatch):
        import experiments.forget_construct.pilot_forget as P

        monkeypatch.setattr(P, "OUT", tmp_path)
        rec = P.train_forget(0, torch.device("cpu"), arm="wake", d_h=16,
                             hidden=8, pre=6, post=6, batch=8, t_win=4,
                             ep_per_cycle=2, wake_batches=2)
        rows = [json.loads(x) for x in
                open(tmp_path / "forget_wake_s0.jsonl")]
        assert len(rows) == 2                     # flip + end
        assert rows[0]["phase"] == "pre" and rows[1]["phase"] == "post"
        for k in ("retention_decode", "retention_latent",
                  "acc_old_flip", "acc_old_end", "replay_old"):
            assert k in rec
        if rec["retention_decode"] is not None:
            assert rec["retention_decode"] > 0

    def test_replay_old_arm(self, tmp_path, monkeypatch):
        import experiments.forget_construct.pilot_forget as P

        monkeypatch.setattr(P, "OUT", tmp_path)
        rec = P.train_forget(0, torch.device("cpu"), arm="replay_old",
                             d_h=16, hidden=8, pre=6, post=6, batch=8,
                             t_win=4, ep_per_cycle=2, wake_batches=2)
        assert rec["replay_old"] is True
        assert rec["arm"] == "replay_old"
        assert rec["retention_decode"] is not None
