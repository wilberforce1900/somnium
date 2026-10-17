"""ITER14 §1 R4b 反幻觉测试：①开环读数全 k 良定义；②Somnium k=1 与手工重算逐位一致
（护闭环实现正确性——喂自身解码的回路装配无误）；③AR 闭环 k=1 输入=真 o0 时的
预测与教师强制一步一致。"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _tiny_rig():
    from src.core import WorldModel
    from src.dream import EpisodeBuffer
    from src.envs import random_rollout
    from src.envs.rich_world import RichWorld
    from src.substrate import LatentState
    from experiments.r4_baseline.run_r4 import ARTransformer

    torch.manual_seed(0)
    env = RichWorld(complexity=4, seed=7)
    som = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=16,
                     hidden=8, norm_kind="ln_post")
    ar = ARTransformer(d_obs=env.d_obs, n_actions=env.n_actions)
    ebuf = EpisodeBuffer(capacity=99)
    for _ in range(5):
        ebuf.add_episode(*random_rollout(env))
    return som, ar, ebuf, env


class TestOpenLoopWellDefined:
    def test_all_horizons(self):
        import experiments.openloop.pilot_openloop as P

        som, ar, ebuf, _ = _tiny_rig()
        out = P.openloop_eval(som, ar, ebuf, torch.device("cpu"),
                              max_k=4, batch=8, seed=4242)
        for k in "1234":
            assert out["mse_som"][k] > 0
            assert out["mse_ar"][k] > 0
            assert out["mse_copy"][k] > 0
            assert out["r2_som"][k] == out["r2_som"][k]     # 非 NaN
        assert out["compound_som"] >= 1.0 or out["compound_som"] > 0


class TestSomniumClosedLoop:
    def test_k1_matches_manual_recomputation(self):
        """开环 Somnium k=1（接地+一步）与手工重算逐位一致——闭环装配无误。"""
        import experiments.openloop.pilot_openloop as P

        from src.substrate import LatentState

        som, ar, ebuf, _ = _tiny_rig()
        out = P.openloop_eval(som, ar, ebuf, torch.device("cpu"),
                              max_k=4, batch=8, seed=4242)
        torch.manual_seed(4242)
        b = ebuf.sample_windows(8, 5, rng=random.Random(4242))
        o, a = b["obs"], b["act"]
        som.eval()
        with torch.no_grad():
            h = som.ground(o[:, 0])
            x = som._x(o[:, 0], a[:, 0])
            s, _, _, _ = som.taiji.mixed_step_parts(LatentState(h), x)
            mse1 = float(((som.decode_obs(s.h) - o[:, 1]) ** 2).mean())
        som.train()
        assert abs(mse1 - out["mse_som"]["1"]) < 1e-6


class TestARClosedLoop:
    def test_k1_equals_teacher_forced(self):
        """AR 闭环 k=1（输入真 o0）≡ 教师强制一步预测。"""
        import experiments.openloop.pilot_openloop as P

        from experiments.r4_baseline.run_r4 import ARTransformer

        som, ar, ebuf, _ = _tiny_rig()
        out = P.openloop_eval(som, ar, ebuf, torch.device("cpu"),
                              max_k=4, batch=8, seed=4242)
        torch.manual_seed(4242)
        b = ebuf.sample_windows(8, 5, rng=random.Random(4242))
        o, a = b["obs"], b["act"]
        ar.eval()
        with torch.no_grad():
            pred = ar(o[:, :1], a[:, :1])[:, -1]
            mse1 = float(((pred - o[:, 1]) ** 2).mean())
        ar.train()
        assert abs(mse1 - out["mse_ar"]["1"]) < 1e-6


class TestPipeline:
    def test_tiny_run_end_to_end(self, tmp_path, monkeypatch):
        import experiments.openloop.pilot_openloop as P

        monkeypatch.setattr(P, "OUT", tmp_path)
        rec = P.run_r4b(0, torch.device("cpu"), cycles=6, batch=8, t_win=4,
                        hidden=8, ep_per_cycle=2, wake_batches=2)
        rows = [json.loads(x) for x in
                open(tmp_path / "openloop_k4_s0.jsonl")]
        assert rows and "mse_som" in rows[0]
        for k in ("delta_r2_16", "delta_r2_8", "compound_som",
                  "compound_ar", "params_som", "params_ar"):
            assert k in rec
        assert isinstance(rec["delta_r2_16"], float)
