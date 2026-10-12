"""ENV-E pilot 反幻觉测试：两个 regime 必须实测动力学不同（slip 机制），
翻转即生效；pilot 管线小规模端到端可跑、τ 计算良定义。"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "env_e_nonstat"))

from src.envs.latent_grid import DIRS, LatentGrid  # noqa: E402
from pilot_adaptation import SLIP_RHO0, SLIP_RHO1, train_pilot  # noqa: E402


def _direction_compliance(slip, seed=7, n=2000):
    """实际位移方向 == 意图动作方向 的占比。

    注（第一性）：slip 是"动作替换为随机"而非"原地不动"——随机动作仍有
    ~3/4 概率位移，故"移动率"对 slip 不敏感（实测 0.89 vs 0.90）；方向服从率
    才是区分 regime 的正确统计量（期望：0.1→~0.93，0.45→~0.66）。
    """
    env = LatentGrid(grid=8, slip=slip, p_threat_move=0.02, horizon=10**9,
                     seed=seed)
    env.reset()
    ok = 0
    for _ in range(n):
        a = env.rng.randrange(4)
        (bx, by), (ax, ay) = env._agent, env._agent
        env.step(a)
        dx, dy = DIRS[a]
        ax, ay = env._agent
        ok += (ax - bx, ay - by) == (dx, dy)
    return ok / n


class TestRegimesDiffer:
    def test_rho1_slips_more(self):
        """ρ1（slip 0.45）的方向服从率显著低于 ρ0（0.1）——regime 真不同。

        实测锚定（seed 7, n=2000）：ρ0≈0.816 / ρ1≈0.570（差 0.245）；
        ρ0 低于理论 (1-slip)+slip/4 ≈0.93 因边缘格挡住意图方向。
        """
        r0 = _direction_compliance(SLIP_RHO0)
        r1 = _direction_compliance(SLIP_RHO1)
        assert r0 > 0.75 and r1 < 0.70, (r0, r1)
        assert r0 - r1 > 0.15

    def test_flip_changes_dynamics(self):
        """训练中翻转 env.slip 立即改变动力学（阶跃实现的根据）。"""
        env = LatentGrid(grid=8, slip=SLIP_RHO0, seed=11)
        env.reset()
        env.slip = SLIP_RHO1
        n, ok = 1000, 0
        for _ in range(n):
            a = env.rng.randrange(4)
            bx, by = env._agent
            env.step(a)
            dx, dy = DIRS[a]
            ok += (env._agent[0] - bx, env._agent[1] - by) == (dx, dy)
        assert ok / n < 0.75


class TestPilotPipeline:
    def test_tiny_run_end_to_end(self, tmp_path, monkeypatch):
        import pilot_adaptation as P
        monkeypatch.setattr(P, "OUT", tmp_path)
        rec = P.train_pilot(0, torch.device("cpu"), d_h=16, pre=6, post=6,
                            eval_every=2, batch=8, t_win=4, hidden=8,
                            ep_per_cycle=3, wake_batches=2, grid=4)
        # 管线级断言（12 cycles 不收敛，τ 逻辑只验证良定义不验证收敛）
        for key in ("m1_at_switch", "m1_end", "m0_end", "m0_pre_end",
                    "rho0_retention_rise"):
            assert isinstance(rec[key], float) and rec[key] == rec[key]
        assert rec["tau50"] is None or rec["tau50"] <= (rec["tau90"] or 10**9)
        rows = [json.loads(l) for l in open(tmp_path / "pilot_s0.jsonl")]
        assert rows and {"pre", "post"} == {r["phase"] for r in rows}
        assert all(0 < r["mse_rho0"] < 10 and 0 < r["mse_rho1"] < 10 for r in rows)
