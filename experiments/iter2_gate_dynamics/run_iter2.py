#!/usr/bin/env python
"""迭代 #2：醒阴梦阳的门控动力学深挖（分析性诊断，不进 registry）。

现象（diag_mpc_gap 留档）：wake 训练终态 α≈0.17（偏阴），
dream_first（梦预训练后）α≈0.96（纯阳）——与 taiji 注释的民俗映射
（醒偏阳、梦偏阴）相反，需查明机制与后果。

三问：
  P1 α 轨迹：门何时/多快滑向极性？梦醒交替后是否回摆（粘滞）？
  P2 推手分解：想象目标两项——方差下限 div（奖多样、罚收缩→应推阳）
     与能量平滑 smooth（削梯度→阴步变小）——谁是主推手？
     消融：{full, div_only, smooth_only, none} × div_target/lam_smooth 置零。
  P3 极性反事实：训好模型运行时强行翻转 gate_mode（learned/fixed_yang/fixed_yin），
     测 pred_mse 与 MPC——极性是承重的还是表象的（与 A2 判定互证）。

用法：.venv/bin/python experiments/iter2_gate_dynamics/run_iter2.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/e0_dream_first"))

from run_e0 import FULL, mpc_return                    # noqa: E402
from src.core import WorldModel                        # noqa: E402
from src.dream import (DreamConfig, DreamLog,          # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def build(seed, cfg):
    torch.manual_seed(seed)
    random.seed(seed)
    env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                     horizon=cfg["horizon"], seed=seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=cfg["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=cfg["batch"], t_win=cfg["t_win"]),
                           log=DreamLog(), rng=random.Random(seed))
    ev = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                    horizon=cfg["horizon"], seed=seed + 999)
    eval_buf = EpisodeBuffer(capacity=9999)
    for _ in range(cfg["eval_episodes"]):
        eval_buf.add_episode(*random_rollout(ev))
    return env, model, opt, buffer, sched, eval_buf


def alpha_on_windows(model, eval_buf):
    """醒语境 α：固定 eval 窗口上的门控均值/标准差。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, FULL["t_win"], rng=random.Random(4242))
    alphas = []
    with torch.no_grad():
        h = model.substrate.spawn(b["obs"].shape[0]).h
        for t in range(b["obs"].shape[1]):
            h, a = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            alphas.append(a)
    model.train()
    alphas = torch.cat(alphas).squeeze(-1)
    return float(alphas.mean()), float(alphas.std())


def pred_mse_on(model, eval_buf):
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, FULL["t_win"], rng=random.Random(4242))
    _, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
    model.train()
    return parts["pred"]


def train_logged(schedule, seed):
    """P1：带 α 轨迹的完整训练（梦预训练每 20 批、醒期每 cycle 记一次）。"""
    cfg = FULL
    env, model, opt, buffer, sched, eval_buf = build(seed, cfg)
    trace = {"schedule": schedule, "seed": seed, "dream": [], "wake": []}
    if schedule == "dream_first":
        for i in range(cfg["dream_pre_batches"]):
            sched.dream_phase(opt)
            if (i + 1) % 20 == 0:
                m, s = alpha_on_windows(model, eval_buf)
                trace["dream"].append(round(m, 4))
    for cycle in range(cfg["cycles"]):
        for _ in range(cfg["episodes_per_cycle"]):
            roll = guided_rollout if env.rng.random() < cfg["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(cfg["wake_batches"]):
            b = buffer.sample_windows(
                cfg["batch"], cfg["t_win"],
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=cfg["episodes_per_cycle"])
            sched.wake_update(opt, b)
        m, s = alpha_on_windows(model, eval_buf)
        trace["wake"].append(round(m, 4))
    return model, eval_buf, trace


def imagine_ablation(name, div_target, lam_smooth, seed):
    """P2：纯想象预训练（200 批）后的门极性——div/smooth 谁在推阳。"""
    cfg = FULL
    env, model, opt, buffer, sched, eval_buf = build(seed, cfg)
    dcfg = DreamConfig(imagination=True, consolidation=False, reverse_learning=False,
                       rehearsal=False, batch=cfg["batch"], t_win=cfg["t_win"],
                       div_target=div_target, lam_smooth=lam_smooth)
    sched.cfg = dcfg
    dream_alpha = None
    for i in range(cfg["dream_pre_batches"]):
        summary = sched.dream_phase(opt)
        st = summary.get("imagination")
        if isinstance(st, dict) and "alpha" in st:
            dream_alpha = st["alpha"]
    m, s = alpha_on_windows(model, eval_buf)
    return {"config": name, "seed": seed, "alpha_wake_ctx": round(m, 4),
            "alpha_dream_ctx": round(dream_alpha, 4) if dream_alpha is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {}

    print("== P1 α 轨迹 ==")
    models = {}
    for schedule in ("wake_only", "dream_first"):
        model, eval_buf, trace = train_logged(schedule, args.seed)
        models[schedule] = (model, eval_buf)
        out.setdefault("p1", []).append(trace)
        m, s = alpha_on_windows(model, eval_buf)
        print(f"  {schedule}: 梦期α轨迹 {trace['dream'][:5]}...{trace['dream'][-1] if trace['dream'] else '-'}"
              f" | 醒期α轨迹 {trace['wake'][:4]}...{trace['wake'][-1]} | 终态 {m:.3f}±{s:.3f}")

    print("== P2 想象目标分解（纯想象预训练 → 门极性）==")
    for name, (dt, ls) in {"full": (0.3, 0.1), "div_only": (0.3, 0.0),
                           "smooth_only": (0.0, 0.1), "none": (0.0, 0.0)}.items():
        for seed in (0, 1):
            r = imagine_ablation(name, dt, ls, seed)
            out.setdefault("p2", []).append(r)
            print(f"  {name:12s} seed{seed}: 醒语境α={r['alpha_wake_ctx']:.3f} "
                  f"梦语境α={r['alpha_dream_ctx']}")

    print("== P3 极性反事实（运行时翻转 gate_mode）==")
    for schedule, (model, eval_buf) in models.items():
        original = model.taiji.gate_mode
        for mode in ("learned", "fixed_yang", "fixed_yin"):
            model.taiji.gate_mode = mode
            pm = pred_mse_on(model, eval_buf)
            mr = mpc_return(model, FULL, args.seed + 500)
            out.setdefault("p3", {})[f"{schedule}/{mode}"] = {
                "pred_mse": round(pm, 4), "mpc": round(mr, 3)}
            print(f"  {schedule} × {mode:10s}: pred_mse={pm:.4f} mpc={mr:+.3f}")
        model.taiji.gate_mode = original

    p = REPO / "experiments/iter2_gate_dynamics/iter2_result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"已写 {p.name}")


if __name__ == "__main__":
    main()
