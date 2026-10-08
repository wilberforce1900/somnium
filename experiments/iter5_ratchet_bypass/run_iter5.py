#!/usr/bin/env python
"""迭代 #5(a)：棘轮-绕行实验——"梦=重入学习通道"假设检验（分析性诊断）。

背景（迭代#4）：门是阅历棘轮（换新任务不回落阴）→ 模型缺"重入学习态"；
E3a 显示梦回放护住保持率。假设：梦回放是绕开棘轮的再学习通道。

设计：收敛于任务 A 的模型（dream_first）换训任务 B，两臂（醒更新严格一致）：
    wake  纯醒（=迭代#4 相位 2）
    dream 醒更新 + 每周期 dream_phase（回放全 buffer：旧 A + 新 B）

预注册判定（2026-10-08 锁定）：
    P1 保持率：retention = mse_A(终)/mse_A(相位1末)。
       ✅ ⇔ retention_dream ≤ retention_wake − 0.15（梦臂遗忘少 ≥15pp）
    P3 门再入：相位 2 中梦臂 min(α_B) < 0.5 且 wake 臂 > 0.8 → 绕行发生在门；
       否则绕行只在权重（门保持棘轮）——两者都支持假设，位置不同。
    P2（信息）B 学习速度；P4（信息）阴臂贡献是否在梦臂复活。

用法：.venv/bin/python experiments/iter5_ratchet_bypass/run_iter5.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                        # noqa: E402
from src.dream import (DreamConfig, DreamLog,          # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402
from src.substrate import LatentState                  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CFG = dict(grid=6, horizon=30, slip=0.1, p_threat=0.02, guided_frac=0.3,
           cycles=12, episodes=8, wake_batches=8, batch=16, t_win=8,
           d_h=48, lr=3e-3, eval_eps=8, dream_pre=200, buffer_cap=800)
GOAL_A, GOAL_B = (5, 5), (0, 5)


def make_env(goal, seed):
    return LatentGrid(grid=CFG["grid"], slip=CFG["slip"], p_threat_move=CFG["p_threat"],
                      horizon=CFG["horizon"], seed=seed, with_hazard=False,
                      fixed_start=(0, 0), fixed_goal=goal)


def build(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(GOAL_A, seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=CFG["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    buffer = EpisodeBuffer(capacity=CFG["buffer_cap"])
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=CFG["batch"], t_win=CFG["t_win"]),
                           log=DreamLog(), rng=random.Random(seed))
    evals = {}
    for tag, goal in (("A", GOAL_A), ("B", GOAL_B)):
        ev = make_env(goal, seed + 900 if tag == "A" else seed + 950)
        eb = EpisodeBuffer(capacity=99)
        for _ in range(CFG["eval_eps"]):
            eb.add_episode(*random_rollout(ev))
        evals[tag] = eb
    return env, model, opt, buffer, sched, evals


def probes(model, eval_buf):
    """窗均 α、B 窗阴臂贡献、误差（err_lat）。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, CFG["t_win"], rng=random.Random(4242))
    B, T = b["obs"].shape[0], b["obs"].shape[1]
    alpha = yin_c = err = 0.0
    with torch.no_grad():
        h = model.substrate.spawn(B).h
        for t in range(T):
            x = model._x(b["obs"][:, t], b["act"][:, t])
            dy = model.yang.dt * model.substrate.forward_delta(h, x)
            h_leaf = h.detach().requires_grad_(True)
            with torch.enable_grad():
                e = model.substrate.energy_of(h_leaf).sum()
                (g,) = torch.autograd.grad(e, h_leaf)
            di = -model.yin.lr * g
            a = model.taiji.alpha(LatentState(h), x)
            alpha += float(a.mean())
            yin_c += float(((1 - a) * di).norm(dim=-1).mean())
            tgt = model.embed(b["obs_next"][:, t])
            err += float((h - tgt).norm(dim=-1).mean())
            h = h + a * dy + (1 - a) * di
    model.train()
    return {"alpha": alpha / T, "yin_c": yin_c / T, "err": err / T}


def run_arm(arm, seed):
    env, model, opt, buffer, sched, evals = build(seed)
    # ---- 相位 1：任务 A 收敛（dream_first）----
    for _ in range(CFG["dream_pre"]):
        sched.dream_phase(opt)
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            roll = guided_rollout if env.rng.random() < CFG["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"], rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
    pA1 = probes(model, evals["A"])
    # ---- 相位 2：换训任务 B ----
    if arm == "dream_reset":
        model.taiji.reset_gate()  # 再入梦接口：任务切换时门重置回阴阳各半
    env_b = make_env(GOAL_B, seed + 5)
    trace = []
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            roll = guided_rollout if env_b.rng.random() < CFG["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env_b))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"],
                rng=random.Random(seed * 7777 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
        if arm in ("dream", "dream_reset"):
            sched.dream_phase(opt)  # 回放全 buffer（旧 A + 新 B）
        pB = probes(model, evals["B"])
        pA = probes(model, evals["A"])
        trace.append({"cycle": cycle, "alpha_B": round(pB["alpha"], 4),
                      "yin_B": round(pB["yin_c"], 5), "err_B": round(pB["err"], 4),
                      "err_A": round(pA["err"], 4), "alpha_A": round(pA["alpha"], 4)})
    return {"arm": arm, "seed": seed, "A_end_phase1": pA1["err"],
            "trace": trace,
            "retention": round(trace[-1]["err_A"] / max(pA1["err"], 1e-6), 4),
            "min_alpha_B": round(min(t["alpha_B"] for t in trace), 4),
            "B_final_err": round(trace[-1]["err_B"], 4),
            "yin_B_mean": round(sum(t["yin_B"] for t in trace) / len(trace), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = {}
    for arm in ("wake", "dream", "dream_reset"):
        r = run_arm(arm, args.seed)
        results[arm] = r
        print(f"[{arm} seed{args.seed}] retention(A遗忘比)={r['retention']} "
              f"min α_B={r['min_alpha_B']} B终误差={r['B_final_err']} "
              f"阴臂均值={r['yin_B_mean']}")
        print(f"[{arm}] α_B 轨迹：" + " ".join(f"{t['alpha_B']:.2f}" for t in r["trace"]))
        print(f"[{arm}] err_A 轨迹：" + " ".join(f"{t['err_A']:.2f}" for t in r["trace"]))
    w, d = results["wake"], results["dream"]
    dr = results["dream_reset"]
    p1 = d["retention"] <= w["retention"] - 0.15
    p3_gate = d["min_alpha_B"] < 0.5 and w["min_alpha_B"] > 0.8
    print(f"\nP1 保持率：dream {d['retention']} vs wake {w['retention']} → "
          f"{'✅' if p1 else '❌'}（阈 0.15；本设置同族正迁移掩盖遗忘，方向参考）")
    print(f"P3 门再入：dream min α_B={d['min_alpha_B']} / wake={w['min_alpha_B']} → "
          f"{'绕行发生在门上' if p3_gate else '门保持棘轮，绕行在权重'}")
    print(f"接口臂 dream_reset：min α_B={dr['min_alpha_B']}（重置后是否再爬升见轨迹）"
          f"retention={dr['retention']} B终误差={dr['B_final_err']}（信息性，对照 dream/wake）")
    p = REPO / "experiments/iter5_ratchet_bypass"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"iter5_result_seed{args.seed}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"已写 iter5_result_seed{args.seed}.json")


if __name__ == "__main__":
    main()
