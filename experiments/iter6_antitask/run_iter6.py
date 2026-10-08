#!/usr/bin/env python
"""迭代 #6(a)：反相关任务对重测梦-保持率（修复迭代#5 的同族正迁移混淆）。

任务构造：同一网格/同一目标几何，任务 B 的动作语义整体反转
（PERM=(2,3,0,1)：标签"上"执行"下"…）——B 的每条 (o, a, o') 数据都与 A 矛盾，
共享动力学权重必受干扰 → 真遗忘压力。

预注册（2026-10-09 锁定）：
    G0 设置有效门：wake 臂训 B 期间 err_A 必须上升（retention_wake > 1.05），
       否则判定设置失败，停止解读。
    P1 保持率：retention_dream ≤ retention_wake − 0.15 → 梦回放在真干扰下护旧。
    P3（信息）：真干扰下门是否终于回落（min α_B）；
    P4（信息）：dream_reset 臂（换任务时门重置）在真干扰下是否有代价/收益。

用法：.venv/bin/python experiments/iter6_antitask/run_iter6.py [--seed 0]
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
from src.envs import LatentGrid                        # noqa: E402
from src.substrate import LatentState                  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CFG = dict(grid=6, horizon=30, slip=0.05, p_threat=0.0, guided_frac=0.3,
           cycles=12, episodes=8, wake_batches=8, batch=16, t_win=8,
           d_h=48, lr=3e-3, eval_eps=8, dream_pre=200, buffer_cap=800)
GOAL = (5, 5)
PERM_ID = (0, 1, 2, 3)
PERM_REV = (2, 3, 0, 1)  # 自反：标签上↔下、右↔左


def make_env(seed):
    return LatentGrid(grid=CFG["grid"], slip=CFG["slip"], p_threat_move=CFG["p_threat"],
                      horizon=CFG["horizon"], seed=seed, with_hazard=False,
                      fixed_start=(0, 0), fixed_goal=GOAL)


def rollout(env, perm, rng_env=None, guided=0.8):
    """带动作置换的收集：执行 exec=perm[label]，记录 label（语义随任务而异）。"""
    obs0 = env.reset()
    obs_l, act_l, rew_l = [obs0], [], []
    cur, done = obs0, False
    G = env.G
    while not done:
        if env.rng.random() < CFG["guided_frac"]:
            gdx, gdy = cur[2].item() * G, cur[3].item() * G
            if abs(gdx) >= abs(gdy) and abs(gdx) > 1e-6:
                d = 1 if gdx > 0 else 3
            elif abs(gdy) > 1e-6:
                d = 0 if gdy > 0 else 2
            else:
                d = env.rng.randrange(4)
            label = perm.index(d)  # 反解：使 exec=perm[label]=d
        else:
            label = env.rng.randrange(4)
        cur, r, done, _ = env.step(perm[label])
        obs_l.append(cur)
        act_l.append(label)
        rew_l.append(r)
    T = len(act_l)
    return (torch.stack(obs_l), torch.tensor(act_l, dtype=torch.long),
            torch.tensor(rew_l, dtype=torch.float32),
            torch.zeros(T, dtype=torch.bool))


def build(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=CFG["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    buffer = EpisodeBuffer(capacity=CFG["buffer_cap"])
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=CFG["batch"], t_win=CFG["t_win"]),
                           log=DreamLog(), rng=random.Random(seed))
    evals = {}
    for tag, perm, off in (("A", PERM_ID, 900), ("B", PERM_REV, 950)):
        ev = make_env(seed + off)
        eb = EpisodeBuffer(capacity=99)
        for _ in range(CFG["eval_eps"]):
            eb.add_episode(*rollout(ev, perm))
        evals[tag] = eb
    return env, model, opt, buffer, sched, evals


def probes(model, eval_buf):
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
    for _ in range(CFG["dream_pre"]):
        sched.dream_phase(opt)
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            buffer.add_episode(*rollout(env, PERM_ID))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"], rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
    pA1 = probes(model, evals["A"])
    if arm == "dream_reset":
        model.taiji.reset_gate()
    env_b = make_env(seed + 5)
    trace = []
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            buffer.add_episode(*rollout(env_b, PERM_REV))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"],
                rng=random.Random(seed * 7777 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
        if arm in ("dream", "dream_reset"):
            sched.dream_phase(opt)
        pB = probes(model, evals["B"])
        pA = probes(model, evals["A"])
        trace.append({"cycle": cycle, "alpha_B": round(pB["alpha"], 4),
                      "err_B": round(pB["err"], 4), "err_A": round(pA["err"], 4)})
    return {"arm": arm, "seed": seed, "A_end_phase1": pA1["err"],
            "trace": trace,
            "retention": round(trace[-1]["err_A"] / max(pA1["err"], 1e-6), 4),
            "min_alpha_B": round(min(t["alpha_B"] for t in trace), 4),
            "B_final_err": round(trace[-1]["err_B"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = {}
    for arm in ("wake", "dream", "dream_reset"):
        r = run_arm(arm, args.seed)
        results[arm] = r
        print(f"[{arm} seed{args.seed}] retention={r['retention']} "
              f"min α_B={r['min_alpha_B']} B终误差={r['B_final_err']}")
        print(f"[{arm}] err_A 轨迹：" + " ".join(f"{t['err_A']:.2f}" for t in r["trace"]))
        print(f"[{arm}] α_B 轨迹：" + " ".join(f"{t['alpha_B']:.2f}" for t in r["trace"]))
    w, d, dr = results["wake"], results["dream"], results["dream_reset"]
    g0 = w["retention"] > 1.05
    print(f"\nG0 设置有效门：wake retention={w['retention']} → "
          f"{'✅ 有真遗忘，可解读' if g0 else '❌ 仍无遗忘，设置失败，停止解读'}")
    if g0:
        p1 = d["retention"] <= w["retention"] - 0.15
        print(f"P1 梦护旧：dream {d['retention']} vs wake {w['retention']} → "
              f"{'✅' if p1 else '❌'}（阈 0.15）")
        print(f"P3 门：dream min α_B={d['min_alpha_B']} wake={w['min_alpha_B']}")
        print(f"P4 reset 臂：retention={dr['retention']} B终误差={dr['B_final_err']}")
    p = REPO / "experiments/iter6_antitask"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"iter6_result_seed{args.seed}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"已写 iter6_result_seed{args.seed}.json")


if __name__ == "__main__":
    main()
