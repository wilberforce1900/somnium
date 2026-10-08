#!/usr/bin/env python
"""E3a · 梦期功能分解（A4）实验入口。ROADMAP §1.4。

问题：梦期四组件（想象/巩固/反向学习/排练）在持续学习中的各自贡献——
     谁在保旧任务、谁在护鲁棒性、谁在伤控制（E0 的 MPC 分离之谜）。

设置：三任务序列（同族不同目标几何，含 hazard/稀有事件供排练组件用武）：
     task1 goal(5,5) → task2 goal(0,5) → task3 goal(5,0)，依次学。
     每阶段：wake 收集（30% 引导局）→ wake 更新 → 梦期（alternate 式；
     buffer 全量含旧任务数据 → 回放保旧知识的机制即在此）。

对照配置（6）：full / none / no_imagination / no_consolidation /
              no_reverse_learning / no_rehearsal
指标：
  保持率（主）：retention_k = 1 − (mse_final_k − mse_phase_end_k)/mse_phase_end_k
              （三阶段学完后，对 task1/task2 的保持率均值；>1 = 回放继续改善）
  鲁棒性（辅）：input_noise=0.1 下 eval MSE / 干净 MSE
预注册阈值（2026-10-08 锁定，跑前）：full − none ≥ +10pp 保持率 → A4 正信号；
  单组件关闭后保持率相对 full 下降 ≥5pp → 该组件有独立贡献。
seeds {0,1}，12 run。判定：experiments/summarize.py --exp E3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                          # noqa: E402
from src.dream import (DreamConfig, DreamLog,            # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results" / "registry.md"

CFG = dict(grid=6, horizon=30, slip=0.1, p_threat=0.1, guided_frac=0.3,
           phases=3, cycles=4, episodes=8, wake_batches=8, batch=16, t_win=8,
           d_h=48, lr=3e-3, eval_eps=6, buffer_cap=600)
TASK_GOALS = [(5, 5), (0, 5), (5, 0)]

DREAM_VARIANTS = {
    "full": dict(imagination=True, consolidation=True, reverse_learning=True, rehearsal=True),
    "none": None,
    "no_imagination": dict(imagination=False, consolidation=True, reverse_learning=True, rehearsal=True),
    "no_consolidation": dict(imagination=True, consolidation=False, reverse_learning=True, rehearsal=True),
    "no_reverse_learning": dict(imagination=True, consolidation=True, reverse_learning=False, rehearsal=True),
    "no_rehearsal": dict(imagination=True, consolidation=True, reverse_learning=True, rehearsal=False),
}


def make_env(goal, seed):
    return LatentGrid(grid=CFG["grid"], slip=CFG["slip"], p_threat_move=CFG["p_threat"],
                      horizon=CFG["horizon"], seed=seed, with_hazard=True,
                      fixed_start=(0, 0), fixed_goal=goal)


def eval_task(model, eval_buf, noise=0.0):
    """单任务 eval MSE（锁 seed 4242）。noise>0 时测鲁棒性。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(CFG["batch"], CFG["t_win"], rng=random.Random(4242))
    _, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"],
                                   input_noise=noise)
    model.train()
    return parts["pred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(DREAM_VARIANTS), default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 cycle 快速流水线验证")
    args = ap.parse_args()
    tag = args.tag or "untagged"
    cycles = 2 if args.smoke else CFG["cycles"]
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = WorldModel(d_obs=8, n_actions=4, d_h=CFG["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    buffer = EpisodeBuffer(capacity=CFG["buffer_cap"])
    dcfg = DREAM_VARIANTS[args.variant]
    sched = (DreamScheduler(model, buffer,
                            DreamConfig(batch=CFG["batch"], t_win=CFG["t_win"], **dcfg)
                            if dcfg else None,
                            log=DreamLog(), rng=random.Random(args.seed))
             if dcfg else DreamScheduler(model, buffer, DreamConfig(
                 imagination=False, consolidation=False, reverse_learning=False,
                 rehearsal=False), log=DreamLog(), rng=random.Random(args.seed)))

    eval_bufs = []
    for i, goal in enumerate(TASK_GOALS):
        ev = make_env(goal, args.seed + 900 + i)
        eb = EpisodeBuffer(capacity=99)
        for _ in range(CFG["eval_eps"]):
            eb.add_episode(*random_rollout(ev))
        eval_bufs.append(eb)

    t0 = time.time()
    env_steps = 0
    phase_end_mse = []
    for phase, goal in enumerate(TASK_GOALS):
        env = make_env(goal, args.seed + 10 * phase)
        for cycle in range(cycles):
            for _ in range(CFG["episodes"]):
                roll = guided_rollout if env.rng.random() < CFG["guided_frac"] else random_rollout
                obs, act, rew, rare = roll(env)
                buffer.add_episode(obs, act, rew, rare)
                env_steps += act.shape[0]
            for i in range(CFG["wake_batches"]):
                b = buffer.sample_windows(
                    CFG["batch"], CFG["t_win"],
                    rng=random.Random(args.seed * 1000 + phase * 37 + cycle * 100 + i),
                    only_last=CFG["episodes"])
                sched.wake_update(opt, b)
            if dcfg:  # 梦期（none 变体跳过）
                sched.dream_phase(opt)
        phase_end_mse.append(eval_task(model, eval_bufs[phase]))
        print(f"[{args.variant}] phase {phase+1}/3 done mse={phase_end_mse[-1]:.4f} "
              f"steps={env_steps}", flush=True)

    final_mse = [eval_task(model, eb) for eb in eval_bufs]
    robust = [eval_task(model, eb, noise=0.1) / max(m, 1e-3) for eb, m in zip(eval_bufs, final_mse)]
    ret = []
    for k in range(len(TASK_GOALS) - 1):  # 旧任务 1、2 的保持率
        ret.append(1.0 - (final_mse[k] - phase_end_mse[k]) / max(phase_end_mse[k], 1e-3))
    rec = {"variant": args.variant, "seed": args.seed, "env_steps": env_steps,
           "phase_end_mse": [round(x, 6) for x in phase_end_mse],
           "final_mse": [round(x, 6) for x in final_mse],
           "retention12": [round(x, 4) for x in ret],
           "retention_mean": round(sum(ret) / len(ret), 4),
           "robust_ratio": [round(x, 3) for x in robust]}
    out = REPO / "experiments/e3_dream_ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"details_e3_{args.variant}_{tag}_{args.seed}.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(
            f"| 2026-10-08 | E3 | dream_{args.variant} | {args.seed} | {env_steps} "
            f"| {final_mse[0]:.4f} | ret12={rec['retention_mean']} "
            f"rob={robust[0]:.2f} | 待判定 | "
            f"{'smoke' if args.smoke else 'full'}/{tag} |\n"
        )
    print(f"完成：{time.time()-t0:.1f}s retention_mean={rec['retention_mean']} "
          f"robust={robust}")


if __name__ == "__main__":
    main()
