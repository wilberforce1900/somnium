#!/usr/bin/env python
"""E2 · 六十四卦态势码本消融（A1，用户点名第一优先）实验入口。ROADMAP §1.3。

问题：64 格离散态势码本 + 转移图上的慢规划，是否优于自由潜空间 rollout 规划？

变量：
    --k         码本容量 {0=无码本基线, 32, 64, 128}
    --planner   graph（卦图价值迭代 + 单步想象落卦）
                / free（自由 random-shooting，k×horizon rollouts）
                / free_match（等 rollout 数对照：n_actions 次 horizon=1）
任务：grid_long——8×8 固定起终点、稀疏奖励、horizon 60（长程 ≥10 步）。
    成功率 = eval 局内到达 goal 的比例（主指标，预注册 +5pp 阈值）。

⚠ 依赖注记：正式 E2 应使用 E0 最优训练日程；E0 正式跑未完成前以 wake_only
训练（偏离记录于登记表）。卦码路径 dump 供人工可解释性盲评（预注册 §1.3）。

smoke（默认小规模）：
    .venv/bin/python experiments/e2_hexagram_codebook/run_e2.py --k 64 --planner graph --tag smoke
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

from src.codebook import HexagramCodebook       # noqa: E402
from src.core import WorldModel                 # noqa: E402
from src.dream import EpisodeBuffer             # noqa: E402
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402
from src.planner import plan_free, plan_free_match, plan_graph, plan_shaped  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results" / "registry.md"

SMALL = dict(cycles=5, episodes=8, d_h=32, wake_batches=8, t_win=8,
             eval_eps=20, lr=3e-3, grid=8, horizon=60, buffer_cap=500,
             free_k=48, free_horizon=6, gamma=0.9, vi_depth=6, guided_frac=0.3,
             lam_graph=1.0)
FULL = dict(cycles=20, episodes=12, d_h=64, wake_batches=12, t_win=8,
            eval_eps=16, lr=1e-3, grid=8, horizon=60, buffer_cap=2000,
            free_k=128, free_horizon=8, gamma=0.9, vi_depth=10, guided_frac=0.3,
            lam_graph=1.0)


def make_env(cfg, seed):
    return LatentGrid(grid=cfg["grid"], slip=0.05, p_threat_move=0.0,
                      horizon=cfg["horizon"], seed=seed, with_hazard=False,
                      fixed_start=(0, 0), fixed_goal=(cfg["grid"] - 1, cfg["grid"] - 1))


def run_planned_episodes(model, planner, codebook, cfg, seed, trace_out=None):
    """用指定规划器跑 eval 局。返回 (成功率, 平均回报, 步数均值)。

    P1-4：锁 eval 随机 seed；P0-2：初始潜态回到 spawn（出生于混沌），
    之后每步用真实转移更新 h——t≥1 起即为训练分布内的"在环"潜态，
    避免编码器直出的接地流形 mismatch 污染落卦。
    """
    torch.manual_seed(4242)
    env = make_env(cfg, seed)
    succ, rets, steps = [], [], []
    hex_trace = None
    model.eval()
    with torch.no_grad():
        for ep in range(cfg["eval_eps"]):
            obs = env.reset()
            h = model.substrate.spawn(1).h[0]
            total, done, t = 0.0, False, 0
            trace = []
            while not done:
                if planner == "graph":
                    a, _ = plan_graph(model, codebook, obs, h,
                                      gamma=cfg["gamma"], depth=cfg["vi_depth"])
                elif planner == "shaped":
                    a, _ = plan_shaped(model, codebook, obs, h,
                                       horizon=cfg["free_horizon"], k=cfg["free_k"],
                                       gamma=cfg["gamma"], depth=cfg["vi_depth"],
                                       lam=cfg["lam_graph"])
                elif planner == "free":
                    a = plan_free(model, obs, h,
                                  horizon=cfg["free_horizon"], k=cfg["free_k"])
                else:
                    a = plan_free_match(model, obs, h)
                obs, r, done, _ = env.step(a)
                total += r
                t += 1
                h, _ = model.rollout_step(h.view(1, -1), obs.view(1, -1),
                                          torch.tensor([a], dtype=torch.long))
                h = h[0]
                if codebook is not None:
                    trace.append(codebook.to_hexagram(int(codebook.assign(h.view(1, -1))[0])))
            succ.append(r == 1.0)
            rets.append(total)
            steps.append(t)
            if ep == 0 and trace_out is not None:
                hex_trace = trace
    model.train()
    if trace_out is not None and hex_trace is not None:
        trace_out.append(hex_trace)
    return sum(succ) / len(succ), sum(rets) / len(rets), sum(steps) / len(steps)


def random_success(cfg, seed):
    """随机策略参考线（同 eval 预算）。"""
    env = make_env(cfg, seed)
    succ = 0
    for _ in range(cfg["eval_eps"]):
        obs, act, rew, _ = random_rollout(env)
        if rew[-1].item() == 1.0 or (rew == 1.0).any().item():
            succ += 1
    return succ / cfg["eval_eps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64, help="码本容量，0=无码本基线")
    ap.add_argument("--planner", choices=["graph", "shaped", "free", "free_match"],
                    default="shaped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = FULL if args.full else SMALL
    tag = args.tag or "untagged"
    if args.planner == "graph" and args.k == 0:
        raise SystemExit("graph 规划器需要 k>0（无码本基线请用 free/free_match）")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env = make_env(cfg, args.seed)
    codebook = HexagramCodebook(cfg["d_h"], k=args.k) if args.k > 0 else None
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=cfg["d_h"],
                       codebook=codebook)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])

    t0 = time.time()
    env_steps = 0
    details = []
    traces = []
    rand_rate = random_success(cfg, args.seed + 777)
    for cycle in range(cfg["cycles"]):
        for _ in range(cfg["episodes"]):
            # P0-1 课程：混入 30% 贪心引导局（奖励头/卦德图需要 +1 信号）
            roll = guided_rollout if env.rng.random() < cfg["guided_frac"] else random_rollout
            obs, act, rew, rare = roll(env)
            buffer.add_episode(obs, act, rew, rare)
            env_steps += act.shape[0]
        for i in range(cfg["wake_batches"]):
            b = buffer.sample_windows(
                16, cfg["t_win"],
                rng=random.Random(args.seed * 1000 + cycle * 100 + i),
                only_last=cfg["episodes"])
            loss, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        succ, ret, steps = run_planned_episodes(model, args.planner, codebook, cfg,
                                                args.seed + 999,
                                                trace_out=traces if cycle == cfg["cycles"] - 1 else None)
        rec = {"cycle": cycle, "env_steps": env_steps, "success": round(succ, 4),
               "return": round(ret, 4), "mean_steps": round(steps, 1),
               "pred_mse": round(parts["pred"], 6)}
        if codebook is not None:
            rec["visited_codes"] = int((codebook.visits > 0).sum())
            rec["cb_loss"] = round(parts.get("cb", -1), 6)
        details.append(rec)
        print(f"[k={args.k}/{args.planner}] cycle {cycle+1}/{cfg['cycles']} "
              f"steps={env_steps} succ={succ:.2f} ret={ret:.2f} rand={rand_rate:.2f} "
              f"visited={rec.get('visited_codes', 0)}")

    out = REPO / "experiments/e2_hexagram_codebook"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"k{args.k}_{args.planner}_{tag}_{args.seed}"
    with open(out / f"details_{suffix}.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")
        if traces:
            f.write(json.dumps({"hex_trace_last_eval": traces[-1][:60]}) + "\n")

    final = details[-1]
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(
            f"| 2026-10-08 | E2 | grid_long/k{args.k}/{args.planner}+g{cfg['guided_frac']} "
            f"| {args.seed} | {final['env_steps']} | {final['pred_mse']} "
            f"| succ={final['success']} ret={final['return']} rand={rand_rate:.2f} "
            f"| 流水线验证 | {'full' if args.full else 'smoke'}/{tag} |\n"
        )
    print(f"完成：{time.time() - t0:.1f}s，已登记 results/registry.md")


if __name__ == "__main__":
    main()
