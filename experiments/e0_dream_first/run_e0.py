#!/usr/bin/env python
"""E0 · 先梦后醒（消融 A3）实验入口。ROADMAP §1.1。

日程：
    wake_only    纯醒基线（真实数据消耗基准）
    dream_first  梦期先行：内生想象预训练 k 批 → 之后与 wake_only 同日程
    alternate    梦醒交替：每 cycle 醒更新后追加一轮梦期

各日程的醒期真实数据消耗严格一致（同一收集流、同一 wake 批数）；
差异只来自梦期额外离线计算——公平性注记见 src/dream.py。

指标：固定 eval 集（独立种子）上的潜态预测 MSE（主）；MPC 回报探头（辅）。
输出：results/registry.md 追加一行 + 本目录 details_<tag>.jsonl + dream log。

smoke（默认小规模，验证流水线）：
    .venv/bin/python experiments/e0_dream_first/run_e0.py --tag smoke
正式 E0（≤12 run，预注册阈值见 ROADMAP §1.1，跑前锁定）：
    加 --full 后按全量配置跑。
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

from src.core import WorldModel            # noqa: E402
from src.dream import (DreamConfig, DreamLog,   # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs.latent_grid import LatentGrid, guided_rollout, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results" / "registry.md"

SMALL = dict(cycles=4, episodes_per_cycle=6, grid=5, horizon=20,
             d_h=32, batch=16, t_win=8, wake_batches=8,
             dream_pre_batches=30, lr=3e-3, eval_episodes=8,
             mpc_episodes=3, mpc_horizon=5, mpc_k=48, buffer_cap=400,
             guided_frac=0.3)
FULL = dict(cycles=20, episodes_per_cycle=10, grid=8, horizon=50,
            d_h=64, batch=32, t_win=8, wake_batches=10,
            dream_pre_batches=200, lr=1e-3, eval_episodes=20,
            mpc_episodes=5, mpc_horizon=6, mpc_k=96, buffer_cap=2000,
            guided_frac=0.3)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["wake_only", "dream_first", "alternate"],
                    default="wake_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="登记表备注标签（如 smoke / run1）")
    ap.add_argument("--full", action="store_true", help="正式全量配置（默认 smoke）")
    return ap.parse_args()


def eval_mse(model, eval_buf, cfg, rng):
    """固定 eval 集上的潜态预测 MSE。

    P1-4：eval 前锁 torch 全局 seed——loss_on_batch 内部的 spawn（随机初始
    潜态）与窗口采样全部确定化，跨日程/seed 可比。
    """
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(cfg["batch"], cfg["t_win"], rng=rng)
    _, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
    model.train()
    return parts["pred"]


def mpc_return(model, cfg, seed):
    """MPC 回报探头：独立环境上跑几局（P1-4：锁 seed 4243）。"""
    torch.manual_seed(4243)
    env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                     horizon=cfg["horizon"], seed=seed)
    model.eval()
    returns = []
    with torch.no_grad():
        for _ in range(cfg["mpc_episodes"]):
            obs = env.reset()
            h = model.substrate.spawn(1).h[0]
            total, done = 0.0, False
            while not done:
                a, _ = model.plan(obs, h, horizon=cfg["mpc_horizon"], k=cfg["mpc_k"])
                obs, r, done, _ = env.step(a)
                total += r
                hs, _ = model.rollout_step(h.unsqueeze(0), obs.unsqueeze(0),
                                           torch.tensor([a], dtype=torch.long))
                h = hs[0]
            returns.append(total)
    model.train()
    return sum(returns) / len(returns)


def main():
    args = parse_args()
    cfg = dict(FULL if args.full else SMALL)
    tag = args.tag or "untagged"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                     horizon=cfg["horizon"], seed=args.seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=cfg["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
    dream_log = DreamLog()
    sched = DreamScheduler(model, buffer, DreamConfig(batch=cfg["batch"], t_win=cfg["t_win"]),
                           log=dream_log, rng=random.Random(args.seed))

    # 固定 eval 集：独立种子，跨日程可比
    ev_env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                        horizon=cfg["horizon"], seed=args.seed + 999)
    eval_buf = EpisodeBuffer(capacity=9999)
    for _ in range(cfg["eval_episodes"]):
        eval_buf.add_episode(*random_rollout(ev_env))

    t0 = time.time()
    env_steps = 0
    details = []

    # 梦期先行：buffer 为空，自动只剩内生想象（E0-b 的"先梦"阶段）
    if args.schedule == "dream_first":
        for _ in range(cfg["dream_pre_batches"]):
            sched.dream_phase(opt)

    for cycle in range(cfg["cycles"]):
        # 醒期数据收集（各日程一致；P0-1 课程：混入 30% 贪心引导局）
        for _ in range(cfg["episodes_per_cycle"]):
            roll = guided_rollout if env.rng.random() < cfg["guided_frac"] else random_rollout
            obs, act, rew, rare = roll(env)
            buffer.add_episode(obs, act, rew, rare)
            env_steps += act.shape[0]
        # 醒期更新：只用本 cycle 新鲜数据
        for i in range(cfg["wake_batches"]):
            b = buffer.sample_windows(cfg["batch"], cfg["t_win"],
                                      rng=random.Random(args.seed * 1000 + cycle * 100 + i),
                                      only_last=cfg["episodes_per_cycle"])
            sched.wake_update(opt, b)
        # 梦醒交替：追加大梦期（此时 buffer 已有数据，四组件全跑）
        if args.schedule == "alternate":
            sched.dream_phase(opt)
        # 评估
        mse = eval_mse(model, eval_buf, cfg, random.Random(4242))
        ret = mpc_return(model, cfg, seed=args.seed + 500)
        details.append({"cycle": cycle, "env_steps": env_steps,
                        "pred_mse": round(mse, 6), "mpc_return": round(ret, 4),
                        "dream_log_len": len(dream_log)})
        print(f"[{args.schedule}] cycle {cycle + 1}/{cfg['cycles']} "
              f"steps={env_steps} pred_mse={mse:.4f} mpc={ret:.3f}")

    dream_log.save(REPO / "experiments/e0_dream_first/details_dream.jsonl")

    out = REPO / "experiments/e0_dream_first"
    with open(out / f"details_{args.schedule}_{tag}_{args.seed}.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")

    final = details[-1]
    scale = "full" if args.full else "smoke"
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY.exists():
        REGISTRY.write_text(
            "# Run 登记表\n\n一行一 run，负结果照记（ROADMAP §0）。\n\n"
            "公平性注记：梦日程（dream_first/alternate）含额外离线梯度步；"
            "E0 比较基准=真实环境交互步数（PRINCIPLES §11 A3）。\n\n"
            "| 日期 | 实验 | 日程 | seed | 真实交互步数 | eval 潜态预测 MSE | MPC 回报 | 结论 | 备注 |\n"
            "|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(
            f"| 2026-10-08 | E0 | {args.schedule}+g{cfg['guided_frac']} | {args.seed} "
            f"| {final['env_steps']} | {final['pred_mse']} | {final['mpc_return']} "
            f"| 流水线验证 | {scale}/{tag} |\n"
        )
    print(f"完成：{time.time() - t0:.1f}s，已登记 results/registry.md")


if __name__ == "__main__":
    main()
