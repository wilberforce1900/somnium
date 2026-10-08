#!/usr/bin/env python
"""E1 · 阴阳门控消融（A2）实验入口。ROADMAP §1.2。

变量隔离：**纯醒训练（无梦期）**——唯一变量 = gate_mode 与 extra_relax。
    gate_mode:  learned（可学习门）/ fixed_yang（α≡1 纯阳）/ fixed_yin（α≡0 纯阴）
    extra_relax: 每步追加的阴弛豫深度（0/1/2/4，深度-性能曲线辅助项）
任务：grid（ENV-A）/ maze（ENV-C）。

⚠ 预注册偏离记录（2026-10-08）：原定任务集 ENV-B+ENV-C；ENV-B（ARC）因外网
不可达暂缺，A2 正式判定任务集调整为 ENV-A+ENV-C，见 ROADMAP §1.2。

指标（登记表辅助指标列）：
    pred_mse   固定 eval 集潜态预测 MSE（主）
    acc        maze=墙位预测准确率 / grid=目标增量容差(0.1)命中率
    alpha      学得门控的阳配比均值（fixed 模式恒 1/0）
    α-难度相关 每窗 α 均值 vs 窗内目标距离的 Pearson 相关
    （A2 调度信号预期：难题多阴 → 相关系数为负）
注意：fixed_yin 的动力学不含观测通道（阴主内），预期接近盲猜基线——
它是机制对照组，不是竞争基线；有信息量的对比是 learned vs fixed_yang。

smoke（默认小规模）：
    .venv/bin/python experiments/e1_yinyang_gate/run_e1.py --task maze --tag smoke
正式（≤8 run，阈值预注册见 ROADMAP §1.2，跑前锁定）：加 --full。
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

from src.core import WorldModel                       # noqa: E402
from src.dream import EpisodeBuffer                   # noqa: E402
from src.envs import LatentGrid, Maze, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results" / "registry.md"

CFG = {
    "grid": dict(
        small=dict(cycles=4, episodes=6, d_h=32, wake_batches=8, t_win=8,
                   eval_eps=8, lr=3e-3, grid=5, horizon=20, buffer_cap=400),
        full=dict(cycles=20, episodes=10, d_h=64, wake_batches=10, t_win=8,
                  eval_eps=20, lr=1e-3, grid=8, horizon=50, buffer_cap=2000),
    ),
    "maze": dict(
        small=dict(cycles=6, episodes=6, d_h=32, wake_batches=10, t_win=8,
                   eval_eps=8, lr=3e-3, size=5, horizon=40, buffer_cap=600),
        full=dict(cycles=25, episodes=10, d_h=64, wake_batches=12, t_win=8,
                  eval_eps=20, lr=1e-3, size=8, horizon=128, buffer_cap=2500),
    ),
}


def pearson(x: torch.Tensor, y: torch.Tensor):
    x = x - x.mean()
    y = y - y.mean()
    d = float(x.norm() * y.norm())
    return float((x * y).sum() / d) if d > 0 else 0.0


def eval_task(model, batch, task):
    """返回 dict(pred_mse, acc, alpha_mean, alpha_diff_corr)。fixed 模式 corr 无义。

    P1-4：调用方在 eval 前锁 torch seed（spawn 的随机初始潜态确定化）。
    """
    torch.manual_seed(4242)
    model.eval()
    obs, act, obs_next = batch["obs"], batch["act"], batch["obs_next"]
    B, T, _ = obs.shape
    with torch.no_grad():
        h = model.substrate.spawn(B).h
        mse_sum = acc_sum = 0.0
        alphas = []
        for t in range(T):
            h, a = model.rollout_step(h, obs[:, t], act[:, t])
            target = model.embed(obs_next[:, t])
            mse_sum += torch.mean((h - target) ** 2)
            o_hat = model.decode_obs(h)
            nxt = obs_next[:, t]
            if task == "maze":  # 墙位 4-bit 准确率
                acc_sum += ((o_hat[:, 4:8] > 0.5) == (nxt[:, 4:8] > 0.5)).float().mean()
            else:               # 目标增量容差命中率
                acc_sum += ((o_hat[:, 2:4] - nxt[:, 2:4]).abs() < 0.1).all(-1).float().mean()
            alphas.append(a.squeeze(-1))
        alphas = torch.stack(alphas, dim=1)             # (B, T)
        alpha_win = alphas.mean(dim=1)                  # 每窗 α 均值
        diff_win = (obs[:, :, 2].abs() + obs[:, :, 3].abs()).mean(dim=1)  # 窗内目标距离
        corr = pearson(alpha_win, diff_win)
    model.train()
    return {
        "pred_mse": float(mse_sum / T),
        "acc": float(acc_sum / T),
        "alpha_mean": float(alphas.mean()),
        "alpha_diff_corr": corr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["grid", "maze"], default="maze")
    ap.add_argument("--gate_mode", choices=["learned", "fixed_yang", "fixed_yin"],
                    default="learned")
    ap.add_argument("--extra_relax", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = CFG[args.task]["full" if args.full else "small"]
    tag = args.tag or "untagged"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.task == "maze":
        env = Maze(size=cfg["size"], horizon=cfg["horizon"], seed=args.seed)
        ev = Maze(size=cfg["size"], horizon=cfg["horizon"], seed=args.seed + 999)
    else:
        env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                         horizon=cfg["horizon"], seed=args.seed)
        ev = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                        horizon=cfg["horizon"], seed=args.seed + 999)

    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=cfg["d_h"],
                       gate_mode=args.gate_mode, extra_relax=args.extra_relax)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
    eval_buf = EpisodeBuffer(capacity=9999)
    for _ in range(cfg["eval_eps"]):
        eval_buf.add_episode(*random_rollout(ev))

    t0 = time.time()
    env_steps = 0
    details = []
    for cycle in range(cfg["cycles"]):
        for _ in range(cfg["episodes"]):
            obs, act, rew, rare = random_rollout(env)
            buffer.add_episode(obs, act, rew, rare)
            env_steps += act.shape[0]
        for i in range(cfg["wake_batches"]):
            b = buffer.sample_windows(
                16, cfg["t_win"],
                rng=random.Random(args.seed * 1000 + cycle * 100 + i),
                only_last=cfg["episodes"])
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        b = eval_buf.sample_windows(64, cfg["t_win"], rng=random.Random(4242))
        m = eval_task(model, b, args.task)
        details.append({"cycle": cycle, "env_steps": env_steps, **{
            k: round(v, 6) for k, v in m.items()}})
        print(f"[{args.task}/{args.gate_mode}+r{args.extra_relax}] cycle {cycle+1}/{cfg['cycles']} "
              f"steps={env_steps} mse={m['pred_mse']:.4f} acc={m['acc']:.3f} "
              f"α={m['alpha_mean']:.2f} corr={m['alpha_diff_corr']:+.2f}")

    out = REPO / "experiments/e1_yinyang_gate"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"details_{args.task}_{args.gate_mode}_r{args.extra_relax}_{tag}_{args.seed}.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")

    final = details[-1]
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(
            f"| 2026-10-08 | E1 | {args.task}/{args.gate_mode}+r{args.extra_relax} "
            f"| {args.seed} | {final['env_steps']} | {final['pred_mse']} "
            f"| acc={final['acc']} α={final['alpha_mean']} corr={final['alpha_diff_corr']} "
            f"| 流水线验证 | {'full' if args.full else 'smoke'}/{tag} |\n"
        )
    print(f"完成：{time.time() - t0:.1f}s，已登记 results/registry.md")


if __name__ == "__main__":
    main()
