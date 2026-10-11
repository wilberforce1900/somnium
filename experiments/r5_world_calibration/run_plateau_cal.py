#!/usr/bin/env python
"""R5-cal v2 · plateau 对齐版世界校准（迭代 #7 Step 2 重设计）。

教训修正（v1 缺陷）：v1 比较同预算终值——1024 档欠训练被误读为饱和。
v2 协议：每个 (k, d_h) 跑到 **plateau**（连续 PLATEAU_PAT 个 cycle 的 MSE
相对变化 < PLATEAU_TOL）或 MAX_CYCLES 上限；比较的是**各自 plateau MSE**
（plateau 段均值）。这才回答"世界是否饱和"。

通过门（预注册，ITER7 §2）：存在 k* 使 d_h 256→1024 的 plateau MSE
改善 ≥10% → R5b 放行。

用法：python run_plateau_cal.py --complexity K --d-h H [--device cuda]
输出：OUT/plateau_k{K}_dh{H}_s{SEED}.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                 # noqa: E402
from src.dream import EpisodeBuffer             # noqa: E402
from src.envs import guided_rollout, random_rollout  # noqa: E402
from src.envs.rich_world import RichWorld       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/r5_world_calibration"
OUT.mkdir(parents=True, exist_ok=True)

PLATEAU_PAT = 10     # 连续 N cycle
PLATEAU_TOL = 0.02   # 相对变化 < 2% 记 plateau


def make_env(k, seed):
    return RichWorld(complexity=k, seed=seed)


def train_to_plateau(k, d_h, seed, device, max_cycles=120, t_win=8, hidden=32, lr=1e-3):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(k, seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions,
                       d_h=d_h, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev = make_env(k, seed + 900)
    eval_buf = EpisodeBuffer(capacity=99)
    for _ in range(10):
        eval_buf.add_episode(*random_rollout(ev))

    def eval_mse():
        torch.manual_seed(4242)
        model.eval()
        b = eval_buf.sample_windows(32, t_win, rng=random.Random(4242))
        h = model.substrate.spawn(b["obs"].shape[0]).h
        err = 0.0
        for t in range(t_win):
            h, _ = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            tgt = model.embed(b["obs_next"][:, t])
            err += float(((h - tgt) ** 2).mean())
        model.train()
        return err / t_win

    total_steps = 0
    curve = []
    plateau_at = None
    plateau_mse = None
    for cycle in range(max_cycles):
        for _ in range(8):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            buffer.add_episode(*roll(env))
            total_steps += buffer.episodes[-1]["act"].shape[0]
        for i in range(10):
            b = buffer.sample_windows(32, t_win,
                                      rng=random.Random(seed * 1000 + cycle * 100 + i),
                                      only_last=8)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        mse = eval_mse()
        curve.append(round(mse, 5))
        if cycle % 10 == 0 or cycle == max_cycles - 1:
            print(f"[k={k} d_h={d_h} s{seed}] cyc{cycle+1}/{max_cycles} "
                  f"mse={mse:.4f} steps={total_steps}", flush=True)
        # plateau 检测（从第 PAT 个 cycle 起）
        if cycle >= PLATEAU_PAT:
            window = curve[-PLATEAU_PAT:]
            ref = window[0]
            if ref > 0 and max(abs(m - ref) / ref for m in window) < PLATEAU_TOL:
                plateau_at = cycle + 1
                plateau_mse = sum(window) / len(window)
                break
    if plateau_mse is None:  # 未达 plateau：取最后 PAT 段均值
        plateau_mse = sum(curve[-PLATEAU_PAT:]) / PLATEAU_PAT
    return {"k": k, "d_h": d_h, "seed": seed, "t_win": t_win, "hidden": hidden, "lr": lr,
            "max_cycles": max_cycles, "stopped_at": len(curve),
            "plateau_at": plateau_at,
            "plateau_mse": round(plateau_mse, 5),
            "first_mse": curve[0], "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", type=int, required=True)
    ap.add_argument("--d-h", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-cycles", type=int, default=120)
    ap.add_argument("--t-win", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32,
                    help="encoder hidden（秩瓶颈假设验证：1024 档配 128）")
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    rec = train_to_plateau(args.complexity, args.d_h, args.seed, dev,
                           max_cycles=args.max_cycles, t_win=args.t_win,
                           hidden=args.hidden, lr=args.lr)
    tag = f"plateau_k{args.complexity}_dh{args.d_h}_h{args.hidden}_lr{args.lr:g}_s{args.seed}"
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print("RECORD:", json.dumps({k2: v for k2, v in rec.items() if k2 != "curve"}))


if __name__ == "__main__":
    main()
