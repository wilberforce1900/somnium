#!/usr/bin/env python
"""R5-cal · 世界校准（迭代 #7 Step 2）。ITER7-WORLD-FIRST.md §2。

问题：ENV-D 富世界各复杂度档上，wake 精度是否重新随容量改善？
通过门：存在 k* 使 256→1024 的 wake 终值 MSE 改善 ≥10% → 该世界值得 R5b。

协议：k ∈ {0,1,2,3,4} × d_h ∈ {64,256,1024} × 1 seed（15 run）+ k* 复核 1 seed。
指标：eval 潜态预测 MSE（wake_only 训练，纯醒无梦——校准不掺梦变量）。
用法：python run_cal.py --complexity K --d-h H [--device cuda]
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


def make_env(k, seed):
    return RichWorld(complexity=k, seed=seed)


def train_eval(k, d_h, seed, device, cycles=30):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(k, seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions,
                       d_h=d_h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = EpisodeBuffer(capacity=2000)
    # 固定 eval 集（独立种子）
    ev = make_env(k, seed + 900)
    eval_buf = EpisodeBuffer(capacity=99)
    for _ in range(10):
        eb_obs, eb_act, eb_rew, eb_rare = random_rollout(ev)
        eval_buf.add_episode(eb_obs, eb_act, eb_rew, eb_rare)

    def eval_mse():
        torch.manual_seed(4242)
        model.eval()
        b = eval_buf.sample_windows(32, 8, rng=random.Random(4242))
        h = model.substrate.spawn(b["obs"].shape[0]).h
        err = 0.0
        for t in range(8):
            h, _ = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            tgt = model.embed(b["obs_next"][:, t])
            err += float(((h - tgt) ** 2).mean())
        model.train()
        return err / 8

    total_steps = 0
    first = None
    for cycle in range(cycles):
        for _ in range(8):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            buffer.add_episode(*roll(env))
            total_steps += buffer.episodes[-1]["act"].shape[0]
        for i in range(10):
            b = buffer.sample_windows(32, 8,
                                      rng=random.Random(seed * 1000 + cycle * 100 + i),
                                      only_last=8)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        mse = eval_mse()
        if first is None:
            first = mse
        if cycle % 5 == 0 or cycle == cycles - 1:
            print(f"[k={k} d_h={d_h} s{seed}] cyc{cycle+1}/30 mse={mse:.4f} "
                  f"steps={total_steps}", flush=True)
    return {"k": k, "d_h": d_h, "seed": seed, "first_mse": round(first, 5),
            "final_mse": round(mse, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", type=int, required=True)
    ap.add_argument("--d-h", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cycles", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    rec = train_eval(args.complexity, args.d_h, args.seed, dev,
                     cycles=args.cycles)
    with open(OUT / f"cal_k{args.complexity}_dh{args.d_h}_s{args.seed}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print("RECORD:", json.dumps(rec))


if __name__ == "__main__":
    main()
