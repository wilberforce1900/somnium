#!/usr/bin/env python
"""ITER15 · 外部海马体：持续经验流上的梦巩固（三臂 W/WR/WS，预登记见 ITER15-PLAN.md）。

问题：agent 不能保留原始历史（块间清缓冲）时，"睡掉今日"的梦巩固（当日真实情节
+ 自身想象，DreamConfig 全开）能否部分替代真实回放，保住旧 regime 的功能？

流设计（制度性 regime 交替，保证有东西可忘）：
    C1@ρ0=0.1 → C2@ρ1=0.45 → C3@ρ0 → C4@ρ1，每块 cycles_chunk 个 cycle，
    醒期协议与 E0 FULL 逐字同（10 局/cycle，guided 0.3，10 wake 批，recent 采样）。

臂（同 seed 配对，环境流逐位相同，唯一差异=块间处理）：
    W   清缓冲，无额外批（持续学习下界）
    WR  保留全历史，+100 批真实旧窗口回放（存储上界）
    WS  先对当前缓冲做 100 批全开关梦巩固，再清缓冲（O(1) 海马体主张）

主判据（跑前锁定，ITER15-PLAN §4）：
    R_arm = (MSE_W_end − MSE_arm_end) / MSE_W_end，ρ0 eval 集（seed+999）@C4 末
    G1: R_WS ≥ 0.15 于 ≥2/3 seeds；G2: R_WS ≥ 0.5 × R_WR 于 ≥2/3 seeds；
    Kill: R_WS ≤ 0.05 或 ≥2 seeds 上 WS ≤ W → 负结果登记。
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

from src.core import WorldModel  # noqa: E402
from src.dream import DreamConfig, DreamScheduler, EpisodeBuffer  # noqa: E402
from src.envs.latent_grid import LatentGrid, guided_rollout, random_rollout  # noqa: E402
from src.substrate import default_device  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "hippocampus"
RHO0, RHO1 = 0.1, 0.45
CHUNK_RHOS = [RHO0, RHO1, RHO0, RHO1]

FULL = dict(cycles_chunk=150, episodes_per_cycle=10, grid=8, horizon=50,
            d_h=64, batch=32, t_win=8, wake_batches=10, lr=1e-3,
            eval_episodes=20, buffer_cap=2000, guided_frac=0.3,
            boundary_batches=100, replay_batch=32)
SMOKE = dict(FULL, cycles_chunk=2, episodes_per_cycle=3, d_h=32, batch=8,
             wake_batches=2, eval_episodes=4, buffer_cap=200, boundary_batches=3,
             replay_batch=8, horizon=20)


def eval_rho0(model, cfg, seed):
    """固定 ρ0 eval 集（两组独立种子）：pred MSE（锁 4242）+ 解码 R²（对拷贝）。"""
    out = {}
    for tag, eseed in (("e999", seed + 999), ("e900", seed + 900)):
        ev = LatentGrid(grid=cfg["grid"], slip=RHO0, p_threat_move=0.02,
                        horizon=cfg["horizon"], seed=eseed)
        buf = EpisodeBuffer(capacity=9999)
        for _ in range(cfg["eval_episodes"]):
            buf.add_episode(*random_rollout(ev))
        torch.manual_seed(4242)
        model.eval()
        b = buf.sample_windows(cfg["batch"], cfg["t_win"], rng=random.Random(4242))
        with torch.no_grad():
            _, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            h_T, _ = model.forward(b["obs"], b["act"])
            dec = model.decode_obs(h_T).cpu()  # 模型可能在 cuda：解码回传 cpu 与观测对齐
            tgt = b["obs"][:, -1, :]
            prev = b["obs"][:, -2, :]
            ss_tot = ((tgt - tgt.mean(dim=0, keepdim=True)) ** 2).sum().item()
            r2 = 1.0 - ((dec - tgt) ** 2).sum().item() / max(ss_tot, 1e-12)
            copy_mse = ((prev - tgt) ** 2).mean().item()
        model.train()
        out[tag] = {"pred_mse": round(parts["pred"], 6),
                    "decode_r2": round(r2, 4), "copy_mse": round(copy_mse, 6)}
    return out


def run_arm(arm: str, seed: int, cfg: dict, dev) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    model = WorldModel(d_obs=8, n_actions=4, d_h=cfg["d_h"], norm_kind="none").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
    history, t0 = [], time.time()

    for chunk, rho in enumerate(CHUNK_RHOS):
        env = LatentGrid(grid=cfg["grid"], slip=rho, p_threat_move=0.02,
                         horizon=cfg["horizon"], seed=seed * 100 + chunk)
        for cycle in range(cfg["cycles_chunk"]):
            for _ in range(cfg["episodes_per_cycle"]):
                roll = guided_rollout if env.rng.random() < cfg["guided_frac"] else random_rollout
                obs, act, rew, rare = roll(env)
                buffer.add_episode(obs, act, rew, rare)
            for i in range(cfg["wake_batches"]):
                b_rng = random.Random(seed * 100000 + chunk * 10000 + cycle * 100 + i)
                b = buffer.sample_windows(cfg["batch"], cfg["t_win"], rng=b_rng,
                                          only_last=cfg["episodes_per_cycle"])
                sched = DreamScheduler(model, buffer,
                                       DreamConfig(batch=cfg["batch"], t_win=cfg["t_win"]),
                                       rng=b_rng)
                sched.wake_update(opt, b)
        # ── 块间处理（三臂唯一差异） ──
        extra = None
        if arm == "WR":
            for i in range(cfg["boundary_batches"]):
                b_rng = random.Random(seed * 100000 + chunk * 10000 + 9000 + i)
                b = buffer.sample_windows(cfg["replay_batch"], cfg["t_win"], rng=b_rng)
                sched = DreamScheduler(model, buffer,
                                       DreamConfig(batch=cfg["replay_batch"], t_win=cfg["t_win"]),
                                       rng=b_rng)
                sched.wake_update(opt, b)
            extra = f"replay{cfg['boundary_batches']}"
        elif arm == "WS":
            m = time.time()
            sched = DreamScheduler(model, buffer,
                                   DreamConfig(batch=cfg["batch"], t_win=cfg["t_win"]),
                                   rng=random.Random(seed * 77 + chunk))
            for _ in range(cfg["boundary_batches"]):
                sched.dream_phase(opt)
            buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])  # 睡完即忘（O(1)）
            extra = f"dream{cfg['boundary_batches']}@{time.time() - m:.0f}s"
        else:  # W
            buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
        ev = eval_rho0(model, cfg, seed)
        history.append({"chunk": chunk, "rho": rho, "eval": ev, "boundary": extra})
        print(f"[{arm} s{seed}] chunk{chunk}(ρ={rho}) e999={ev['e999']['pred_mse']:.4f} "
              f"r2={ev['e999']['decode_r2']:.3f} boundary={extra}", flush=True)

    return {"arm": arm, "seed": seed, "history": history,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["W", "WR", "WS"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = dict(SMOKE if args.smoke else FULL)
    dev = default_device() if args.device == "auto" else torch.device(args.device)

    res = run_arm(args.arm, args.seed, cfg, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "full"
    with open(OUT / f"details_{args.arm.lower()}_s{args.seed}_{suffix}.jsonl", "w") as f:
        for h in res["history"]:
            f.write(json.dumps(h) + "\n")
    if not args.smoke:
        # 保持分数汇总（主判据输入）：需要 W 臂的终值，收割时跨文件配对计算
        with open(OUT / f"final_{args.arm.lower()}_s{args.seed}.json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[done] {args.arm} s{args.seed} {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
