#!/usr/bin/env python
"""ENV-E pilot · 适应时间测量（ITER8-PLAN §P0-C，跑前锁版）。

目的（奥卡姆）：ENV-E 全矩阵的唯一前置 = 模型对 regime 切换的**适应时间 τ**。
原提案 λ∈{0.002–0.05} 的驻留时长（200–500 步）≪ 实测适应时间（~2–5K 步）——
未校时钟的矩阵测到的全是"学混合分布"而非"适应切换"。本 pilot 用**单次阶跃**
（cleanest step response）实测 τ，λ 阶梯由 τ 反推（另行预登记，不在本文件）。

协议（锁版）：
- 环境：LatentGrid grid8 / horizon50 / p_threat 0.02（=E0 锚）；ρ0: slip=0.1
  （E0 同款，仪器连续性；偏离提案的 0.05——理由：与 A3 重做矩阵同锚）；
  ρ1: slip=0.45。regime 不进观测（部分可观测），切换无提示。
- 模型：d_h=64 / ln_post / hidden32 / lr1e-3（=健康仪器标准配置）。
- 日程：wake-only + **mixed replay**（iter8 阶段0 标准协议）+ clip 1.0；
  30% 引导局；8 局/cycle；10 wake 批。
- 时序：PRE=1000 cycles @ρ0 → 一次性翻转 env.slip → POST=1000 cycles @ρ1。
- 评估（每 2 cycle，无 MPC）：两个**预固定** eval 缓冲（ρ0 env seed+900 /
  ρ1 env seed+901，各 10 局）上的潜态预测 MSE。

τ 定义（锁版，无阈值魔法）：
- m1(c) = ρ1-eval MSE；D = m1(T_switch) − mean(POST 段末 20 次 eval)；
  τ50 / τ90 = 切换后首个满足 m1(c) ≤ m1(T_switch) − {0.5, 0.9}·D 的 cycle。
- ρ0 保持：Δ0 = mean(POST 末 20 次 ρ0 eval) − mean(PRE 末 20 次 ρ0 eval)
  （上升 = 干扰/遗忘量）。

判读边界（反幻觉）：本 pilot 只测 τ（系统属性），**不判梦增益**——
dream vs wake 的恢复对照属于 λ 阶梯锁定后的全矩阵（≥3 seed 预注册）；
n=1 的"增益"一律不作数。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                     # noqa: E402
from src.dream import EpisodeBuffer                 # noqa: E402
from src.envs.latent_grid import (LatentGrid,       # noqa: E402
                                  guided_rollout, random_rollout)
from src.substrate import LatentState               # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

SLIP_RHO0 = 0.10
SLIP_RHO1 = 0.45


def make_env(seed, slip):
    return LatentGrid(grid=8, slip=slip, p_threat_move=0.02, horizon=50, seed=seed)


def eval_regime(model, buf, batch, t_win):
    torch.manual_seed(4242)
    model.eval()
    b = buf.sample_windows(batch, t_win, rng=random.Random(4242))
    h = model.substrate.spawn(b["obs"].shape[0]).h
    err = 0.0
    with torch.no_grad():
        for t in range(b["obs"].shape[1]):
            x = model._x(b["obs"][:, t], b["act"][:, t])
            s, _, _, _ = model.taiji.mixed_step_parts(LatentState(h), x)
            h = s.h
            tgt = model.embed(b["obs_next"][:, t])
            err += float(((h - tgt) ** 2).mean())
    model.train()
    return err / b["obs"].shape[1]


def train_pilot(seed, device, d_h=64, pre=1000, post=1000, eval_every=2,
                batch=32, t_win=8, lr=1e-3, hidden=32, ep_per_cycle=8,
                wake_batches=10, grid=8, clip=1.0, recent_frac=0.5):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(seed, SLIP_RHO0)                 # 训练 env：slip 可翻转
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev0 = make_env(seed + 900, SLIP_RHO0)           # 预固定双 regime eval 集
    ev1 = make_env(seed + 901, SLIP_RHO1)
    eval0, eval1 = EpisodeBuffer(capacity=99), EpisodeBuffer(capacity=99)
    for _ in range(10):
        eval0.add_episode(*random_rollout(ev0))
        eval1.add_episode(*random_rollout(ev1))

    total = pre + post
    rows = []
    m1_at_switch = None
    for cycle in range(1, total + 1):
        if cycle == pre + 1:
            env.slip = SLIP_RHO1                    # 一次性阶跃，无提示
        for _ in range(ep_per_cycle):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(wake_batches):
            b = buffer.sample_windows_mixed(
                batch, t_win, rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=recent_frac)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if cycle == 1 or cycle % eval_every == 0:
            m0 = eval_regime(model, eval0, batch, t_win)
            m1 = eval_regime(model, eval1, batch, t_win)
            phase = "pre" if cycle <= pre else "post"
            rows.append({"cycle": cycle, "phase": phase,
                         "mse_rho0": round(m0, 6), "mse_rho1": round(m1, 6)})
            if cycle == pre:
                m1_at_switch = m1
            if cycle % 100 == 0 or cycle == total:
                print(f"[pilot s{seed}] c{cycle}/{total} {phase} "
                      f"m0={m0:.4f} m1={m1:.4f}", flush=True)

    post_rows = [r for r in rows if r["phase"] == "post"]
    m1_end = sum(r["mse_rho1"] for r in post_rows[-20:]) / 20
    m0_end = sum(r["mse_rho0"] for r in post_rows[-20:]) / 20
    pre_rows = [r for r in rows if r["phase"] == "pre"]
    m0_pre_end = sum(r["mse_rho0"] for r in pre_rows[-20:]) / 20
    drop = m1_at_switch - m1_end

    def tau(frac):
        thr = m1_at_switch - frac * drop
        for r in post_rows:
            if r["mse_rho1"] <= thr:
                return r["cycle"] - pre
        return None

    rec = {"seed": seed, "d_h": d_h, "pre": pre, "post": post,
           "slip_rho0": SLIP_RHO0, "slip_rho1": SLIP_RHO1,
           "m1_at_switch": round(m1_at_switch, 6), "m1_end": round(m1_end, 6),
           "m0_end": round(m0_end, 6), "m0_pre_end": round(m0_pre_end, 6),
           "rho0_retention_rise": round(m0_end - m0_pre_end, 6),
           "tau50": tau(0.5), "tau90": tau(0.9)}
    tag = f"pilot_s{seed}"
    with open(OUT / f"{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{tag}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"PILOT s{seed}:", json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--pre", type=int, default=1000)
    ap.add_argument("--post", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_pilot(args.seed, dev, d_h=args.d_h, pre=args.pre, post=args.post,
                eval_every=args.eval_every, grid=args.grid)


if __name__ == "__main__":
    main()
