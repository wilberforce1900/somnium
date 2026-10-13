#!/usr/bin/env python
"""ENV-E v2 pilot · 终局条件化版本（P0-D3，第三台仪器，跑前锁版）。

仪器迭代史（全登记）：
- v1 MPC 回报：无效——随机射击 MPC 在本架构从未形成稳健奔目标行为（翻转前
  m0 仅 +0.07~+0.11），行为层读数测不出适应。
- v2 奖励头全转移 MAE：无效——奖励流 99% 为 −0.01 步进背景，终局 ±1 事件
  仅 ~0.8%，regime 语义信号（Δ≈0.008）被背景波动（±0.01）淹没（数学上
  稀释：e1 翻转前后 0.0367→0.0366 不动）。
- v3 终局条件化（本文件）：只在"到达 A/B"的转移上测——
    acc(c) = 符号准确率 sign(r̂)==sign(r)（regime 语义的直接读数）
    mae(c) = 终局转移上的 |r̂−r|
  终局事件在两 regime 下语义相反（ρ0: A=+1/B=−1；ρ1 反转）——翻转前 acc1
  应≈0（预测反号），恢复到≈1。背景步进完全剥离。

协议：其余与 v1/v2 全同（PRE600@ρ0→翻转→POST600@ρ1；d_h64/ln_post/mixed/
clip1.0/30% 引导；seed{0,1}）。评估：每 5 cycle 各 regime 40 局随机 rollout，
收集 A/B 到达转移（锁 seed 4245）。
τ 定义（同形，方向适配）：D = acc1_end − acc1(T_flip)（应>0）；
τ50/τ90 = 首个 acc1(c) ≥ acc1(T_flip) + {0.5,0.9}·D。
反幻觉：仍不判梦增益；λ 阶梯由 τ 反推另立预登记。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                     # noqa: E402
from src.dream import EpisodeBuffer                 # noqa: E402
from src.envs import guided_rollout, random_rollout  # noqa: E402
from src.envs.nonstat_world import NonStatWorld     # noqa: E402
from src.substrate import LatentState               # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE


def terminal_acc(model, env, episodes=40):
    """A/B 到达转移上的符号准确率与 MAE（锁 seed 4245）。"""
    torch.manual_seed(4245)
    model.eval()
    hits, errs = 0, 0.0
    n = 0
    with torch.no_grad():
        for _ in range(episodes):
            obs, act, rew, _ = random_rollout(env)
            h = model.substrate.spawn(1).h
            for t in range(act.shape[0]):
                x = model._x(obs[t:t + 1], act[t:t + 1])
                s, _, _, _ = model.taiji.mixed_step_parts(LatentState(h), x)
                h = s.h
                if abs(float(rew[t])) > 0.5:        # A/B 到达（±1）
                    r_hat = float(model.predict_reward(h))
                    hits += int((r_hat > 0) == (float(rew[t]) > 0))
                    errs += abs(r_hat - float(rew[t]))
                    n += 1
    model.train()
    if n == 0:
        return None, None, 0
    return hits / n, errs / n, n


def train_pilot(seed, device, d_h=64, pre=600, post=600, eval_every=5,
                batch=32, t_win=8, lr=1e-3, hidden=32, ep_per_cycle=8,
                wake_batches=10, clip=1.0):
    torch.manual_seed(seed)
    random.seed(seed)
    env = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed,
                       regime=0)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev0 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 900)
    ev1 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 901)

    total = pre + post
    rows = []
    a1_at_flip = None
    for cycle in range(1, total + 1):
        if cycle == pre + 1:
            env.regime = 1
        for _ in range(ep_per_cycle):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(wake_batches):
            b = buffer.sample_windows_mixed(
                batch, t_win, rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=0.5)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if cycle == 1 or cycle % eval_every == 0:
            ev0.regime, ev1.regime = 0, 1
            a0, m0e, n0 = terminal_acc(model, ev0)
            a1, m1e, n1 = terminal_acc(model, ev1)
            phase = "pre" if cycle <= pre else "post"
            rows.append({"cycle": cycle, "phase": phase,
                         "acc0": a0, "acc1": a1,
                         "mae0": m0e, "mae1": m1e, "n1": n1})
            if cycle == pre:
                a1_at_flip = a1
            if cycle % 100 == 0 or cycle == total:
                print(f"[pilotD3 s{seed}] c{cycle}/{total} {phase} "
                      f"acc0={a0} acc1={a1} (n1={n1})", flush=True)

    post_rows = [r for r in rows if r["phase"] == "post" and r["acc1"] is not None]
    a1_end = sum(r["acc1"] for r in post_rows[-10:]) / min(10, len(post_rows))
    a0_end = sum(r["acc0"] for r in post_rows[-10:]) / min(10, len(post_rows))
    pre_rows = [r for r in rows if r["phase"] == "pre" and r["acc0"] is not None]
    a0_pre = sum(r["acc0"] for r in pre_rows[-10:]) / min(10, len(pre_rows))
    rise = a1_end - (a1_at_flip if a1_at_flip is not None else a1_end)

    def tau(frac):
        if a1_at_flip is None:
            return None
        thr = a1_at_flip + frac * rise
        for r in post_rows:
            if r["acc1"] is not None and r["acc1"] >= thr:
                return r["cycle"] - pre
        return None

    rec = {"seed": seed, "d_h": d_h, "pre": pre, "post": post,
           "acc1_at_flip": a1_at_flip, "acc1_end": round(a1_end, 4),
           "acc0_end": round(a0_end, 4), "acc0_pre_end": round(a0_pre, 4),
           "rho0_retention_drop": round(a0_pre - a0_end, 4),
           "tau50": tau(0.5), "tau90": tau(0.9)}
    with open(OUT / f"pilotD3_s{seed}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"pilotD3_s{seed}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"PILOTD3 s{seed}:", json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_pilot(a.seed, dev)


if __name__ == "__main__":
    main()
