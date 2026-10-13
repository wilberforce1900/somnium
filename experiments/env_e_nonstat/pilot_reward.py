#!/usr/bin/env python
"""ENV-E v2 pilot · 奖励头版本（P0-D2，v1 无效后的仪器更换，跑前锁版）。

v1（MPC 回报版）无效判定（2026-10-13 双 seed）：翻转前 m0 仅 +0.07~+0.11
——MPC 随机射击 + obs 头闭环在本架构上从未形成稳健奔目标行为（E0 历史同病），
行为层读数测不出"适应"，τ 无意义。如实登记。

P0-D2 仪器（更换理由：角色翻转本就打在**奖励通道**上——A 到达在两 regime
下分别为 +1/−1；奖励头的逐 regime 误差是预测层读数（仪器连续性）且直接
绑定策略语义（哪格是目标），且不依赖行为形成）：

- 协议与 v1 全同（PRE600@ρ0 → 翻转 → POST600@ρ1；d_h64/ln_post/mixed/clip1.0
  /30% 引导；seed{0,1}）；仅换评估器。
- 评估：每 5 cycle，在强制 regime=0/1 的克隆环境上各采 6 局随机 rollout 的
  全部转移，计算奖励头平均绝对误差 e0(c)/e1(c)（锁 seed 4244）。
- τ 定义（同形）：D = e1(T_flip) − mean(POST 末 10 次 e1)；
  τ50/τ90 = 首个 e1(c) ≤ e1(T_flip) − {0.5,0.9}·D 的 post cycle 数。
- ρ0 保持：Δ0 = mean(POST 末 10 次 e0) − mean(PRE 末 10 次 e0)。
- 反幻觉：仍不判梦增益；λ 阶梯由 τ 反推另立预登记。
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

HERE = Path(__file__).resolve().parent
OUT = HERE


def reward_err(model, env, episodes=6):
    """逐 regime 奖励头平均绝对误差（随机 rollout 全转移；锁 seed 4244）。"""
    torch.manual_seed(4244)
    model.eval()
    errs = []
    with torch.no_grad():
        for _ in range(episodes):
            obs, act, rew, _ = random_rollout(env)
            h = model.substrate.spawn(1).h
            for t in range(act.shape[0]):
                x = model._x(obs[t:t + 1], act[t:t + 1])
                from src.substrate import LatentState
                s, _, _, _ = model.taiji.mixed_step_parts(LatentState(h), x)
                h = s.h
                r_hat = model.predict_reward(h)
                errs.append(float((r_hat - rew[t]).abs()))
    model.train()
    return sum(errs) / len(errs)


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
    e1_at_flip = None
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
            e0 = reward_err(model, ev0)
            e1 = reward_err(model, ev1)
            phase = "pre" if cycle <= pre else "post"
            rows.append({"cycle": cycle, "phase": phase,
                         "e0": round(e0, 5), "e1": round(e1, 5)})
            if cycle == pre:
                e1_at_flip = e1
            if cycle % 100 == 0 or cycle == total:
                print(f"[pilotD2 s{seed}] c{cycle}/{total} {phase} "
                      f"e0={e0:.4f} e1={e1:.4f}", flush=True)

    post_rows = [r for r in rows if r["phase"] == "post"]
    e1_end = sum(r["e1"] for r in post_rows[-10:]) / min(10, len(post_rows))
    e0_end = sum(r["e0"] for r in post_rows[-10:]) / min(10, len(post_rows))
    pre_rows = [r for r in rows if r["phase"] == "pre"]
    e0_pre = sum(r["e0"] for r in pre_rows[-10:]) / min(10, len(pre_rows))
    drop = e1_at_flip - e1_end

    def tau(frac):
        thr = e1_at_flip - frac * drop
        for r in post_rows:
            if r["e1"] <= thr:
                return r["cycle"] - pre
        return None

    rec = {"seed": seed, "d_h": d_h, "pre": pre, "post": post,
           "e1_at_flip": round(e1_at_flip, 5), "e1_end": round(e1_end, 5),
           "e0_end": round(e0_end, 5), "e0_pre_end": round(e0_pre, 5),
           "rho0_retention_rise": round(e0_end - e0_pre, 5),
           "tau50": tau(0.5), "tau90": tau(0.9)}
    with open(OUT / f"pilotD2_s{seed}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"pilotD2_s{seed}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"PILOTD2 s{seed}:", json.dumps(rec), flush=True)
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
