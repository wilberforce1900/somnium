#!/usr/bin/env python
"""ENV-E v2 pilot · 策略层适应时间测量（ITER8-PLAN §7，跑前锁版）。

v1（slip）教训：预测层 regime 太温和（预切换已泛化）。v2 用**角色翻转**
（A/B 的 goal/hazard 语义随 regime 互换）强制策略层+奖励头同时重学。

协议（锁版）：
- 环境：NonStatWorld(grid8, slip 0.1, lam=0, horizon 50)——lam=0、regime 由
  脚本控制（阶跃式：PRE 全 ρ0 → 翻转 → POST 全 ρ1；λ 矩阵另立预登记）。
- 模型：d_h=64 / ln_post / mixed replay / clip 1.0 / 30% 引导局
  （引导读 obs[2:4]=A 增量：ρ0 期指向真目标；ρ1 期指向陷阱——负样本采集
  反而加速发现翻转，如实保留并注记）。
- 主指标：**逐 regime MPC 回报** m_regime(c)——同一模型在强制 regime=0/1 的
  克隆环境上各跑 3 局 MPC（horizon 5, k 32, seed 4243，锁版）。
- τ 定义（与预测层 pilot 同形，仪器连续性）：
  D = m1(T_flip) − mean(POST 末 10 次 m1)；τ50/τ90 = 首个
  m1(c) ≥ m1(T_flip) + {0.5, 0.9}·D 的 post cycle 数。
- ρ0 保持：Δ0 = mean(POST 末 10 次 m0) − mean(PRE 末 10 次 m0)。
- 反幻觉：本 pilot 不判梦增益（n=2 的增益不作数）；λ 阶梯由 τ 反推
  （dwell ∈ {5τ, τ, τ/3}）后另立预登记才跑全矩阵。
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


def mpc_return(model, env, episodes=3, horizon=5, k=32):
    """逐 regime MPC 回报（env.regime 由调用方强制）。锁 seed 4243。"""
    torch.manual_seed(4243)
    model.eval()
    rets = []
    with torch.no_grad():
        for _ in range(episodes):
            obs = env.reset()
            h = model.ground(obs)
            total, done = 0.0, False
            while not done:
                a, _ = model.plan(obs, h, horizon=horizon, k=k)
                obs, r, done, _ = env.step(a)
                total += r
                hs, _ = model.rollout_step(h.unsqueeze(0), obs.unsqueeze(0),
                                           torch.tensor([a], dtype=torch.long))
                h = hs[0]
            rets.append(total)
    model.train()
    return sum(rets) / len(rets)


def train_pilot(seed, device, d_h=64, pre=600, post=600, eval_every=10,
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
    # 逐 regime 评估环境（共享世界几何：重置随机但 regime 强制）
    ev0 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 900)
    ev1 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 901)

    total = pre + post
    rows = []
    m1_at_flip = None
    for cycle in range(1, total + 1):
        if cycle == pre + 1:
            env.regime = 1                     # 一次性角色翻转，无提示
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
            m0 = mpc_return(model, ev0)
            m1 = mpc_return(model, ev1)
            phase = "pre" if cycle <= pre else "post"
            rows.append({"cycle": cycle, "phase": phase,
                         "m0": round(m0, 4), "m1": round(m1, 4)})
            if cycle == pre:
                m1_at_flip = m1
            if cycle % 100 == 0 or cycle == total:
                print(f"[pilot2 s{seed}] c{cycle}/{total} {phase} "
                      f"m0={m0:.3f} m1={m1:.3f}", flush=True)

    post_rows = [r for r in rows if r["phase"] == "post"]
    m1_end = sum(r["m1"] for r in post_rows[-10:]) / min(10, len(post_rows))
    m0_end = sum(r["m0"] for r in post_rows[-10:]) / min(10, len(post_rows))
    pre_rows = [r for r in rows if r["phase"] == "pre"]
    m0_pre = sum(r["m0"] for r in pre_rows[-10:]) / min(10, len(pre_rows))
    rise = m1_end - m1_at_flip

    def tau(frac):
        thr = m1_at_flip + frac * rise
        for r in post_rows:
            if r["m1"] >= thr:
                return r["cycle"] - pre
        return None

    rec = {"seed": seed, "d_h": d_h, "pre": pre, "post": post,
           "m1_at_flip": round(m1_at_flip, 4), "m1_end": round(m1_end, 4),
           "m0_end": round(m0_end, 4), "m0_pre_end": round(m0_pre, 4),
           "rho0_retention_drop": round(m0_pre - m0_end, 4),
           "tau50": tau(0.5), "tau90": tau(0.9)}
    with open(OUT / f"pilot2_s{seed}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"pilot2_s{seed}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"PILOT2 s{seed}:", json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pre", type=int, default=600)
    ap.add_argument("--post", type=int, default=600)
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_pilot(a.seed, dev, pre=a.pre, post=a.post)


if __name__ == "__main__":
    main()
