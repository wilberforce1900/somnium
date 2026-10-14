#!/usr/bin/env python
"""ITER9 · 梦侧深挖第一站：终局语义的容量阶梯与 λ_rew 加权。

预登记：ITER9-PLAN.md（阈值与决策树跑前锁定）。协议全同 P0-D3/D4 的 PRE 段
（ITER8-PLAN.md §10），单变元 = {d_h, λ_rew}，周期 2400（4× 协议预算——
四世界教训：d_h≥256 在 600 周期=未启动，平坦读数欠训/容量不可分）。

关键复用：终局符号准确率仪器 `terminal_acc` 从冻结实现
（experiments/env_e_nonstat/pilot_terminal.py）导入——语义与 P0-D4 逐位同源；
T64 臂的 acc0@600 应当逐位复刻 D4 的 0.5424/0.5064（同代码路径 + 同 seed 的
等价性自检，见 ITER9-PLAN.md §2.1）。

反幻觉：不判梦增益（无梦期训练）；读数 = 可读性（readout），非行为利用。

用法：
  python experiments/term_scale/pilot_terminal_scale.py \
      --seed 0 --d-h 256 --lam-rew 1.0 --cycles 2400 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                    # noqa: E402
from src.dream import EpisodeBuffer                                # noqa: E402
from src.envs import random_rollout                                # noqa: E402
from src.envs.nonstat_world import NonStatWorld                    # noqa: E402
from src.substrate import LatentState                              # noqa: E402
from experiments.env_e_nonstat.pilot_terminal import (             # noqa: E402
    _rollout_toward,
    terminal_acc,
)

HERE = Path(__file__).resolve().parent
OUT = HERE


def terminal_acc_split(model, env, episodes=40):
    """A/B 拆分诊断（探索性，非预登记读数）：+1/−1 终局各自的符号准确率。

    与 terminal_acc 同口径（锁 torch seed 4245、40 局随机 rollout、±1 转移），
    但用独立评估环境（ev2）——不扰动主读数 ev0 的 rng 流。
    返回 (acc_pos, acc_neg, n_pos, n_neg)；无事件侧为 None。
    """
    torch.manual_seed(4245)
    model.eval()
    n_pos = n_neg = ok_pos = ok_neg = 0
    with torch.no_grad():
        for _ in range(episodes):
            obs, act, rew, _ = random_rollout(env)
            h = model.substrate.spawn(1).h
            for t in range(act.shape[0]):
                x = model._x(obs[t:t + 1], act[t:t + 1])
                s, _, _, _ = model.taiji.mixed_step_parts(LatentState(h), x)
                h = s.h
                if abs(float(rew[t])) > 0.5:
                    r_hat = float(model.predict_reward(h))
                    pos = float(rew[t]) > 0
                    ok = (r_hat > 0) == pos
                    if pos:
                        n_pos += 1
                        ok_pos += ok
                    else:
                        n_neg += 1
                        ok_neg += ok
    model.train()
    return (ok_pos / n_pos if n_pos else None,
            ok_neg / n_neg if n_neg else None, n_pos, n_neg)


def _mean_last(rows, key, k=10, upto=None):
    """末 k 个良值（可选 cycle≤upto）的均值；无良值返回 None。"""
    rs = [r[key] for r in rows
          if r["cycle"] <= (upto or 10**9) and r[key] is not None]
    rs = rs[-k:]
    return sum(rs) / len(rs) if rs else None


def train_scale(seed, device, d_h=64, lam_rew=1.0, cycles=2400, eval_every=5,
                batch=32, t_win=8, lr=1e-3, hidden=32, ep_per_cycle=8,
                wake_batches=10, clip=1.0, frac_a=0.3, frac_b=0.15):
    torch.manual_seed(seed)
    random.seed(seed)
    env = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed,
                       regime=0)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, lam_rew=lam_rew,
                       norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev0 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 900)
    ev2 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 902)

    rows = []
    for cycle in range(1, cycles + 1):
        for _ in range(ep_per_cycle):
            r = env.rng.random()
            if r < frac_a:
                buffer.add_episode(*_rollout_toward(env, (2, 3)))
            elif r < frac_a + frac_b:
                buffer.add_episode(*_rollout_toward(env, (4, 5)))
            else:
                buffer.add_episode(*random_rollout(env))
        for i in range(wake_batches):
            b = buffer.sample_windows_mixed(
                batch, t_win,
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=0.5)
            loss, _ = model.loss_on_batch(b["obs"], b["act"],
                                          b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if cycle == 1 or cycle % eval_every == 0:
            a0, m0, n0 = terminal_acc(model, ev0)
            rows.append({"cycle": cycle, "phase": "train",
                         "acc0": a0, "mae0": m0, "n0": n0})
            if cycle % 100 == 0 or cycle == cycles:
                print(f"[term-scale s{seed} dh{d_h} lam{lam_rew:g}] "
                      f"c{cycle}/{cycles} acc0={a0} (n0={n0})", flush=True)

    acc600 = _mean_last(rows, "acc0", upto=600)
    acc1200 = _mean_last(rows, "acc0", upto=1200)
    acc_end = _mean_last(rows, "acc0")
    split = terminal_acc_split(model, ev2)
    last = next((r for r in reversed(rows) if r["acc0"] is not None), {})
    rec = {"seed": seed, "d_h": d_h, "lam_rew": lam_rew, "norm": "ln_post",
           "cycles": cycles, "eval_every": eval_every,
           "frac_a": frac_a, "frac_b": frac_b,
           "acc0_at600": acc600, "acc0_at1200": acc1200, "acc0_end": acc_end,
           "delta_1200_end": (acc_end - acc1200
                              if acc_end is not None and acc1200 is not None
                              else None),
           "mae_end": last.get("mae0"), "n_end": last.get("n0"),
           "split_acc_pos": split[0], "split_acc_neg": split[1],
           "split_n_pos": split[2], "split_n_neg": split[3]}
    name = f"term_scale_dh{d_h}_lam{lam_rew:g}_s{seed}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"TERM-SCALE s{seed} dh{d_h} lam{lam_rew:g}: "
          f"{json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--lam-rew", type=float, default=1.0)
    ap.add_argument("--cycles", type=int, default=2400)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_scale(a.seed, dev, d_h=a.d_h, lam_rew=a.lam_rew, cycles=a.cycles)


if __name__ == "__main__":
    main()
