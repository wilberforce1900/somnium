#!/usr/bin/env python
"""ITER10 · 梦侧深挖第二站：梦调度因果（dream-first × 容量 → 终局语义建立速率）。

预登记：ITER10-PLAN.md（阈值与决策树跑前锁定）。

- 梦前奏：E0 标准 `dream_phase × 200`（空缓冲→自动只剩内生想象=E0-b"先梦"引擎；
  clip 1.0 与醒期统一——E0 原版无 clip，登记为夜二后协议选择）。
- 醒期：与 iter9 冻结实现（pilot_terminal_scale.py）逐字同构；无梦路径的
  位级等价由 tests/test_dream_semantics.py 守——醒-only 对照=iter9 归档
  （T64/T256/T1024 × s0/s1）。
公平性：梦前奏=额外离线梯度步；比较轴=醒期周期（真实交互），同 E0 惯例
（PRINCIPLES §11 A3）。反幻觉：不判行为梦增益（E0/P0-B 主题）；只测终局语义建立速率。

用法：
  python experiments/term_scale/pilot_dream_semantics.py --seed 0 --d-h 256 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                     # noqa: E402
from src.dream import (DreamConfig, DreamLog, DreamScheduler,       # noqa: E402
                       EpisodeBuffer)
from src.envs import random_rollout                                 # noqa: E402
from src.envs.nonstat_world import NonStatWorld                     # noqa: E402
from experiments.env_e_nonstat.pilot_terminal import (              # noqa: E402
    _rollout_toward, terminal_acc)
from experiments.term_scale.pilot_terminal_scale import (           # noqa: E402
    _mean_last, terminal_acc_split)

HERE = Path(__file__).resolve().parent
OUT = HERE


def train_dreamsem(seed, device, d_h=64, dream_batches=200, cycles=2400,
                   eval_every=5, batch=32, t_win=8, lr=1e-3, hidden=32,
                   ep_per_cycle=8, wake_batches=10, clip=1.0,
                   frac_a=0.3, frac_b=0.15, tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed,
                       regime=0)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev0 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 900)
    ev2 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 902)
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=batch, t_win=t_win),
                           log=DreamLog(), rng=random.Random(seed))

    # 梦期先行（E0-b）：buffer 空 → dream_phase 自动只剩内生想象（三组件 skip）
    for j in range(dream_batches):
        sched.dream_phase(opt, clip=clip)
        if (j + 1) % 50 == 0 or j + 1 == dream_batches:
            print(f"[dream-sem s{seed} dh{d_h}] dream {j + 1}/{dream_batches}",
                  flush=True)

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
                print(f"[dream-sem s{seed} dh{d_h}] "
                      f"c{cycle}/{cycles} acc0={a0} (n0={n0})", flush=True)

    acc600 = _mean_last(rows, "acc0", upto=600)
    acc1200 = _mean_last(rows, "acc0", upto=1200)
    acc_end = _mean_last(rows, "acc0")
    split = terminal_acc_split(model, ev2)
    last = next((r for r in reversed(rows) if r["acc0"] is not None), {})
    rec = {"seed": seed, "d_h": d_h, "schedule": "dream_first",
           "dream_batches": dream_batches, "norm": "ln_post",
           "cycles": cycles, "eval_every": eval_every,
           "frac_a": frac_a, "frac_b": frac_b,
           "acc0_at600": acc600, "acc0_at1200": acc1200, "acc0_end": acc_end,
           "delta_1200_end": (acc_end - acc1200
                              if acc_end is not None and acc1200 is not None
                              else None),
           "mae_end": last.get("mae0"), "n_end": last.get("n0"),
           "split_acc_pos": split[0], "split_acc_neg": split[1],
           "split_n_pos": split[2], "split_n_neg": split[3]}
    name = f"dream_sem_dh{d_h}_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    sched.log.save(OUT / f"{name}.dream.jsonl")
    print(f"DREAM-SEM s{seed} dh{d_h}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--dream-pre", type=int, default=200,
                    help="梦前奏批数（E0 标准 200；0=无梦对照路径）")
    ap.add_argument("--cycles", type=int, default=2400)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_dreamsem(a.seed, dev, d_h=a.d_h, dream_batches=a.dream_pre,
                   cycles=a.cycles, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
