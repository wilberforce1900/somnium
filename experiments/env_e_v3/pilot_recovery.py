#!/usr/bin/env python
"""ITER11 · ENV-E 重开第一站：单翻转恢复（梦的语义加速在切换下存续吗）。

预登记：ITER11-PLAN.md（阈值与决策树跑前锁定）。
- PRE 2400@ρ0（建立）→ 手动翻转 → POST 2400@ρ1（恢复）；协议逐字复用 ITER9/10
  冻结件（NonStatWorld ρ0 / fracA.3+fracB.15+随机 / mixed / clip1.0 / eval 每 5
  锁 seed 4245 / terminal_acc 冻结仪器）；梦臂=E0 标准前奏（与 ITER10 逐字同）。
- **PRE 段与归档逐位一致**（等价性测试守）：wake 臂 ≡ term_scale_dh64_lam1_s{0,1}、
  dream 臂 ≡ dream_sem_dh64_s{0,1}（c≤2400）。
- τ 定义沿 P0-D3 §10：D = acc1_end − acc1(T_flip)；τ50/τ90 = 首个
  acc1(c) ≥ acc1(T_flip) + {0.5, 0.9}·D；截尾=POST 长。
反幻觉：不判行为层梦增益；本 stage 只测语义恢复；λ>0 不在本 stage。

用法：
  python experiments/env_e_v3/pilot_recovery.py --seed 0 --schedule wake --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                       # noqa: E402
from src.dream import (DreamConfig, DreamLog, DreamScheduler,         # noqa: E402
                       EpisodeBuffer)
from src.envs import random_rollout                                   # noqa: E402
from src.envs.nonstat_world import NonStatWorld                       # noqa: E402
from experiments.env_e_nonstat.pilot_terminal import (                # noqa: E402
    _rollout_toward, terminal_acc)
from experiments.term_scale.pilot_terminal_scale import (             # noqa: E402
    _mean_last, terminal_acc_split)

HERE = Path(__file__).resolve().parent
OUT = HERE


def train_recovery(seed, device, schedule="wake", d_h=64, dream_batches=200,
                   pre=2400, post=2400, eval_every=5, batch=32, t_win=8,
                   lr=1e-3, hidden=32, ep_per_cycle=8, wake_batches=10,
                   clip=1.0, frac_a=0.3, frac_b=0.15, tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed,
                       regime=0)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev0 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 900)
    ev1 = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed + 901,
                       regime=1)
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=batch, t_win=t_win),
                           log=DreamLog(), rng=random.Random(seed))

    if schedule == "dream":
        for j in range(dream_batches):
            sched.dream_phase(opt, clip=clip)
            if (j + 1) % 50 == 0 or j + 1 == dream_batches:
                print(f"[recovery {schedule} s{seed}] dream {j + 1}/{dream_batches}",
                      flush=True)

    rows = []
    total = pre + post
    for cycle in range(1, total + 1):
        if cycle == pre + 1:
            env.regime = 1
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
        phase = "pre" if cycle <= pre else "post"
        if cycle == 1 or cycle % eval_every == 0 or cycle == pre:
            a0, m0, n0 = terminal_acc(model, ev0)
            row = {"cycle": cycle, "phase": phase,
                   "acc0": a0, "mae0": m0, "n0": n0}
            if phase == "post" or cycle == pre:
                a1, m1, n1 = terminal_acc(model, ev1)
                row.update(acc1=a1, mae1=m1, n1=n1)
            rows.append(row)
            if cycle % 200 == 0 or cycle == total:
                extra = (f" acc1={row['acc1']}" if "acc1" in row else "")
                print(f"[recovery {schedule} s{seed} dh{d_h}] "
                      f"c{cycle}/{total} {phase} acc0={a0}{extra}", flush=True)

    pre_rows = [r for r in rows if r["phase"] == "pre"]
    post_rows = [r for r in rows if r["phase"] == "post"]
    a1_rows = [r for r in post_rows if r.get("acc1") is not None]
    acc0_pre_end = _mean_last(pre_rows, "acc0")
    cross_pre = None
    for i in range(len(pre_rows)):
        w = pre_rows[max(0, i - 9):i + 1]
        if len(w) >= 5 and sum(x["acc0"] for x in w) / len(w) >= 0.85:
            cross_pre = pre_rows[i]["cycle"]
            break
    a1_flip = next((r["acc1"] for r in rows if r["cycle"] == pre), None)
    acc1_end = _mean_last(a1_rows, "acc1")
    acc0_post_end = _mean_last(post_rows, "acc0")
    d_tot = (acc1_end - a1_flip
             if acc1_end is not None and a1_flip is not None else None)

    def tau(frac):
        if d_tot is None:
            return None
        thr = a1_flip + frac * d_tot
        for r in a1_rows:
            if r["acc1"] >= thr:
                return r["cycle"] - pre
        return None

    split = terminal_acc_split(model, ev1)
    flip_row = next((r for r in rows if r["cycle"] == pre), {})
    last = a1_rows[-1] if a1_rows else {}
    rec = {"seed": seed, "schedule": schedule, "d_h": d_h,
           "dream_batches": dream_batches if schedule == "dream" else 0,
           "pre": pre, "post": post, "eval_every": eval_every,
           "frac_a": frac_a, "frac_b": frac_b,
           "acc0_pre_end": acc0_pre_end, "cross_pre": cross_pre,
           "mae0_pre_end": _mean_last(pre_rows, "mae0"),
           "acc1_flip": a1_flip, "n_flip": flip_row.get("n1"),
           "acc1_end": acc1_end, "acc1_D": d_tot,
           "tau50": tau(0.5), "tau90": tau(0.9),
           "acc0_post_end": acc0_post_end,
           "mae1_end": last.get("mae1"), "n1_end": last.get("n1"),
           "split_acc_pos": split[0], "split_acc_neg": split[1],
           "split_n_pos": split[2], "split_n_neg": split[3]}
    name = f"recovery_{schedule}_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    if schedule == "dream":
        sched.log.save(OUT / f"{name}.dream.jsonl")
    print(f"RECOVERY {schedule} s{seed}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", choices=["wake", "dream"], default="wake")
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--dream-pre", type=int, default=200)
    ap.add_argument("--pre", type=int, default=2400)
    ap.add_argument("--post", type=int, default=2400)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_recovery(a.seed, dev, schedule=a.schedule, d_h=a.d_h,
                   dream_batches=a.dream_pre, pre=a.pre, post=a.post,
                   tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
