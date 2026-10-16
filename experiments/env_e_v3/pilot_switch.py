#!/usr/bin/env python
"""ITER12 · ENV-E 重开第二站（Stage 2）：λ 阶梯矩阵——非平稳维护度与梦调度。

预登记：ITER12-PLAN.md（阈值与决策树跑前锁定）。
- PRE 2400@ρ0（建立，同 ITER11——**与归档逐位一致**）→ **切换相位 2400**：
  env.lam=1/dwell 连续随机翻转（NonStatWorld lam 机制，不作手动翻转）。
- 读数：acc_sw(t) = 当前 env.regime 对应半的终局符号准确率（in-regime 维护度）；
  主读数 acc_sw_end=末十均值。
- 梦臂=E0 标准前奏（同 ITER10/11）。
反幻觉：只测语义维护度；不判行为梦增益；不碰剂量/分阶。

用法：
  python experiments/env_e_v3/pilot_switch.py --seed 0 --schedule wake --dwell 750 --device cuda
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


def train_switch(seed, device, schedule="wake", dwell=None, d_h=64,
                 dream_batches=200, pre=2400, post=2400, eval_every=5,
                 batch=32, t_win=8, lr=1e-3, hidden=32, ep_per_cycle=8,
                 wake_batches=10, clip=1.0, frac_a=0.3, frac_b=0.15,
                 lam=None, tag_suffix=""):
    # lam 直接给定（ITER12 §6 修正版：由 180 步/周期校准换算 dwell-周期阶梯）；
    # 缺省走 dwell=步 的旧路径（ITER12 §5-R 作废的首矩阵）。
    if lam is None:
        dwell = 750 if dwell is None else int(dwell)
        lam = 1.0 / dwell
        name = f"switch_{schedule}_dwell{dwell}_s{seed}{tag_suffix}"
    else:
        dwell = None
        name = f"switch_{schedule}_lam{lam:.3e}_s{seed}{tag_suffix}"
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
                print(f"[switch {schedule} s{seed} dwell{dwell}] "
                      f"dream {j + 1}/{dream_batches}", flush=True)

    rows = []
    total = pre + post
    for cycle in range(1, total + 1):
        if cycle == pre + 1:
            env.lam = lam          # 切换相位开始：连续随机翻转
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
        phase = "pre" if cycle <= pre else "sw"
        if cycle == 1 or cycle % eval_every == 0 or cycle == pre:
            a0, m0, n0 = terminal_acc(model, ev0)
            row = {"cycle": cycle, "phase": phase,
                   "acc0": a0, "mae0": m0, "n0": n0}
            if phase == "sw" or cycle == pre:
                a1, m1, n1 = terminal_acc(model, ev1)
                reg = env.regime
                row.update(acc1=a1, mae1=m1, n1=n1, regime=reg,
                           acc_sw=(a0 if reg == 0 else a1))
            rows.append(row)
            if cycle % 200 == 0 or cycle == total:
                extra = (f" acc_sw={row['acc_sw']} regime={row['regime']}"
                         if "acc_sw" in row else "")
                print(f"[switch {schedule} s{seed} dwell{dwell}] "
                      f"c{cycle}/{total} {phase} acc0={a0}{extra} "
                      f"switches={env.switches}", flush=True)

    pre_rows = [r for r in rows if r["phase"] == "pre"]
    sw_rows = [r for r in rows if r["phase"] == "sw"]
    acc0_pre_end = _mean_last(pre_rows, "acc0")
    cross_pre = None
    for i in range(len(pre_rows)):
        w = pre_rows[max(0, i - 9):i + 1]
        if len(w) >= 5 and sum(x["acc0"] for x in w) / len(w) >= 0.85:
            cross_pre = pre_rows[i]["cycle"]
            break
    acc_sw_end = _mean_last(sw_rows, "acc_sw")
    sw_valid = [r["acc_sw"] for r in sw_rows if r["acc_sw"] is not None]
    acc_sw_phase = sum(sw_valid) / len(sw_valid) if sw_valid else None
    cur_env = ev1 if env.regime == 1 else ev0
    split = terminal_acc_split(model, cur_env)
    swc = [r for r in sw_rows if r.get("acc_sw") is not None]
    last = swc[-1] if swc else {}
    rec = {"seed": seed, "schedule": schedule, "d_h": d_h,
           "dwell": dwell, "lam": lam,
           "dream_batches": dream_batches if schedule == "dream" else 0,
           "pre": pre, "post": post, "eval_every": eval_every,
           "frac_a": frac_a, "frac_b": frac_b,
           "acc0_pre_end": acc0_pre_end, "cross_pre": cross_pre,
           "mae0_pre_end": _mean_last(pre_rows, "mae0"),
           "acc1_at_switch": next((r.get("acc1") for r in rows
                                   if r["cycle"] == pre), None),
           "acc_sw_end": acc_sw_end, "acc_sw_phase": acc_sw_phase,
           "acc0_sw_end": _mean_last(sw_rows, "acc0"),
           "acc1_sw_end": _mean_last(sw_rows, "acc1"),
           "switches": env.switches, "final_regime": env.regime,
           "mae_sw_end": last.get("mae1" if last.get("regime") == 1
                                  else "mae0"),
           "n_sw_end": last.get("n1" if last.get("regime") == 1 else "n0"),
           "split_acc_pos": split[0], "split_acc_neg": split[1],
           "split_n_pos": split[2], "split_n_neg": split[3]}
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    if schedule == "dream":
        sched.log.save(OUT / f"{name}.dream.jsonl")
    print(f"SWITCH {schedule} dwell{dwell} s{seed}: {json.dumps(rec)}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", choices=["wake", "dream"], default="wake")
    ap.add_argument("--dwell", type=int, default=None,
                    help="平均驻留（步）：λ=1/dwell（旧路径；ITER12 §6 修正版请用 --lam）")
    ap.add_argument("--lam", type=float, default=None,
                    help="直接给每步切换率（§6 修正版：λ={1/(750·180),1/(150·180),1/(50·180)}）")
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--dream-pre", type=int, default=200)
    ap.add_argument("--pre", type=int, default=2400)
    ap.add_argument("--post", type=int, default=2400)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_switch(a.seed, dev, schedule=a.schedule, dwell=a.dwell, d_h=a.d_h,
                 dream_batches=a.dream_pre, pre=a.pre, post=a.post,
                 lam=a.lam, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
