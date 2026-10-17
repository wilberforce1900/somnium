#!/usr/bin/env python
"""ITER14 §2 · 读梦 v2 三臂复测（R2 门触发）：噪声余量下的读梦反馈能否修复校准。

预登记：ITER14-PLAN.md §2（全锁）。
- 协议与 R2 逐字同：k5 + σ0.2 + d64 + 2400cyc + guided 0.3 + mixed + clip1.0；
  噪声收集/eval 同分布（noisy_obs/measure_cov95 从 cal_margin 冻结导入）。
- 三臂（单变元=读梦反馈）：
  A 基线=纯 wake；B 读梦=每 cycle 一轮 consolidation-only 梦期（read_replay=True、
  ObsCoverage blind=True 盲区过采样）；C 对照=同 B 但 blind=False（均匀格位）。
- 读数：cov95_end（主）+ 终局步覆盖 + σ̂ 分箱 + mean_sig/mae（报告）+ consol_steps。
- 门（双 seed 配对）：G1 cov95(B_s) ≥ cov95(A_s)+0.20；G2 cov95(B_s) ≥ cov95(C_s)+0.10；
  正=G1∧G2。
反幻觉：只测校准修复，不判梦增益。

用法：python experiments/readdream_v2/pilot_readdream.py --arm B --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                        # noqa: E402
from src.dream import (DreamConfig, DreamLog, DreamScheduler,          # noqa: E402
                       EpisodeBuffer, ObsCoverage)
from src.envs import guided_rollout, random_rollout                    # noqa: E402
from src.envs.rich_world import RichWorld                              # noqa: E402
from experiments.cal_margin.pilot_cal_margin import (                  # noqa: E402
    K, measure_cov95, noisy_obs)

HERE = Path(__file__).resolve().parent
OUT = HERE

SIGMA = 0.2
D64 = 64


def train_readdream(seed, device, arm="A", d_h=D64, sigma_n=SIGMA,
                    cycles=2400, eval_every=100, batch=32, t_win=8, lr=1e-3,
                    hidden=32, ep_per_cycle=8, wake_batches=10, clip=1.0,
                    tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = RichWorld(complexity=K, seed=seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev = RichWorld(complexity=K, seed=seed + 900)

    sched = None
    coverage = None
    if arm in ("B", "C"):
        cfg = DreamConfig(imagination=False, consolidation=True,
                          reverse_learning=False, rehearsal=False,
                          read_replay=True, batch=batch, t_win=t_win)
        coverage = ObsCoverage(d_obs=env.d_obs, dims=(0, 1), bins=6)
        coverage.blind = (arm == "B")
        sched = DreamScheduler(model, buffer, cfg, log=DreamLog(),
                               rng=random.Random(seed), coverage=coverage)

    rows = []
    consol_steps = 0
    for cycle in range(1, cycles + 1):
        for _ in range(ep_per_cycle):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            obs, act, rew, rare = roll(env)
            buffer.add_episode(noisy_obs(obs, sigma_n), act, rew, rare)
        for i in range(wake_batches):
            b = buffer.sample_windows_mixed(
                batch, t_win,
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=0.5)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"],
                                          b["rew"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if coverage is not None:
                coverage.update(b["obs"])
        if sched is not None:
            sched.dream_phase(opt, clip=clip)   # 仅 consolidation（读梦反馈轮）
            consol_steps += 1
        if cycle == 1 or cycle % eval_every == 0:
            c = measure_cov95(model, ev, sigma_n)
            rows.append({"cycle": cycle, **c})
            if cycle % 200 == 0 or cycle == cycles:
                print(f"[readdream {arm} s{seed}] c{cycle}/{cycles} "
                      f"cov95={c['cov95']} (term={c['cov95_term']})",
                      flush=True)

    end = rows[-1]
    rec = {"seed": seed, "arm": arm, "d_h": d_h, "k": K, "sigma_n": sigma_n,
           "cycles": cycles, "eval_every": eval_every,
           "consol_steps": consol_steps,
           "cov95_end": end["cov95"], "cov95_term_end": end["cov95_term"],
           "bins_end": end["bins"], "n_end": end["n"],
           "mean_sig_end": end["mean_sig"], "mae_end": end["mae"]}
    name = f"readdream_arm{arm}_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"READDREAM {arm} s{seed}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["A", "B", "C"], default="A")
    ap.add_argument("--d-h", type=int, default=D64)
    ap.add_argument("--cycles", type=int, default=2400)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_readdream(a.seed, dev, arm=a.arm, d_h=a.d_h, cycles=a.cycles,
                    eval_every=a.eval_every, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
