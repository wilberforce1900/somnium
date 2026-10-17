#!/usr/bin/env python
"""ITER14 §1 · R4b：多步开环 rollout——喂自身预测、误差复合（JEPA/潜态主张的主场）。

预登记：ITER14-PLAN.md §1（全锁）。
- 训练与 R4 逐字同（k4 锁步双训、同批次、参数匹配、2400cyc）；结束在进程内开环评估。
- 开环：Somnium = h_0=ground(真 o_0)，此后输入=自身解码（closed-loop in
  imagination、无真实观测回流）；AR = 因果 ctx=8 滑动窗输入 [o_0, 自身预测…]。
- 读数：horizon k∈{1,2,4,8,16} 的 MSE_k / R²_k + 帧拷贝参照（persistent start-copy）
  + 复合比 MSE_16/MSE_1。
- 门：Δ_k = R²(som)−R²(ar)；k=16 主判 ±5pp，k=8 次报告。
反幻觉：架构对照性质，不判梦增益。

用法：python experiments/openloop/pilot_openloop.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                     # noqa: E402
from src.dream import EpisodeBuffer                                 # noqa: E402
from src.substrate import LatentState                               # noqa: E402
from experiments.r4_baseline.run_r4 import (                        # noqa: E402
    ARTransformer, build_eval_buffer, collect_episode, make_env)

HERE = Path(__file__).resolve().parent
OUT = HERE

MAX_K = 16
AR_CTX = 8


def openloop_eval(som, ar, ebuf, device, max_k=MAX_K, batch=32, seed=4242):
    """开环评估（双模型同口径）：k 步后预测 vs 真值；帧拷贝参照。RNG 存档/恢复。"""
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    b = ebuf.sample_windows(batch, max_k + 1, rng=random.Random(seed))
    o = b["obs"].to(device)
    a = b["act"].to(device)
    o_true = o[:, 1:]
    ovar = float(o_true.reshape(-1, o_true.shape[-1]).var(dim=0).mean())
    som.eval()
    ar.eval()
    with torch.no_grad():
        copy = {k: float(((o[:, 0] - o[:, k]) ** 2).mean())
                for k in range(1, max_k + 1)}
        # --- Somnium 闭环：接地起点 + 自身解码回流 ---
        h = som.ground(o[:, 0])
        preds = []
        for t in range(max_k):
            obs_in = o[:, 0] if t == 0 else preds[-1]
            x = som._x(obs_in, a[:, t])
            s, _, _, _ = som.taiji.mixed_step_parts(LatentState(h), x)
            h = s.h
            preds.append(som.decode_obs(h))
        mse_som = {k: float(((preds[k - 1] - o[:, k]) ** 2).mean())
                   for k in range(1, max_k + 1)}
        # --- AR 闭环：自身预测滑入 ctx=8 窗口 ---
        hist = [o[:, 0]]
        mse_ar = {}
        for t in range(max_k):
            win = hist[-AR_CTX:]
            obs_seq = torch.stack(win, dim=1)
            t_in = obs_seq.shape[1]
            act_seq = a[:, max(0, t - t_in + 1): t + 1]
            pred = ar(obs_seq, act_seq)[:, -1]
            hist.append(pred)
            mse_ar[t + 1] = float(((pred - o[:, t + 1]) ** 2).mean())
    som.train()
    ar.train()
    torch.set_rng_state(state)

    def r2(m):
        return round(100 * (1 - m / ovar), 2) if ovar > 0 else float("nan")

    return {"ovar": round(ovar, 6),
            "mse_som": {str(k): round(v, 6) for k, v in mse_som.items()},
            "mse_ar": {str(k): round(v, 6) for k, v in mse_ar.items()},
            "mse_copy": {str(k): round(v, 6) for k, v in copy.items()},
            "r2_som": {str(k): r2(v) for k, v in mse_som.items()},
            "r2_ar": {str(k): r2(v) for k, v in mse_ar.items()},
            "r2_copy": {str(k): r2(v) for k, v in copy.items()},
            "compound_som": round(mse_som[max_k] / max(mse_som[1], 1e-12), 4),
            "compound_ar": round(mse_ar[max_k] / max(mse_ar[1], 1e-12), 4)}


def run_r4b(seed, device, cycles=2400, batch=32, t_win=8, lr=1e-3,
            hidden=32, ep_per_cycle=8, wake_batches=10, clip=1.0,
            tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env("k4", seed)
    som = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=64,
                     hidden=hidden, norm_kind="ln_post").to(device)
    ar = ARTransformer(d_obs=env.d_obs, n_actions=env.n_actions).to(device)
    som_opt = torch.optim.Adam(som.parameters(), lr=lr)
    ar_opt = torch.optim.Adam(ar.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ebuf = build_eval_buffer("k4", seed)

    for cycle in range(1, cycles + 1):
        for _ in range(ep_per_cycle):
            buffer.add_episode(*collect_episode(env, "k4"))
        for i in range(wake_batches):
            b = buffer.sample_windows_mixed(
                batch, t_win,
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=0.5)
            loss_s, _ = som.loss_on_batch(b["obs"], b["act"], b["obs_next"],
                                          b["rew"])
            som_opt.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(som.parameters(), clip)
            som_opt.step()
            pred = ar(b["obs"].to(device), b["act"].to(device))
            loss_a = F.mse_loss(pred, b["obs_next"].to(device))
            ar_opt.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), clip)
            ar_opt.step()
        if cycle % 200 == 0 or cycle == cycles:
            print(f"[r4b s{seed}] c{cycle}/{cycles}", flush=True)

    out = openloop_eval(som, ar, ebuf, device)
    rec = {"seed": seed, "world": "k4", "d_h": 64, "cycles": cycles,
           "max_k": MAX_K, "ar_ctx": AR_CTX,
           "params_som": sum(p.numel() for p in som.parameters()),
           "params_ar": sum(p.numel() for p in ar.parameters()),
           "delta_r2_16": round(out["r2_som"]["16"] - out["r2_ar"]["16"], 2),
           "delta_r2_8": round(out["r2_som"]["8"] - out["r2_ar"]["8"], 2),
           **out}
    name = f"openloop_k4_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        f.write(json.dumps({"cycle": cycles, **out}) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"OPENLOOP s{seed}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=2400)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    run_r4b(a.seed, dev, cycles=a.cycles, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
