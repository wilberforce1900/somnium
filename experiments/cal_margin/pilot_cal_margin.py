#!/usr/bin/env python
"""ITER13 · R2：校准余量重审（A5 复审前置）——更难任务+观测噪声是否开出 cov95 余量。

预登记：ITER13-PLAN.md §2（门锁）+ §2-R 补锁（数值锁）。
- 环境：RichWorld k5（有序目标 4、H=260、d_obs=31；run_long 冻结配方：guided 0.3
  引导 + mixed 回放 + clip 1.0）。
- 压力轴：观测噪声 σ_n ∈ {0.0, 0.2} 收集期加（训练与 eval 同分布）。
- cov95 仪器：固定 eval 集（seed+900 随机局 10 条、同分布加噪）教师强制 rollout
  每步 (r̂, σ̂, r)；coverage95 = mean(|r−r̂| ≤ 1.96σ̂) 全体步 + 终局步 + σ̂ 四分位分箱。
- 门（§2）：cov95 < 90%（d64 双 seed、任一噪声档）→ 校准余量实存 → 读梦 v2 三臂
  另立预登记；≥90% → A5 挂起维持（负结果照记）。
反幻觉：只测校准，不判梦增益。

用法：python experiments/cal_margin/pilot_cal_margin.py --seed 0 --noise 0.2 --device cuda
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
from src.dream import EpisodeBuffer                                 # noqa: E402
from src.envs import guided_rollout, random_rollout                 # noqa: E402
from src.envs.rich_world import RichWorld                           # noqa: E402
from src.substrate import LatentState                               # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE

K = 5                    # R2 补锁：k5 有序目标世界
SIGMAS = (0.0, 0.2)      # 观测噪声档（补锁）
D64, D256 = 64, 256


def noisy_obs(obs, sigma_n):
    return obs + sigma_n * torch.randn_like(obs) if sigma_n > 0 else obs


def measure_cov95(model, eval_env, sigma_n, episodes=10, seed=4242):
    """校准覆盖：教师强制 rollout 每步 (r̂, σ̂, r)。RNG 存档/恢复不扰训练流。

    返回 cov95 全体步、终局步、σ̂ 四分位分箱覆盖、样本量。
    """
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    model.eval()
    r_hats, sigs, trues = [], [], []
    term_idx = set()
    with torch.no_grad():
        for _ in range(episodes):
            obs, act, rew, _ = random_rollout(eval_env)
            obs = noisy_obs(obs, sigma_n)
            h = model.substrate.spawn(1).h
            for t in range(act.shape[0]):
                x = model._x(obs[t:t + 1], act[t:t + 1])
                s, _, _, _ = model.taiji.mixed_step_parts(LatentState(h), x)
                h = s.h
                r_hats.append(float(model.predict_reward(h)))
                sigs.append(float(model.predict_uncertainty(h)))
                trues.append(float(rew[t]))
                if abs(float(rew[t])) > 0.5:
                    term_idx.add(len(trues) - 1)
    model.train()
    torch.set_rng_state(state)
    if not trues:
        return {"cov95": None, "cov95_term": None, "bins": None, "n": 0}
    z = 1.96

    def cov(idx_list):
        ok = sum(abs(trues[i] - r_hats[i]) <= z * sigs[i] for i in idx_list)
        return round(ok / len(idx_list), 4) if idx_list else None

    order = sorted(range(len(sigs)), key=lambda i: sigs[i])
    q = len(order) // 4
    bins = [cov(order[lo:lo + q]) for lo in (0, q, 2 * q, 3 * q)] + \
        [cov(order[3 * q:])]
    return {"cov95": cov(range(len(trues))),
            "cov95_term": cov(sorted(term_idx)),
            "bins": bins, "n": len(trues),
            "mean_sig": round(sum(sigs) / len(sigs), 4),
            "mae": round(sum(abs(trues[i] - r_hats[i])
                             for i in range(len(trues))) / len(trues), 4)}


def train_cal(seed, device, d_h=D64, sigma_n=0.2, cycles=2400,
              eval_every=100, batch=32, t_win=8, lr=1e-3, hidden=32,
              ep_per_cycle=8, wake_batches=10, clip=1.0, tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = RichWorld(complexity=K, seed=seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev = RichWorld(complexity=K, seed=seed + 900)

    rows = []
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
        if cycle == 1 or cycle % eval_every == 0:
            c = measure_cov95(model, ev, sigma_n)
            rows.append({"cycle": cycle, **c})
            if cycle % 200 == 0 or cycle == cycles:
                print(f"[cal s{seed} dh{d_h} sig{sigma_n:g}] "
                      f"c{cycle}/{cycles} cov95={c['cov95']} "
                      f"(term={c['cov95_term']} bins={c['bins']})", flush=True)

    end = rows[-1]
    rec = {"seed": seed, "d_h": d_h, "k": K, "sigma_n": sigma_n,
           "cycles": cycles, "eval_every": eval_every,
           "cov95_end": end["cov95"], "cov95_term_end": end["cov95_term"],
           "bins_end": end["bins"], "n_end": end["n"],
           "mean_sig_end": end["mean_sig"], "mae_end": end["mae"]}
    name = f"cal_k{K}_dh{d_h}_sig{sigma_n:g}_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"CAL dh{d_h} sig{sigma_n:g} s{seed}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-h", type=int, default=D64)
    ap.add_argument("--noise", type=float, default=0.2,
                    choices=SIGMAS, help=f"观测噪声档 {SIGMAS}")
    ap.add_argument("--cycles", type=int, default=2400)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_cal(a.seed, dev, d_h=a.d_h, sigma_n=a.noise, cycles=a.cycles,
              eval_every=a.eval_every, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
