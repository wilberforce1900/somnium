#!/usr/bin/env python
"""ITER13 · R3：遗忘构造（镜像坐标冲突）——双层门：Stage A 有效性 → Stage B 梦护旧。

预登记：ITER13-PLAN.md §3（门锁）+ §3-R 补锁（数值锁）。
- 环境：NonStatWorld ρ0（grid8/obs8；收集=iter9 家族 fracA.3+fracB.15+随机；mixed 回放）
- 镜像（锁）：phase 2 收集期 obs 坐标 x 镜像——out[0]=1−obs[0]、out[2]=−obs[2]、
  out[4]=−obs[4]（x 位置与 x 增量反号；任务不变、观测翻转）
- 读数：旧坐标固定 eval 集（seed+900 随机局）解码 MSE + 潜态 MSE 于 flip/end 两测；
  retention_decode = mse_end/mse_flip；终局符号 acc_old 用冻结仪器 terminal_acc
- Stage A（arm=wake ×2 seed）：retention_decode > 1.05 双 seed → 遗忘可构造 → Stage B
- Stage B 臂：dream（E0 前奏 200 于出生）+ replay_old（phase 2 半数批次来自冻结旧缓冲）
  梦护旧判据=retention_dream < retention_wake − 0.05 双 seed（收割时判，描述性为主）
反幻觉：只测遗忘与保护，不判梦增益。

用法：python experiments/forget_construct/pilot_forget.py --arm wake --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                      # noqa: E402
from src.dream import (DreamConfig, DreamLog, DreamScheduler,        # noqa: E402
                       EpisodeBuffer)
from src.envs import random_rollout                                  # noqa: E402
from src.envs.nonstat_world import NonStatWorld                      # noqa: E402
from src.substrate import LatentState                                # noqa: E402
from experiments.env_e_nonstat.pilot_terminal import (               # noqa: E402
    _rollout_toward, terminal_acc)

HERE = Path(__file__).resolve().parent
OUT = HERE


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    """坐标 x 镜像（§3-R 锁）：x 位置与 x 增量反号，其余维度不动。

    列索引（d_obs 维）：x 位置/目标 x 增量/威胁 x 增量 = obs 的第 0/2/4 列。
    """
    out = obs.clone()
    out[..., 0] = 1.0 - obs[..., 0]
    out[..., 2] = -obs[..., 2]
    out[..., 4] = -obs[..., 4]
    return out


def collect_ep(env):
    """ρ0 家族采集：奔A 0.3 + 反奔B 0.15 + 随机 0.55。"""
    r = env.rng.random()
    if r < 0.3:
        return _rollout_toward(env, (2, 3))
    if r < 0.45:
        return _rollout_toward(env, (4, 5))
    return random_rollout(env)


def old_decode_mse(model, ev, device, episodes=10, seed=4246):
    """旧坐标固定 eval 集解码 MSE + 潜态 MSE（教师强制）。RNG 存档/恢复。"""
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    model.eval()
    buf = EpisodeBuffer(capacity=99)
    for _ in range(episodes):
        buf.add_episode(*random_rollout(ev))
    b = buf.sample_windows(16, 8, rng=random.Random(seed))
    o = b["obs"].to(device)
    a = b["act"].to(device)
    on = b["obs_next"].to(device)
    with torch.no_grad():
        h = model.substrate.spawn(o.shape[0]).h
        dec = lat = 0.0
        for t in range(8):
            x = model._x(o[:, t], a[:, t])
            s, _, _, _ = model.taiji.mixed_step_parts(LatentState(h), x)
            h = s.h
            dec += float(((model.decode_obs(h) - on[:, t]) ** 2).mean())
            lat += float(((h - model.embed(on[:, t])) ** 2).mean())
    model.train()
    torch.set_rng_state(state)
    return dec / 8, lat / 8


def train_forget(seed, device, arm="wake", pre=2400, post=2400,
                 dream_batches=200, batch=32, t_win=8, lr=1e-3, hidden=32,
                 ep_per_cycle=8, wake_batches=10, clip=1.0, d_h=64,
                 tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed,
                       regime=0)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind="ln_post").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    old_buffer = EpisodeBuffer(capacity=2000)   # replay_old 臂的冻结旧缓冲
    ev_old = NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50,
                          seed=seed + 900, regime=0)
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=batch, t_win=t_win),
                           log=DreamLog(), rng=random.Random(seed))

    if arm == "dream":
        for j in range(dream_batches):
            sched.dream_phase(opt, clip=clip)
            if (j + 1) % 50 == 0 or j + 1 == dream_batches:
                print(f"[forget {arm} s{seed}] dream {j + 1}/{dream_batches}",
                      flush=True)

    rows = []
    total = pre + post
    for cycle in range(1, total + 1):
        mirrored = cycle > pre
        for _ in range(ep_per_cycle):
            obs, act, rew, rare = collect_ep(env)
            if mirrored:
                obs = mirror_obs(obs)
            buffer.add_episode(obs, act, rew, rare)
            if not mirrored:
                old_buffer.add_episode(obs, act, rew, rare)
        for i in range(wake_batches):
            use_old = arm == "replay_old" and mirrored and i % 2 == 0
            src = old_buffer if use_old else buffer
            b = src.sample_windows_mixed(
                batch, t_win,
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=0.5)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"],
                                          b["rew"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if cycle == pre or cycle == total:
            dec, lat = old_decode_mse(model, ev_old, device)
            state = torch.get_rng_state()
            acc, _, _ = terminal_acc(model, ev_old)
            torch.set_rng_state(state)
            rows.append({"cycle": cycle, "phase": "pre" if cycle == pre else "post",
                         "dec_mse_old": round(dec, 6),
                         "lat_mse_old": round(lat, 6), "acc_old": acc})
            print(f"[forget {arm} s{seed}] c{cycle}/{total} "
                  f"dec={dec:.4f} lat={lat:.4f} acc={acc}", flush=True)

    flip, end = rows[0], rows[-1]
    rec = {"seed": seed, "arm": arm, "d_h": d_h, "pre": pre, "post": post,
           "dream_batches": dream_batches if arm == "dream" else 0,
           "replay_old": arm == "replay_old",
           "dec_mse_flip": flip["dec_mse_old"], "dec_mse_end": end["dec_mse_old"],
           "retention_decode": (round(end["dec_mse_old"] / flip["dec_mse_old"], 4)
                                if flip["dec_mse_old"] > 0 else None),
           "lat_mse_flip": flip["lat_mse_old"], "lat_mse_end": end["lat_mse_old"],
           "retention_latent": (round(end["lat_mse_old"] / flip["lat_mse_old"], 4)
                                if flip["lat_mse_old"] > 0 else None),
           "acc_old_flip": flip["acc_old"], "acc_old_end": end["acc_old"]}
    name = f"forget_{arm}_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"FORGET {arm} s{seed}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["wake", "dream", "replay_old"],
                    default="wake")
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--dream-pre", type=int, default=200)
    ap.add_argument("--pre", type=int, default=2400)
    ap.add_argument("--post", type=int, default=2400)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    train_forget(a.seed, dev, arm=a.arm, pre=a.pre, post=a.post,
                 dream_batches=a.dream_pre, d_h=a.d_h, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
