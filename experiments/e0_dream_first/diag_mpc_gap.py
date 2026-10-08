#!/usr/bin/env python
"""E0 遗留问题诊断：dream_first 潜态预测更好（MSE 0.235 vs 0.905）但 MPC 控制
反而纯超时（-0.5 vs -0.202）——预测-控制分离的定位。

三假设：
    H1 奖励头欠接地：dream 的 rew_mse 显著高于 wake（想象预训练无奖励信号）
    H2 想象塌缩：dream 的射击轨迹预测回报方差≈0 / 解码端点趋同 → argmax 无效
    H3 门控偏置：α 分布显著不同 → 动力学模式差异

用法：.venv/bin/python experiments/e0_dream_first/diag_mpc_gap.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_e0 import FULL, mpc_return                    # noqa: E402
from src.core import WorldModel                        # noqa: E402
from src.dream import (DreamConfig, DreamLog,          # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def train(schedule, seed, cfg):
    torch.manual_seed(seed)
    random.seed(seed)
    env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                     horizon=cfg["horizon"], seed=seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=cfg["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=cfg["batch"], t_win=cfg["t_win"]),
                           log=DreamLog(), rng=random.Random(seed))
    ev_env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                        horizon=cfg["horizon"], seed=seed + 999)
    eval_buf = EpisodeBuffer(capacity=9999)
    for _ in range(cfg["eval_episodes"]):
        eval_buf.add_episode(*random_rollout(ev_env))

    if schedule == "dream_first":
        for _ in range(cfg["dream_pre_batches"]):
            sched.dream_phase(opt)
    for cycle in range(cfg["cycles"]):
        for _ in range(cfg["episodes_per_cycle"]):
            roll = guided_rollout if env.rng.random() < cfg["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(cfg["wake_batches"]):
            b = buffer.sample_windows(
                cfg["batch"], cfg["t_win"],
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=cfg["episodes_per_cycle"])
            sched.wake_update(opt, b)
    return model, eval_buf, cfg, seed


def dissect(model, eval_buf, cfg, seed):
    """逐部件测量：pred/rew/dec MSE、α 分布、射击多样性、MPC 回报。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, cfg["t_win"], rng=random.Random(4242))
    B, T = b["obs"].shape[0], b["obs"].shape[1]
    l_pred = l_rew = l_dec = 0.0
    alphas = []
    with torch.no_grad():
        h = model.substrate.spawn(B).h
        for t in range(T):
            h, a = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            target = model.embed(b["obs_next"][:, t])
            l_pred += torch.mean((h - target) ** 2)
            l_rew += torch.mean((model.predict_reward(h) - b["rew"][:, t]) ** 2)
            l_dec += torch.mean((model.decode_obs(h) - b["obs_next"][:, t]) ** 2)
            alphas.append(a)
    alphas = torch.cat(alphas).squeeze(-1)
    # H2：射击多样性——同一起点的 k 条想象轨迹，预测回报方差 + 解码端点离散度
    diversity = []
    for trial in range(5):
        obs = b["obs"][trial, 0]
        h0 = model.substrate.spawn(1).h[0]
        k = 64
        k_obs = obs.unsqueeze(0).expand(k, -1)
        h_b = h0.unsqueeze(0).expand(k, -1)
        acts = torch.randint(0, 4, (k, 6))
        R = torch.zeros(k)
        ends = []
        for t in range(6):
            h_b, _ = model.rollout_step(h_b, k_obs, acts[:, t])
            R += model.predict_reward(h_b)
            k_obs = model.decode_obs(h_b)
        ends.append(k_obs)
        e = ends[0]
        diversity.append((float(R.std()),
                          float((e - e.mean(0)).norm(dim=1).mean())))
    model.train()
    return {
        "pred_mse": float(l_pred / T), "rew_mse": float(l_rew / T),
        "dec_mse": float(l_dec / T),
        "alpha_mean": float(alphas.mean()), "alpha_std": float(alphas.std()),
        "R_std_mean": sum(d[0] for d in diversity) / len(diversity),
        "endpoint_disp": sum(d[1] for d in diversity) / len(diversity),
        "mpc_return": mpc_return(model, cfg, seed + 500),
    }


def uncert_mpc_return(model, cfg, seed, lam=1.0, episodes=None):
    """修复候选 F2：不确定性感知 MPC——轨迹得分 = Σr̂ − λΣσ（知不知→控制保守）。"""
    torch.manual_seed(4243)
    env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                     horizon=cfg["horizon"], seed=seed)
    model.eval()
    returns = []
    with torch.no_grad():
        for _ in range(episodes or cfg["mpc_episodes"]):
            obs = env.reset()
            h = model.ground(obs) if hasattr(model, "ground") else model.substrate.spawn(1).h[0]
            total, done = 0.0, False
            while not done:
                k = cfg["mpc_k"]
                k_obs = obs.unsqueeze(0).expand(k, -1)
                h_b = h.unsqueeze(0).expand(k, -1)
                acts = torch.randint(0, model.n_actions, (k, cfg["mpc_horizon"]))
                R = torch.zeros(k)
                for t in range(cfg["mpc_horizon"]):
                    h_b, _ = model.rollout_step(h_b, k_obs, acts[:, t])
                    R += model.predict_reward(h_b) - lam * model.predict_uncertainty(h_b)
                    k_obs = model.decode_obs(h_b)
                a = int(acts[int(torch.argmax(R)), 0].item())
                obs, r, done, _ = env.step(a)
                total += r
                hs, _ = model.rollout_step(h.unsqueeze(0), obs.unsqueeze(0),
                                           torch.tensor([a], dtype=torch.long))
                h = hs[0]
            returns.append(total)
    model.train()
    return sum(returns) / len(returns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {}
    models = {}
    for schedule in ("wake_only", "dream_first"):
        model, eval_buf, cfg, seed = train(schedule, args.seed, FULL)
        models[schedule] = (model, cfg, seed)
        d = dissect(model, eval_buf, cfg, seed)
        out[schedule] = {k: round(v, 6) for k, v in d.items()}
        print(f"[{schedule}] " + " ".join(f"{k}={v:.4f}" for k, v in d.items()))
    w, dr = out["wake_only"], out["dream_first"]
    print("\n== 假设检验 ==")
    print(f"H1 奖励头欠接地: rew_mse {dr['rew_mse']:.4f} vs {w['rew_mse']:.4f} "
          f"→ {'成立' if dr['rew_mse'] > 2 * w['rew_mse'] else '不成立'}")
    print(f"H2 想象塌缩: R_std {dr['R_std_mean']:.5f} vs {w['R_std_mean']:.5f}, "
          f"端点离散 {dr['endpoint_disp']:.4f} vs {w['endpoint_disp']:.4f} "
          f"→ {'成立' if dr['R_std_mean'] < 0.5 * w['R_std_mean'] else '不成立'}")
    print(f"H3 门控偏置: α {dr['alpha_mean']:.3f}±{dr['alpha_std']:.3f} vs "
          f"{w['alpha_mean']:.3f}±{w['alpha_std']:.3f}")
    print("\n== 修复候选 F2：不确定性感知 MPC（R − λΣσ）==")
    for schedule, (model, cfg, seed) in models.items():
        for lam in (0.0, 0.5, 1.0, 2.0):
            r = uncert_mpc_return(model, cfg, seed + 500, lam=lam)
            print(f"  {schedule} λ={lam:.1f}: mpc={r:+.3f}")
            out.setdefault("f2", {})[f"{schedule}_lam{lam}"] = round(r, 4)
    p = REPO / "experiments/e0_dream_first/diag_mpc_gap.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"已写 {p.name}")


if __name__ == "__main__":
    main()
