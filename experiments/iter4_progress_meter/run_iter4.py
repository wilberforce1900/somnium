#!/usr/bin/env python
"""迭代 #4：门=收敛进度计的可证伪检验 + α/σ 双探针对比（分析性诊断）。

迭代#3 读法：阴=纠错学，阳=前向用，门极性编码学习阶段。本轮预注册判定：

    D1 注入检验：收敛模型（任务A，α_A 高）换训新任务 B →
       α_B 应在 1–2 cycle 内回落 <0.5 再爬升；同期 α_A 保持 > α_B+0.15。
       若 α_B 不回落，或 α_A 同步大跌 → 进度计读法被证伪。
    D2 轨迹耦合：相位 2 中 corr(α_B 序列, mse_B 序列) ≤ −0.5。
    D3 探针矩阵：逐窗 α / σ 对真实误差（潜态残差 err_lat、奖励误差 err_rew）
       的 Pearson 相关，A+B 池化。|corr(α,·)| ≥ 0.3 记为有信息量。
       附加：相位 1 末（B 从未训练）α 在 B 窗上是否已低于 A 窗
       （门作为"陌生度探测器"的泛化性——低则强，高则弱但不伤 D1/D2）。

用法：.venv/bin/python experiments/iter4_progress_meter/run_iter4.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                        # noqa: E402
from src.dream import (DreamConfig, DreamLog,          # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CFG = dict(grid=6, horizon=30, slip=0.1, p_threat=0.02, guided_frac=0.3,
           cycles=12, episodes=8, wake_batches=8, batch=16, t_win=8,
           d_h=48, lr=3e-3, eval_eps=8, dream_pre=200, buffer_cap=800)
GOAL_A, GOAL_B = (5, 5), (0, 5)


def make_env(goal, seed):
    return LatentGrid(grid=CFG["grid"], slip=CFG["slip"], p_threat_move=CFG["p_threat"],
                      horizon=CFG["horizon"], seed=seed, with_hazard=False,
                      fixed_start=(0, 0), fixed_goal=goal)


def build(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(GOAL_A, seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=CFG["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    buffer = EpisodeBuffer(capacity=CFG["buffer_cap"])
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=CFG["batch"], t_win=CFG["t_win"]),
                           log=DreamLog(), rng=random.Random(seed))
    evals = {}
    for tag, goal in (("A", GOAL_A), ("B", GOAL_B)):
        ev = make_env(goal, seed + 900 if tag == "A" else seed + 950)
        eb = EpisodeBuffer(capacity=99)
        for _ in range(CFG["eval_eps"]):
            eb.add_episode(*random_rollout(ev))
        evals[tag] = eb
    return env, model, opt, buffer, sched, evals


def window_probes(model, eval_buf):
    """逐窗探针：α、σ、err_lat（潜态残差）、err_rew（奖励误差）。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, CFG["t_win"], rng=random.Random(4242))
    B, T = b["obs"].shape[0], b["obs"].shape[1]
    alpha_w = torch.zeros(B); sigma_w = torch.zeros(B)
    err_lat = torch.zeros(B); err_rew = torch.zeros(B)
    with torch.no_grad():
        h = model.substrate.spawn(B).h
        for t in range(T):
            h, a = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            tgt = model.embed(b["obs_next"][:, t])
            alpha_w += a.squeeze(-1)
            sigma_w += model.predict_uncertainty(h)
            err_lat += (h - tgt).norm(dim=-1)
            err_rew += (model.predict_reward(h) - b["rew"][:, t]).abs()
    model.train()
    return (alpha_w / T, sigma_w / T, err_lat / T, err_rew / T)


def corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(); y = y - y.mean()
    den = float(x.norm() * y.norm())
    return float((x * y).sum() / den) if den > 0 else 0.0


def probe_matrix(model, evals):
    out = {}
    pooled = {"a": [], "s": [], "el": [], "er": []}
    for tag in ("A", "B"):
        a, s, el, er = window_probes(model, evals[tag])
        out[f"alpha_{tag}"] = round(float(a.mean()), 4)
        out[f"corr_alpha_errlat_{tag}"] = round(corr(a, el), 3)
        out[f"corr_alpha_errew_{tag}"] = round(corr(a, er), 3)
        out[f"corr_sigma_errlat_{tag}"] = round(corr(s, el), 3)
        out[f"corr_sigma_errew_{tag}"] = round(corr(s, er), 3)
        for k, v in zip(pooled, (a, s, el, er)):
            pooled[k].append(v)
    p = {k: torch.cat(v) for k, v in pooled.items()}
    out["pooled_corr_alpha_errlat"] = round(corr(p["a"], p["el"]), 3)
    out["pooled_corr_alpha_errew"] = round(corr(p["a"], p["er"]), 3)
    out["pooled_corr_sigma_errlat"] = round(corr(p["s"], p["el"]), 3)
    out["pooled_corr_sigma_errew"] = round(corr(p["s"], p["er"]), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed = args.seed
    env, model, opt, buffer, sched, evals = build(seed)

    # ---- 相位 1：dream_first 训任务 A 至收敛 ----
    for _ in range(CFG["dream_pre"]):
        sched.dream_phase(opt)
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            roll = guided_rollout if env.rng.random() < CFG["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"], rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
    pre = probe_matrix(model, evals)
    print(f"[seed{seed}] 相位1末（A 收敛，B 未见）：alpha_A={pre['alpha_A']} "
          f"alpha_B={pre['alpha_B']}（泛化探针：B 上已低则强）")

    # ---- 相位 2：注入任务 B，只训 B 新鲜数据 ----
    env_b = make_env(GOAL_B, seed + 5)
    trace = []
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            roll = guided_rollout if env_b.rng.random() < CFG["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env_b))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"],
                rng=random.Random(seed * 7777 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
        aB = window_probes(model, evals["B"])
        aA = window_probes(model, evals["A"])
        trace.append({"cycle": cycle,
                      "alpha_B": round(float(aB[0].mean()), 4),
                      "alpha_A": round(float(aA[0].mean()), 4),
                      "err_lat_B": round(float(aB[2].mean()), 4)})
    print(f"[seed{seed}] 相位2 α_B 轨迹：" + " ".join(f"{t['alpha_B']:.2f}" for t in trace))
    print(f"[seed{seed}] 相位2 α_A 轨迹：" + " ".join(f"{t['alpha_A']:.2f}" for t in trace))
    post = probe_matrix(model, evals)

    # ---- 预注册判定 ----
    aB = [t["alpha_B"] for t in trace]
    aA = [t["alpha_A"] for t in trace]
    eB = [t["err_lat_B"] for t in trace]
    dip = min(aB[:4])
    dip_i = aB[:4].index(dip)
    d1 = dip < 0.5 and aA[dip_i] > dip + 0.15 and aB[-1] > dip + 0.2
    d2 = corr(torch.tensor(aB), torch.tensor(eB)) <= -0.5
    print(f"\n[seed{seed}] D1 注入回落：dip={dip:.2f}@cyc{dip_i} α_A={aA[dip_i]:.2f} "
          f"终值α_B={aB[-1]:.2f} → {'✅' if d1 else '❌'}")
    print(f"[seed{seed}] D2 轨迹耦合 corr(α_B, err_B)={corr(torch.tensor(aB), torch.tensor(eB)):+.2f} "
          f"→ {'✅' if d2 else '❌'}")
    print(f"[seed{seed}] D3 探针矩阵 post：α×errLat={post['pooled_corr_alpha_errlat']} "
          f"α×errRew={post['pooled_corr_alpha_errew']} σ×errLat={post['pooled_corr_sigma_errlat']} "
          f"σ×errRew={post['pooled_corr_sigma_errew']}")
    p = REPO / "experiments/iter4_progress_meter"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"iter4_result_seed{seed}.json").write_text(json.dumps(
        {"pre": pre, "trace": trace, "post": post, "D1": bool(d1), "D2": bool(d2)},
        indent=2, ensure_ascii=False))
    print(f"已写 iter4_result_seed{seed}.json")


if __name__ == "__main__":
    main()
