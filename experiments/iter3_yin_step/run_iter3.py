#!/usr/bin/env python
"""迭代 #3(a)：阴步幅 ‖∇E‖ 轨迹——解释梦模型醒期晚期回阳（分析性诊断）。

现象（迭代#2 留档）：两日程醒训练初期 α 都滑向 ~0.16；wake_only 停在阴，
dream_first 晚期回升至 0.6–0.96。

主假设（阴步衰减说）：梦预训练的能量平滑抹平地形 → 阴步有效幅值
    lr·‖∇E(h)‖ 变小失效 → 预测目标追踪只能靠阳步 → α 被抬升。
预测：dream 的 gradE/阴臂贡献 随训练下降且低于 wake；阳臂贡献补位上升。

逐 cycle 测量（固定 eval 窗，锁 seed）：
    α、‖∇E‖、阴臂贡献 ‖(1−α)Δ阴‖、阳臂贡献 ‖α·Δ阳‖、
    目标间距 ‖enc(o')−h‖、能量头权重范数（地形演化）。

用法：.venv/bin/python experiments/iter3_yin_step/run_iter3.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/e0_dream_first"))

from run_e0 import FULL                                # noqa: E402
from src.core import WorldModel                        # noqa: E402
from src.dream import (DreamConfig, DreamLog,          # noqa: E402
                       DreamScheduler, EpisodeBuffer)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402
from src.substrate import LatentState                  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def build(seed, cfg):
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
    ev = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                    horizon=cfg["horizon"], seed=seed + 999)
    eval_buf = EpisodeBuffer(capacity=9999)
    for _ in range(cfg["eval_episodes"]):
        eval_buf.add_episode(*random_rollout(ev))
    return env, model, opt, buffer, sched, eval_buf


def dissect(model, eval_buf):
    """混合更新两臂分解 + 能量地形统计（与 taiji.mixed_step 逐步同式重算）。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, FULL["t_win"], rng=random.Random(4242))
    B, T = b["obs"].shape[0], b["obs"].shape[1]
    acc = {"alpha": 0.0, "gradE": 0.0, "yin_c": 0.0, "yang_c": 0.0, "tgap": 0.0}
    n = 0
    with torch.no_grad():
        h = model.substrate.spawn(B).h
        for t in range(T):
            obs_t, act_t = b["obs"][:, t], b["act"][:, t]
            x = model._x(obs_t, act_t)
            # 阳臂：dt·tanh(cell([h,x]))
            dy = model.yang.dt * model.substrate.forward_delta(h, x)
            # 阴臂：−lr·∇E（局地开梯度，同 yin.step 手法）
            h_leaf = h.detach().requires_grad_(True)
            with torch.enable_grad():
                e = model.substrate.energy_of(h_leaf).sum()
                (g,) = torch.autograd.grad(e, h_leaf)
            di = -model.yin.lr * g
            a = model.taiji.alpha(LatentState(h), x)
            acc["alpha"] += float(a.mean())
            acc["gradE"] += float(g.norm(dim=-1).mean())
            acc["yin_c"] += float(((1 - a) * di).norm(dim=-1).mean())
            acc["yang_c"] += float((a * dy).norm(dim=-1).mean())
            tgt = model.embed(b["obs_next"][:, t])
            acc["tgap"] += float((tgt - h).norm(dim=-1).mean())
            h = h + a * dy + (1 - a) * di  # 与 mixed_step 同式推进
            n += 1
    model.train()
    ew = float(model.substrate.energy_head[0].weight.norm())
    return {k: round(v / n, 5) for k, v in acc.items()} | {"energy_w": round(ew, 4)}


def train_logged(schedule, seed):
    cfg = FULL
    env, model, opt, buffer, sched, eval_buf = build(seed, cfg)
    if schedule == "dream_first":
        for _ in range(cfg["dream_pre_batches"]):
            sched.dream_phase(opt)
    trace = []
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
        trace.append({"cycle": cycle, **dissect(model, eval_buf)})
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {}
    for schedule in ("wake_only", "dream_first"):
        trace = train_logged(schedule, args.seed)
        out[schedule] = trace
        alphas = [t["alpha"] for t in trace]
        gradE = [t["gradE"] for t in trace]
        yin_c = [t["yin_c"] for t in trace]
        yang_c = [t["yang_c"] for t in trace]
        print(f"\n[{schedule}] α:   " + " ".join(f"{a:.2f}" for a in alphas))
        print(f"[{schedule}] ‖∇E‖:" + " ".join(f"{g:.3f}" for g in gradE))
        print(f"[{schedule}] 阴臂:" + " ".join(f"{v:.3f}" for v in yin_c))
        print(f"[{schedule}] 阳臂:" + " ".join(f"{v:.3f}" for v in yang_c))
        print(f"[{schedule}] 能量头w: {trace[0]['energy_w']} → {trace[-1]['energy_w']}")
    # 分歧点与相关性
    w, d = out["wake_only"], out["dream_first"]
    print("\n== 分歧诊断 ==")
    for i, (a, b2) in enumerate(zip(w, d)):
        if abs(a["alpha"] - b2["alpha"]) > 0.2:
            print(f"首个 α 分歧 >0.2：cycle {i}（wake {a['alpha']:.2f} vs dream {b2['alpha']:.2f}；"
                  f"‖∇E‖ {a['gradE']:.3f} vs {b2['gradE']:.3f}；"
                  f"阴臂 {a['yin_c']:.3f} vs {b2['yin_c']:.3f}）")
            break
    def corr(x, y):
        x = torch.tensor(x); y = torch.tensor(y)
        x = x - x.mean(); y = y - y.mean()
        den = float(x.norm() * y.norm())
        return float((x * y).sum() / den) if den > 0 else 0.0
    print(f"dream: corr(α, 阴臂) = {corr([t['alpha'] for t in d], [t['yin_c'] for t in d]):+.2f}；"
          f"corr(α, ‖∇E‖) = {corr([t['alpha'] for t in d], [t['gradE'] for t in d]):+.2f}；"
          f"corr(α, 目标间距) = {corr([t['alpha'] for t in d], [t['tgap'] for t in d]):+.2f}")
    p = REPO / "experiments/iter3_yin_step/iter3_result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"已写 {p.name}")


if __name__ == "__main__":
    main()
