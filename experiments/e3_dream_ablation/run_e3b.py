#!/usr/bin/env python
"""E3b · 读梦（A5，重设版 2026-10-08）实验入口。ROADMAP §1.4。

原设计依赖 E2 卦码盲区统计——卦码已降级，故重设（预注册偏离记录）：
    读梦 = ObsCoverage 直方图（醒期 obs 空间覆盖）→ 盲区格位 → 想象伪观测偏置。
    首个闭环反馈：梦的产物被读取，反过来改变下一晚的梦（螺旋进化的实体）。

新增 L5 种子：WorldModel.var_head（奖励异方差 σ）——校准度量基础（E3b 引入，
    两臂同置，不影响对照公平性）。

三臂对照（想象伪观测来源，其余全同）：
    read   盲区偏置合成 obs（读梦开）
    random 均匀格位合成 obs（同分布、无读梦偏置——干净对照）
    none   无梦

预注册（2026-10-08 锁定，跑前）：
    校准误差 = 按 σ 五分位分箱，各箱 |r−μ|≤1.96σ 经验覆盖率对 0.95 的平均绝对偏差
    A5 正信号 ⇔ read 臂校准误差相对 random 臂下降 ≥20%（两 seed 方向一致）
    辅助（记录不设阈）：按 σ 弃权的风险-覆盖选择性增益；pred_mse。
seeds {0,1} × 3 臂 = 6 run。判定：登记表结论列（跑后回填）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                          # noqa: E402
from src.dream import (DreamConfig, DreamLog,            # noqa: E402
                       DreamScheduler, EpisodeBuffer, ObsCoverage)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results" / "registry.md"

CFG = dict(grid=6, horizon=30, slip=0.1, p_threat=0.1, guided_frac=0.3,
           cycles=6, episodes=8, wake_batches=8, batch=16, t_win=8,
           d_h=48, lr=3e-3, eval_eps=8, buffer_cap=400, bins=6)
GOAL = (5, 5)


def make_env(seed):
    return LatentGrid(grid=CFG["grid"], slip=CFG["slip"], p_threat_move=CFG["p_threat"],
                      horizon=CFG["horizon"], seed=seed, with_hazard=True,
                      fixed_start=(0, 0), fixed_goal=GOAL)


def calibration_eval(model, eval_buf):
    """校准误差 + 选择性增益 + pred_mse（锁 seed 4242）。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(64, CFG["t_win"], rng=random.Random(4242))
    mus, sigmas, rs = [], [], []
    with torch.no_grad():
        h = model.substrate.spawn(b["obs"].shape[0]).h
        for t in range(CFG["t_win"]):
            h, _ = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            mus.append(model.predict_reward(h))
            sigmas.append(model.predict_uncertainty(h))
            rs.append(b["rew"][:, t])
    mu = torch.cat(mus); sg = torch.cat(sigmas); r = torch.cat(rs)
    # pred mse（奖励维度）
    mse = float(((mu - r) ** 2).mean())
    # 校准误差：σ 五分位分箱的 95% 覆盖率偏差
    order = torch.argsort(sg)
    bins = torch.chunk(order, 5)
    errs = []
    for idx in bins:
        cov = float(((r[idx] - mu[idx]).abs() <= 1.96 * sg[idx]).float().mean())
        errs.append(abs(cov - 0.95))
    cal_err = sum(errs) / len(errs)
    # 选择性：按 σ 弃权的风险-覆盖曲线增益
    risk_all = float((r - mu).abs().mean())
    risks, covs = [], []
    for frac in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = int(len(order) * frac)
        risks.append(float((r[order[:k]] - mu[order[:k]]).abs().mean()))
        covs.append(frac)
    sel_gain = risk_all - (sum(risks[:-1]) / (len(risks) - 1))  # 平均选择性增益
    model.train()
    return {"cal_err": cal_err, "cov95": 1 - cal_err, "sel_gain": sel_gain,
            "rew_mse": mse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["read", "random", "none"], default="read")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or "untagged"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env = make_env(args.seed)
    model = WorldModel(d_obs=8, n_actions=4, d_h=CFG["d_h"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    buffer = EpisodeBuffer(capacity=CFG["buffer_cap"])
    cov = ObsCoverage(d_obs=8, dims=(0, 1), bins=CFG["bins"])
    dream_on = args.arm != "none"
    # read 与 random 共用同一覆盖器；差异只在 pseudo_obs 的 blind 开关
    cov.blind = (args.arm == "read")
    sched = DreamScheduler(
        model, buffer,
        DreamConfig(batch=CFG["batch"], t_win=CFG["t_win"],
                    imagination=dream_on, consolidation=dream_on,
                    reverse_learning=dream_on, rehearsal=dream_on,
                    read_replay=(args.arm == "read")),
        log=DreamLog(), rng=random.Random(args.seed),
        coverage=cov if dream_on else None)

    ev = make_env(args.seed + 900)
    eval_buf = EpisodeBuffer(capacity=99)
    for _ in range(CFG["eval_eps"]):
        eval_buf.add_episode(*random_rollout(ev))

    t0 = time.time()
    env_steps = 0
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            roll = guided_rollout if env.rng.random() < CFG["guided_frac"] else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(CFG["wake_batches"]):
            b = buffer.sample_windows(
                CFG["batch"], CFG["t_win"],
                rng=random.Random(args.seed * 1000 + cycle * 100 + i),
                only_last=CFG["episodes"])
            sched.wake_update(opt, b)
        if dream_on:
            sched.dream_phase(opt)
        m = calibration_eval(model, eval_buf)
        print(f"[{args.arm}] cycle {cycle+1}/{CFG['cycles']} cal_err={m['cal_err']:.4f} "
              f"cov95={m['cov95']:.3f} sel={m['sel_gain']:+.4f} mse={m['rew_mse']:.4f} "
              f"cover={cov.covered_frac():.2f}", flush=True)

    rec = {"arm": args.arm, "seed": args.seed,
           "cal_err": round(m["cal_err"], 6), "cov95": round(m["cov95"], 4),
           "sel_gain": round(m["sel_gain"], 6), "rew_mse": round(m["rew_mse"], 6),
           "covered_frac": round(cov.covered_frac(), 4),
           "dream_log_len": len(sched.log)}
    out = REPO / "experiments/e3_dream_ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"details_e3b_{args.arm}_{tag}_{args.seed}.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(
            f"| 2026-10-08 | E3b | read_dream_{args.arm} | {args.seed} | - "
            f"| {rec['rew_mse']} | cal={rec['cal_err']} sel={rec['sel_gain']} "
            f"| 待判定 | full/{tag} |\n"
        )
    print(f"完成：{time.time()-t0:.1f}s cal_err={rec['cal_err']}")


if __name__ == "__main__":
    main()
