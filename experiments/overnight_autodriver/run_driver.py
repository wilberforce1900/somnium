#!/usr/bin/env python
"""Somnium 过夜自驱迭代 · 工程探索档（2026-10-09）

⚠ 基本法合规声明（PRINCIPLES 纪律）：本回路是**工程探索**，不是预注册实验——
其产物是种子库与假设，供次日人审/后续预注册实验取材，不直接产生科学判定。

用户规格：Somnium 连续自驱——每轮训练的"自我疑问"（四探针：σ 不确定度 /
覆盖盲区 / 预测残差 / 门极性）与"梦境残留"（buffer + 梦 log）作为下一轮自主
搜索学习的种子；有效成分与"灰度"（阈间/看不懂的观察）留**种子库**。

轮结构（全部确定性规则，落盘可审计）：
    wake(env_t) → dream（四开关，回放=残留）→ probe → choose → maybe grow → 落盘
  - 探针：residual（eval 窗潜态残差范数）、sigma（σ 均值）、coverage、alpha
  - choose 规则 v0：
      residual > 0.9                → 环境收缩（grid −1，巩固）
      residual < 0.6 且 coverage>0.9 → 环境扩张（grid +1，探索）
      连续两轮 residual 改善 >10%   → 生长事件（widen ×1.5 + 梦剂量 ×1.5）
      其余 → 保持；任何探针落入阈间 → 记 gray（灰度，留种子库待审）
  - 生长：d_h 阶梯 64→96→144→216→324→486→729（cap 1024），梦养大梦者；
    生长后重建优化器（Adam 矩清零，文档化）
  - 硬停：--end-utc（默认 23:00 UTC=明早 07:00 CST）或 --max-rounds
  - 异常：捕获 → 记 gray → 续；连续 3 次 → 安全停止

用法：python experiments/overnight_autodriver/run_driver.py [--end-utc ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.core import WorldModel                       # noqa: E402
from src.dream import (DreamConfig, DreamLog,         # noqa: E402
                       DreamScheduler, EpisodeBuffer, ObsCoverage)
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402
from src.growth import widen_world_model              # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/overnight_autodriver/out"

D_H_LADDER = [64, 96, 144, 216, 324, 486, 729, 1024]
CLIP = 1.0            # v2.2：梯度裁剪（夜二 ~870 轮无裁剪长训发散之鉴）


def _finite(x) -> bool:
    import math
    return x is not None and math.isfinite(x)


def make_env(grid, seed):
    return LatentGrid(grid=grid, slip=0.1, p_threat_move=0.1,
                      horizon=4 * grid, seed=seed, with_hazard=True,
                      fixed_start=(0, 0), fixed_goal=(grid - 1, grid - 1))


def probes(model, eval_buf, sched):
    """四探针：自我疑问的工程化。"""
    torch.manual_seed(4242)
    model.eval()
    b = eval_buf.sample_windows(32, 8, rng=random.Random(4242))
    with torch.no_grad():
        h = model.substrate.spawn(32).h
        err = sig = alp = 0.0
        for t in range(8):
            h, a = model.rollout_step(h, b["obs"][:, t], b["act"][:, t])
            tgt = model.embed(b["obs_next"][:, t])
            err += float((h - tgt).norm(dim=-1).mean())
            sig += float(model.predict_uncertainty(h).mean())
            alp += float(a.mean())
    model.train()
    import math
    return {"residual": round(err / 8, 4),
            "res_norm": round(err / 8 / math.sqrt(model.substrate.d_h), 4),
            "sigma": round(sig / 8, 4),
            "alpha": round(alp / 8, 4),
            "coverage": round(sched.coverage.covered_frac(), 4) if sched.coverage else None}


def build_eval(env_grid, seed):
    ev = make_env(env_grid, seed + 900)
    eb = EpisodeBuffer(capacity=99)
    for _ in range(6):
        eb.add_episode(*random_rollout(ev))
    return eb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-utc", default="2026-10-09T23:00:00")
    ap.add_argument("--max-rounds", type=int, default=99999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=6, help="每轮 wake cycle 数（smoke 可调小）")
    ap.add_argument("--resume", default=None,
                    help="从断点续跑（v2.2）：模型/优化器/缓冲/控制态全恢复")
    args = ap.parse_args()
    end = datetime.fromisoformat(args.end_utc).replace(tzinfo=timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seeds").mkdir(exist_ok=True)
    (OUT / "ckpts").mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    grid, d_h_i, dream_dose = 6, 0, 200
    if args.resume:  # v2.2.1：先按断点宽度建身，再装权重（729 装不进 64 之鉴）
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        dh0 = int(st["extra"].get("seed_rec", {}).get("d_h", D_H_LADDER[0]))
        d_h_i = D_H_LADDER.index(dh0) if dh0 in D_H_LADDER else 0
        del st
    model = WorldModel(d_obs=8, n_actions=4, d_h=D_H_LADDER[d_h_i])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    buffer = EpisodeBuffer(capacity=1200)
    prev_residual, improve_streak, err_streak, round_id = None, 0, 0, 0
    wall, wall_strikes, rounds_since_growth = False, 0, 99
    last_healthy = None

    def restore_from(ckpt_path):
        """从断点恢复模型/优化器/缓冲/控制态（resume 与发散自愈共用）。"""
        nonlocal grid, d_h_i, dream_dose, wall, prev_residual, round_id
        nonlocal improve_streak, wall_strikes, rounds_since_growth, opt
        cyc, extra = load_checkpoint(ckpt_path, model=model, optimizer=opt,
                                     buffer=buffer)
        rec = extra.get("seed_rec", {})
        round_id = int(cyc)
        grid = int(rec.get("grid", grid))
        dh = int(rec.get("d_h", D_H_LADDER[0]))
        d_h_i = D_H_LADDER.index(dh) if dh in D_H_LADDER else 0
        dream_dose = int(rec.get("dream_dose", 200))
        wall = bool(rec.get("wall", False))
        rn = rec.get("probes", {}).get("res_norm")
        prev_residual = rn if _finite(rn) else None
        improve_streak, wall_strikes, rounds_since_growth = 0, 0, 99
        print(f"[resume] 轮 {round_id} d_h{dh} grid{grid} 剂量{dream_dose} 墙{wall}")

    if args.resume:
        restore_from(args.resume)

    cov = ObsCoverage(d_obs=8, bins=grid)
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=16, t_win=8, rare_bias=3.0),
                           log=DreamLog(), rng=random.Random(args.seed), coverage=cov)
    t0 = time.time()

    while round_id < args.max_rounds:
        if datetime.now(timezone.utc) >= end:
            (OUT / "final_report.md").write_text(
                f"# Somnium 过夜自驱 · 终报\n\n硬停于 {datetime.now(timezone.utc).isoformat()}"
                f"（共 {round_id} 轮，{time.time()-t0:.0f}s）。种子库见 seeds/，模型见 ckpts/。\n",
                encoding="utf-8")
            print("== END（时间到）==")
            break
        round_id += 1
        try:
            env = make_env(grid, args.seed * 100 + round_id)
            eval_buf = build_eval(grid, args.seed + round_id)
            # ---- wake：采集 + 更新（梦残留=buffer 全量回放）----
            for cyc in range(args.cycles):
                for _ in range(8):
                    roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
                    buffer.add_episode(*roll(env))
                for i in range(8):
                    b = buffer.sample_windows(16, 8,
                                              rng=random.Random(round_id * 1000 + cyc * 10 + i),
                                              only_last=8)
                    sched.wake_update(opt, b, clip=CLIP)
                sched.dream_phase(opt, clip=CLIP)  # 梦加深（剂量=dream_dose 累计次数由 dose 控制）
            # ---- 梦剂量：额外梦期（回放=残留；生长事件后 ×1.5）----
            for _ in range(max(1, dream_dose // 200)):
                sched.dream_phase(opt, clip=CLIP)
            # ---- probe：自我疑问 ----
            p = probes(model, eval_buf, sched)
            # ---- v2.2 发散自愈：病了就回到上次健康的自己 ----
            gray_pre = []
            if (not _finite(p["res_norm"]) or p["res_norm"] > 3.0
                    or not _finite(p["sigma"])):
                cands = sorted(int(f.stem.split("_")[1])
                               for f in (OUT / "ckpts").glob("round_*.pt"))
                healthy = [r for r in cands
                           if last_healthy is not None and r <= last_healthy]
                if healthy:
                    tgt = max(healthy)
                    gray_pre.append(
                        f"divergence(res={p['res_norm']},σ={p['sigma']})→回滚 round {tgt}")
                    restore_from(OUT / "ckpts" / f"round_{tgt:03d}.pt")
                    eval_buf = build_eval(grid, args.seed + round_id)
                    p = probes(model, eval_buf, sched)
                else:
                    gray_pre.append("divergence 无健康断点，继续观察")
            elif 0 < p["res_norm"] < 2.0 and _finite(p["sigma"]):
                last_healthy = round_id
            # ---- choose：确定性规则 + 灰度（v2.1：并入夜一全部教训）----
            # 教训①阈值用 √d_h 归一残差（健康带 0.29–0.6，墙 0.9+）；
            # 教训②反饥饿：残差高但覆盖高 = 容量失配而非环境太难——只有
            #      覆盖不足(<0.85)时才收缩环境（夜一曾缩到 grid5 地板空转 667 轮）；
            # 教训③生长门槛：仅当前规模健康(res_norm<0.6)才允许生长；
            # 教训④墙检测：生长后 4 轮内 res_norm 恶化>1.2 连续 3 次 →
            #      标记容量墙，本夜不再生长（夜一 729→1024 爆 6.8× 后无路可退）。
            gray = list(gray_pre)
            if 0.55 <= p["res_norm"] <= 0.8:
                gray.append(f"res_norm 阈间 {p['res_norm']}")
            if p["coverage"] is not None and 0.5 <= p["coverage"] <= 0.9:
                gray.append(f"coverage 阈间 {p['coverage']}")
            action = "hold"
            if (p["res_norm"] > 0.8 and p["coverage"] is not None
                    and p["coverage"] < 0.85 and grid > 6):
                grid -= 1
                action = f"shrink→grid{grid}"
            elif p["res_norm"] < 0.55 and p["coverage"] and p["coverage"] > 0.9 and grid < 10:
                grid += 1
                action = f"expand→grid{grid}"
            grew = False
            if prev_residual is not None and prev_residual - p["res_norm"] > 0.1 * prev_residual:
                improve_streak += 1
            else:
                improve_streak = 0
            if (improve_streak >= 2 and not wall
                    and p["res_norm"] < 0.6 and d_h_i < len(D_H_LADDER) - 1):
                d_h_i += 1
                widen_world_model(model, D_H_LADDER[d_h_i],
                                  rng=random.Random(round_id), noise=0.01)
                opt = torch.optim.Adam(model.parameters(), lr=2e-3)  # 生长后重建
                dream_dose = int(dream_dose * 1.5)
                improve_streak = 0
                grew = True
                rounds_since_growth = 0
                action += f" +GROW→d_h{D_H_LADDER[d_h_i]}"
            else:
                rounds_since_growth += 1
            # 墙检测：生长后短窗内急剧恶化 → 停止本夜生长
            if (not wall and grew is False and rounds_since_growth <= 4
                    and prev_residual is not None and p["res_norm"] > 1.2 * prev_residual):
                wall_strikes += 1
                if wall_strikes >= 3:
                    wall = True
                    gray.append(f"capacity wall @d_h{D_H_LADDER[d_h_i]} → 本夜停止生长")
            else:
                wall_strikes = 0
            prev_residual = p["res_norm"]
            # ---- 落盘：种子库 + ckpt ----
            seed_rec = {"round": round_id, "utc": datetime.now(timezone.utc).isoformat(),
                        "grid": grid, "d_h": model.substrate.d_h, "dream_dose": dream_dose,
                        "probes": p, "action": action, "grew": grew, "gray": gray,
                        "wall": wall, "dream_log_len": len(sched.log)}
            (OUT / f"seeds/round_{round_id:03d}.json").write_text(
                json.dumps(seed_rec, ensure_ascii=False, indent=1), encoding="utf-8")
            save_checkpoint(OUT / f"ckpts/round_{round_id:03d}.pt", model=model,
                            optimizer=opt, cycle=round_id, extra={"seed_rec": seed_rec},
                            buffer=buffer)
            # v2 教训修正：ckpt 保留上限（最近 5 份 + 每 50 轮里程碑），防写满盘
            keep = {f"round_{r:03d}.pt" for r in range(max(1, round_id - 4), round_id + 1)}
            keep |= {f"round_{r:03d}.pt" for r in range(50, round_id + 1, 50)}
            for f in (OUT / "ckpts").glob("round_*.pt"):
                if f.name not in keep:
                    f.unlink(missing_ok=True)
            sched.log.save(OUT / f"seeds/dreamlog_{round_id:03d}.jsonl")
            print(f"[round {round_id}] {seed_rec}", flush=True)
        except Exception:
            err_streak += 1
            tb = traceback.format_exc()
            (OUT / f"seeds/round_{round_id:03d}_ERROR.json").write_text(tb, encoding="utf-8")
            print(f"[round {round_id}] 异常已入种子库（连续 {err_streak}）", flush=True)
            if err_streak >= 3:
                print("== END（连续异常安全停止）==")
                break
        else:
            err_streak = 0


if __name__ == "__main__":
    main()
