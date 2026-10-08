#!/usr/bin/env python
"""P0-2 修正验证小对照（分析性检查，非正式 run，不进 registry）。

历史：v1 检查（graph 单步落卦打分）❌ 不通过——graph 0.17 << free_match 0.85，
     卦价值直接做得分项会误导（随机/引导混合策略污染转移统计）。
v2（本版）：改用 plan_shaped（电位整形 λ(γV(s′)−V(s)) 引导 random-shooting，
     λ=0 精确退化为 free），A1 对照变为 shaped vs free。
判定（预注册于本文件）：
    通过 ⇔ k64-shaped ≥ k64-free_match 且 > 0（两个 seed 平均）
    通过 → E2 全量用 shaped 作"图搜索"实现（graph 保留为失败记录）；
    不通过 → E2 正式跑照常执行但预记 A1 实现效力警告，不阻塞 E0/E1。

用法：.venv/bin/python experiments/e2_hexagram_codebook/p0_check.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.codebook import HexagramCodebook                # noqa: E402
from src.core import WorldModel                          # noqa: E402
from src.dream import EpisodeBuffer                      # noqa: E402
from src.envs import LatentGrid, guided_rollout, random_rollout  # noqa: E402

CFG = dict(cycles=8, episodes=8, d_h=48, wake_batches=10, t_win=8,
           eval_eps=20, lr=3e-3, grid=6, horizon=40, buffer_cap=500,
           free_k=48, free_horizon=6, gamma=0.9, vi_depth=10, guided_frac=0.3,
           lam_graph=1.0)


def make_env(seed):
    return LatentGrid(grid=CFG["grid"], slip=0.05, p_threat_move=0.0,
                      horizon=CFG["horizon"], seed=seed, with_hazard=False,
                      fixed_start=(0, 0), fixed_goal=(CFG["grid"] - 1, CFG["grid"] - 1))


def one_run(k, planner, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(seed)
    cb = HexagramCodebook(CFG["d_h"], k=k) if k > 0 else None
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=CFG["d_h"],
                       codebook=cb)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    buf = EpisodeBuffer(capacity=CFG["buffer_cap"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_e2 import run_planned_episodes
    for cycle in range(CFG["cycles"]):
        for _ in range(CFG["episodes"]):
            roll = guided_rollout if env.rng.random() < CFG["guided_frac"] else random_rollout
            buf.add_episode(*roll(env))
        for i in range(CFG["wake_batches"]):
            b = buf.sample_windows(16, CFG["t_win"],
                                   rng=random.Random(seed * 1000 + cycle * 100 + i),
                                   only_last=CFG["episodes"])
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
    succ, ret, steps = run_planned_episodes(model, planner, cb, CFG, seed + 999)
    return succ, ret, steps


def main():
    configs = [("k0-free", 0, "free"), ("k64-shaped", 64, "shaped"),
               ("k64-free", 64, "free"), ("k64-free_match", 64, "free_match"),
               ("k64-graph", 64, "graph")]
    results = {}
    for name, k, planner in configs:
        vals = [one_run(k, planner, s) for s in (0, 1)]
        results[name] = sum(v[0] for v in vals) / len(vals)
        print(f"{name:16s} succ={results[name]:.3f} "
              f"(seeds: {[round(v[0],2) for v in vals]})", flush=True)
    shaped, match = results["k64-shaped"], results["k64-free_match"]
    ok = shaped >= match and shaped > 0
    print(f"\n判定：shaped({shaped:.2f}) ≥ free_match({match:.2f}) 且 >0 → "
          f"{'✅ 通过，E2 全量用 shaped 作图搜索实现' if ok else '❌ 不通过（A1 预记实现效力警告，不阻塞 E0/E1）'}")
    out = Path(__file__).parent / "p0_check_result.json"
    out.write_text(json.dumps({"v2_results": results, "pass": ok,
                               "v1_record": "graph 单步落卦打分 0.17 vs free_match 0.85，不通过，已弃用"},
                              ensure_ascii=False, indent=2))
    print(f"已写 {out.name}")


if __name__ == "__main__":
    main()
