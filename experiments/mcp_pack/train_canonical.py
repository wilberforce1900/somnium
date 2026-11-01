#!/usr/bin/env python
"""规范 MCP 打包模型训练入口（E0 dream_first FULL 配方的无 MPC 变体）。

与 experiments/e0_dream_first/run_e0.py --full --schedule dream_first 逐字同配方
（grid8/d_h64/seed0/dream_pre 200/guided 0.3/recent 回放/lr 1e-3），仅两处工程差异：
  1) 去掉 MPC 回报探头（CPU 打包场景下 plan(k=96) 每 cycle 数十秒，非必需）；
  2) 评估追加 decode R²（对拷贝参照），checkpoint extra 内嵌全部重建所需配置。

产物：mcp/somnium-canonical-dh64.pt（自描述 checkpoint：extra.model_config 可独立重建）。
本入口是打包产物生产，不是预登记实验——不写 registry；配方与指标如实记录于 extra
与 mcp/MODEL-PROVENANCE.md。
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

from src.checkpoint import save_checkpoint  # noqa: E402
from src.core import WorldModel  # noqa: E402
from src.dream import DreamConfig, DreamLog, DreamScheduler, EpisodeBuffer  # noqa: E402
from src.envs.latent_grid import LatentGrid, guided_rollout, random_rollout  # noqa: E402
from src.substrate import default_device  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "mcp" / "somnium-canonical-dh64.pt"

# E0 FULL 配方（run_e0.py FULL dict），cycles 可覆盖
RECIPE = dict(cycles=300, episodes_per_cycle=10, grid=8, horizon=50,
              d_h=64, batch=32, t_win=8, wake_batches=10,
              dream_pre_batches=200, lr=1e-3, eval_episodes=20,
              buffer_cap=2000, guided_frac=0.3)


def eval_model(model, eval_buf, cfg):
    """固定 eval 集：潜态预测 MSE（E0 主指标）+ 解码 R²（对拷贝参照）。"""
    torch.manual_seed(4242)
    model.eval()
    rng = random.Random(4242)
    b = eval_buf.sample_windows(cfg["batch"], cfg["t_win"], rng=rng)
    with torch.no_grad():
        _, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        # 解码 R²：teacher-forced 末步潜态（forward 返回 h_T (B,d_h)）的 obs 重建 vs 拷贝参照
        h_T, _ = model.forward(b["obs"], b["act"])
        dec = model.decode_obs(h_T)
        tgt = b["obs"][:, -1, :]
        prev = b["obs"][:, -2, :]
        ss_res = ((dec - tgt) ** 2).sum().item()
        ss_tot = ((tgt - tgt.mean(dim=0, keepdim=True)) ** 2).sum().item()
        decode_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        copy_mse = ((prev - tgt) ** 2).mean().item()
    model.train()
    return {"pred_mse": round(parts["pred"], 6),
            "decode_r2": round(decode_r2, 4),
            "copy_mse": round(copy_mse, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=RECIPE["cycles"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--d-h", type=int, default=RECIPE["d_h"])
    ap.add_argument("--dream-pre", type=int, default=RECIPE["dream_pre_batches"])
    args = ap.parse_args()

    cfg = dict(RECIPE)
    cfg["cycles"] = args.cycles
    cfg["d_h"] = args.d_h
    cfg["dream_pre_batches"] = args.dream_pre
    cfg["seed"] = args.seed
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                     horizon=cfg["horizon"], seed=args.seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=cfg["d_h"],
                       norm_kind="none").to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = EpisodeBuffer(capacity=cfg["buffer_cap"])
    dream_log = DreamLog()
    sched = DreamScheduler(model, buffer, DreamConfig(batch=cfg["batch"], t_win=cfg["t_win"]),
                           log=dream_log, rng=random.Random(args.seed))

    ev_env = LatentGrid(grid=cfg["grid"], slip=0.1, p_threat_move=0.02,
                        horizon=cfg["horizon"], seed=args.seed + 999)
    eval_buf = EpisodeBuffer(capacity=9999)
    for _ in range(cfg["eval_episodes"]):
        eval_buf.add_episode(*random_rollout(ev_env))

    t0 = time.time()
    env_steps = 0
    history = []
    print(f"[train] params={n_params} cfg={cfg}", flush=True)

    # 梦期先行（E0-b：空缓冲 → 纯内生想象）
    for _ in range(cfg["dream_pre_batches"]):
        sched.dream_phase(opt)

    for cycle in range(cfg["cycles"]):
        for _ in range(cfg["episodes_per_cycle"]):
            roll = guided_rollout if env.rng.random() < cfg["guided_frac"] else random_rollout
            obs, act, rew, rare = roll(env)
            buffer.add_episode(obs, act, rew, rare)
            env_steps += act.shape[0]
        for i in range(cfg["wake_batches"]):
            b_rng = random.Random(args.seed * 1000 + cycle * 100 + i)
            b = buffer.sample_windows(cfg["batch"], cfg["t_win"], rng=b_rng,
                                      only_last=cfg["episodes_per_cycle"])
            sched.wake_update(opt, b)
        if (cycle + 1) % 25 == 0 or cycle == cfg["cycles"] - 1:
            m = eval_model(model, eval_buf, cfg)
            history.append({"cycle": cycle + 1, "env_steps": env_steps, **m})
            print(f"[cycle {cycle + 1}/{cfg['cycles']}] {m}", flush=True)

    model_config = {"d_obs": env.d_obs, "n_actions": env.n_actions,
                    "d_h": cfg["d_h"], "norm_kind": "none"}
    final = history[-1]
    save_checkpoint(args.out, model=model, cycle=cfg["cycles"],
                    extra={"model_config": model_config, "recipe": cfg,
                           "provenance": "e0 dream_first FULL (no-MPC packaging variant), "
                                         "experiments/mcp_pack/train_canonical.py",
                           "params": n_params, "eval_history": history,
                           "final": final, "env_steps": env_steps,
                           "trained_at": time.strftime("%Y-%m-%d")})
    print(f"[done] {time.time() - t0:.1f}s -> {args.out} final={final}", flush=True)


if __name__ == "__main__":
    main()
