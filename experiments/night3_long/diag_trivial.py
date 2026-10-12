#!/usr/bin/env python
"""平凡化/尺度检核（harvest 用）：JEPA MSE 是"移动标尺"？

背景（夜三 harvest 发现）：编码器目标尺度随训练漂移（k0/dh64 收敛后目标
逐维 var=0.229 vs k4/dh1024 的 0.020），跨预算/跨宽度的原始 MSE 不可直比。
本诊断给出尺度不变的量纲：
  - R²（vs predict-mean，模型自选目标空间内）
  - 解码 R²（obs 空间，外标尺；并含"拷贝上一帧"参照）
  - rank90（目标有效秩）
判读：解码 R² 高且优于拷贝参照 → 状态真承载世界信息。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkpoint import load_checkpoint      # noqa: E402
from src.core import WorldModel                 # noqa: E402
from src.dream import EpisodeBuffer             # noqa: E402
from src.envs import random_rollout             # noqa: E402
from src.envs.rich_world import RichWorld       # noqa: E402
from src.substrate import LatentState           # noqa: E402


def diag_from_model(model, k, batch=32, t_win=8):
    """与 run_long eval 同协议（seed 900 env / Random(4242) + torch 4242）。"""
    env = RichWorld(complexity=k, seed=0)
    ev = RichWorld(complexity=k, seed=900)
    buf = EpisodeBuffer(capacity=99)
    for _ in range(10):
        buf.add_episode(*random_rollout(ev))
    d_h = model.substrate.d_h
    model.eval()
    torch.manual_seed(4242)
    b = buf.sample_windows(batch, t_win, rng=random.Random(4242))
    with torch.no_grad():
        tgts = torch.stack([model.embed(b["obs_next"][:, t])
                            for t in range(t_win)], dim=0)      # (T,B,dh)
        tflat = tgts.reshape(-1, d_h)
        tvar_dim = float(tflat.var(dim=0).mean())
        mean_mse = float(((tflat - tflat.mean(0)) ** 2).mean())
        c = tflat - tflat.mean(0)
        sv2 = torch.linalg.svdvals(c) ** 2
        energy = sv2.cumsum(0) / sv2.sum()
        rank90 = int((energy < 0.9).sum()) + 1
        h = model.substrate.spawn(b["obs"].shape[0]).h
        err = dec = 0.0
        for t in range(t_win):
            x = model._x(b["obs"][:, t], b["act"][:, t])
            s, a, dy, di = model.taiji.mixed_step_parts(LatentState(h), x)
            h = s.h
            err += float(((h - tgts[t]) ** 2).mean())
            dec += float(((model.decode_obs(h) - b["obs_next"][:, t]) ** 2).mean())
        mse, decmse = err / t_win, dec / t_win
        o = b["obs_next"]
        ovar = float(o.reshape(-1, env.d_obs).var(dim=0).mean())
        copy_mse = float(((b["obs"] - o) ** 2).mean())   # 拷贝上一帧参照
    r2 = 100 * (1 - mse / mean_mse) if mean_mse > 0 else float("nan")
    dec_r2 = 100 * (1 - decmse / ovar) if ovar > 0 else float("nan")
    rec = {"k": k, "d_h": d_h, "mse": round(mse, 6),
           "target_var_dim": round(tvar_dim, 6), "predict_mean_mse": round(mean_mse, 6),
           "r2_pct": round(r2, 2), "rank90": rank90,
           "dec_mse": round(decmse, 6), "obs_var": round(ovar, 6),
           "dec_r2_pct": round(dec_r2, 2), "copy_mse": round(copy_mse, 6)}
    print(f"k{k} dh{d_h:4d}: mse={mse:.5f} | 目标var/维={tvar_dim:.4f} rank90={rank90} | "
          f"R²={r2:.1f}% | dec={decmse:.4f}(obs var={ovar:.3f} 拷贝参照={copy_mse:.4f}) "
          f"解码R²={dec_r2:.1f}%", flush=True)
    return rec


def diag(k, d_h, ckpt):
    env = RichWorld(complexity=k, seed=0)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=32, norm_kind="ln_post")
    cyc, _ = load_checkpoint(ckpt, model=model)
    print(f"[ckpt {Path(ckpt).name} cycle={cyc}]", flush=True)
    return diag_from_model(model, k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--d-h", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    diag(args.k, args.d_h, args.ckpt)
