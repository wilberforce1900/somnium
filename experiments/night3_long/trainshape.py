#!/usr/bin/env python
"""协议桥（P4）：在 {120, 1000, 4000} 处停止 run_long 同环训练 → 平凡化/尺度诊断。

目的：120cyc 校准协议的 MSE 与 9000cyc 收敛协议的 MSE 因编码器尺度漂移不可
直比；本桥在多个预算点采样同一套尺度不变指标（R²/解码R²），给出跨预算曲线。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))

from src.core import WorldModel                      # noqa: E402
from src.dream import EpisodeBuffer                  # noqa: E402
from src.envs import guided_rollout, random_rollout  # noqa: E402
from src.envs.rich_world import RichWorld            # noqa: E402
from diag_trivial import diag_from_model             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", type=int, required=True)
    ap.add_argument("--d-h", type=int, required=True)
    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    env = RichWorld(complexity=a.complexity, seed=a.seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=a.d_h,
                       hidden=32, norm_kind="ln_post").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = EpisodeBuffer(capacity=2000)
    for cycle in range(1, a.cycles + 1):
        for _ in range(8):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            buffer.add_episode(*roll(env))
        for i in range(10):
            b = buffer.sample_windows(32, 8,
                                      rng=random.Random(a.seed * 1000 + cycle * 100 + i),
                                      only_last=8)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if cycle % 200 == 0 or cycle == a.cycles:
            print(f"[bridge k{a.complexity} dh{a.d_h}] cyc{cycle}/{a.cycles}",
                  flush=True)
    model.to("cpu")
    rec = diag_from_model(model, a.complexity)
    print(f"BRIDGE k{a.complexity} dh{a.d_h} c{a.cycles}:", rec)


if __name__ == "__main__":
    main()
