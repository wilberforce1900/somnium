#!/usr/bin/env python
"""夜三长训（NIGHT3-PLAN §E）：健康宽模型（ln_post）在 ENV-D k 档的长程运行。

设计：复用 R5-cal v2.2 训练环（30% 引导局 + 10 批/cycle wake 更新），
去掉 plateau 早停——观察长程漂移 / 遗忘 / 门极性演化；每 EVAL_EVERY cycle
评估 + 机制面板（α/hN/阴阳步幅/饱和）；每 SAVE_EVERY cycle 原子 checkpoint
（保留最近 KEEP_LAST + 每 MILESTONE 个永久保留）；每评估追加一行 jsonl
（崩溃安全，duck 可与 --resume 配合续跑）。

与短程校准协议的差异（记录在案）：本脚本启用 grad clip 1.0
（夜二教训——长训发散保险；短程 120cyc 校准未裁剪）。

用法：python run_long.py --complexity 4 --d-h 1024 --norm-kind ln_post
输出：OUT/long_k{K}_dh{H}_{norm}_s{SEED}.jsonl + .final.json；OUT/ckpts/
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.core import WorldModel                     # noqa: E402
from src.dream import EpisodeBuffer                 # noqa: E402
from src.envs import guided_rollout, random_rollout  # noqa: E402
from src.envs.rich_world import RichWorld           # noqa: E402
from src.substrate import LatentState               # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/night3_long"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "ckpts"
CKPT.mkdir(parents=True, exist_ok=True)

KEEP_LAST = 5      # checkpoint 保留：最近 N 个
MILESTONE = 500    # + 每 N cycle 里程碑永久保留


def train_long(k, d_h, seed, device, norm_kind, max_cycles, resume_path=None,
               batch=32, t_win=8, lr=1e-3, hidden=32, ep_per_cycle=8,
               wake_batches=10, eval_every=5, save_every=100, clip=1.0):
    torch.manual_seed(seed)
    random.seed(seed)
    env = RichWorld(complexity=k, seed=seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=d_h,
                       hidden=hidden, norm_kind=norm_kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ev = RichWorld(complexity=k, seed=seed + 900)
    eval_buf = EpisodeBuffer(capacity=99)
    for _ in range(10):
        eval_buf.add_episode(*random_rollout(ev))
    sq = math.sqrt(d_h)
    tag = f"long_k{k}_dh{d_h}_{norm_kind}_s{seed}"
    jsonl = OUT / f"{tag}.jsonl"

    def eval_and_probe(cycle, env_steps):
        torch.manual_seed(4242)
        model.eval()
        b = eval_buf.sample_windows(batch, t_win, rng=random.Random(4242))
        h = model.substrate.spawn(b["obs"].shape[0]).h
        err = 0.0
        alphas, hns, dys, dis, sats = [], [], [], [], []
        with torch.no_grad():
            for t in range(t_win):
                x = model._x(b["obs"][:, t], b["act"][:, t])
                s, a, dh_y, dh_i = model.taiji.mixed_step_parts(
                    LatentState(h), x)
                h = s.h
                tgt = model.embed(b["obs_next"][:, t])
                err += float(((h - tgt) ** 2).mean())
                z = model.substrate.cell_preact(h, x)
                sats.append(float((z.abs() > 2.0).float().mean()))
                alphas.append(float(a.mean()))
                hns.append(float(h.norm(dim=-1).mean()) / sq)
                dys.append(float(dh_y.norm(dim=-1).mean()) / sq)
                dis.append(float(dh_i.norm(dim=-1).mean()) / sq)
        model.train()
        rec = {"cycle": cycle, "env_steps": env_steps,
               "mse": round(err / t_win, 5),
               "alpha": round(sum(alphas) / len(alphas), 5),
               "h_norm": round(sum(hns) / len(hns), 5),
               "dy": round(sum(dys) / len(dys), 5),
               "di": round(sum(dis) / len(dis), 5),
               "sat": round(sum(sats) / len(sats), 5)}
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def retain():
        cks = sorted(CKPT.glob(f"{tag}_c*.pt"),
                     key=lambda p: int(p.stem.split("_c")[-1]))
        for p in cks[:-KEEP_LAST]:
            if int(p.stem.split("_c")[-1]) % MILESTONE != 0:
                p.unlink()

    env_steps = 0
    start_cycle = 0  # 已完成的最大 cycle（fresh=0；resume=断点记录值）
    if resume_path:
        start_cycle, extra = load_checkpoint(
            resume_path, model=model, optimizer=opt, env_rng=env.rng,
            buffer=buffer)
        env_steps = int(extra.get("env_steps", 0))
        print(f"[resume] {tag} 从 cycle {start_cycle + 1} 续（steps={env_steps}）",
              flush=True)

    last = None
    for cycle in range(start_cycle + 1, max_cycles + 1):
        for _ in range(ep_per_cycle):
            roll = guided_rollout if env.rng.random() < 0.3 else random_rollout
            buffer.add_episode(*roll(env))
            env_steps += buffer.episodes[-1]["act"].shape[0]
        for i in range(wake_batches):
            b = buffer.sample_windows(batch, t_win,
                                      rng=random.Random(seed * 1000 + cycle * 100 + i),
                                      only_last=8)
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if cycle == 1 or cycle % eval_every == 0:
            last = eval_and_probe(cycle, env_steps)
            print(f"[{tag}] cyc{cycle}/{max_cycles} mse={last['mse']} "
                  f"a={last['alpha']} hN={last['h_norm']} sat={last['sat']} "
                  f"steps={env_steps}", flush=True)
        if save_every and cycle % save_every == 0:
            save_checkpoint(CKPT / f"{tag}_c{cycle}.pt", model=model, optimizer=opt,
                            cycle=cycle, extra={"env_steps": env_steps},
                            env_rng=env.rng, buffer=buffer)
            retain()
    if last is not None:
        with open(OUT / f"{tag}.final.json", "w") as f:
            json.dump({"k": k, "d_h": d_h, "seed": seed, "norm_kind": norm_kind,
                       "max_cycles": max_cycles, "last": last}, f, indent=1)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", type=int, required=True)
    ap.add_argument("--d-h", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--norm-kind",
                    choices=["none", "ln_post", "rms_post", "ln_pre"],
                    default="ln_post")
    ap.add_argument("--max-cycles", type=int, default=9000)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", default=None, help="断点档（崩溃续跑）")
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    rec = train_long(args.complexity, args.d_h, args.seed, dev, args.norm_kind,
                     args.max_cycles, resume_path=args.resume,
                     eval_every=args.eval_every, save_every=args.save_every)
    print("LONG LAST:", json.dumps(rec))


if __name__ == "__main__":
    main()
