#!/usr/bin/env python
"""ITER13 · R4：潜态 vs 同参数量自回归基线（"思"层首次外部对照）。

预登记：ITER13-PLAN.md §1（全锁；含烟测修正——主判世界改为 k4）。
- **锁步**：同一脚本内 Somnium（d_h=64，wake-only）与 AR 因果 transformer 基线
  每 cycle 消费**同一批次张量**——真实交互流位流相同。
- **两世界**：k4=RichWorld complexity 4（run_long 冻结配方镜像：guided 0.3 引导 +
  eval=seed+900 随机局 10 条/batch 32）——**主判**（拷贝基线 ~30%，判别带宽）；
  rho0=NonStatWorld ρ0（iter9 家族配方，一致性行；烟测示拷贝基线 92% 压缩带）。
- **参数匹配**：AR dm=30 对 Somnium d_h=64：rho0（d_obs8）差 −0.12%、k4（d_obs31）
  差 −4.6%——均在 ±10% 内（rec 记录实测值）。
- **读数**：每 100 cycle 固定 eval 集测两模型同口径**解码 R²**（教师强制输入
  rollout，预测 o_{t+1}，demean obs 空间；拷贝参照同报）；eval 包 RNG 存档/恢复。
- **门（k4、双 seed、2400cyc 终值）**：Δ=dec_r²(Som)−dec_r²(AR)：≥+5pp 潜态优势 /
  ≤−5pp AR 优势（诚实负结果）/ 之间=平手→R4c k7 escalation。
反幻觉：架构对照性质，不判梦增益。

用法：python experiments/r4_baseline/run_r4.py --seed 0 --world k4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import WorldModel                                      # noqa: E402
from src.dream import EpisodeBuffer                                  # noqa: E402
from src.envs import guided_rollout, random_rollout                  # noqa: E402
from src.envs.nonstat_world import NonStatWorld                      # noqa: E402
from src.envs.rich_world import RichWorld                            # noqa: E402
from src.substrate import LatentState                                # noqa: E402
from experiments.env_e_nonstat.pilot_terminal import _rollout_toward  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE


class _ARBlock(nn.Module):
    """pre-LN 因果 transformer 块（手搭以精确控制参数量）。"""

    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(),
                                nn.Linear(ffn_dim, d_model))

    def forward(self, x, mask):
        h = self.ln1(x)
        att, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + att
        return x + self.ff(self.ln2(x))


class ARTransformer(nn.Module):
    """因果自回归基线：输入 (o_t, a_t) 序列 → 预测 o_{t+1}（与 Somnium 步同接口）。"""

    def __init__(self, d_obs=8, n_actions=4, ctx=8, d_model=30, n_layers=2,
                 n_heads=2, ffn_dim=60):
        super().__init__()
        self.d_obs = d_obs
        self.ctx = ctx
        self.obs_enc = nn.Linear(d_obs, d_model)
        self.act_enc = nn.Linear(n_actions, d_model)
        self.pos = nn.Parameter(torch.zeros(ctx, d_model))
        self.blocks = nn.ModuleList(
            [_ARBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_obs)

    def forward(self, obs, act):
        T = obs.shape[1]
        a = F.one_hot(act.long(), self.act_enc.in_features).to(obs.dtype)
        x = self.obs_enc(obs) + self.act_enc(a) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, device=obs.device), diagonal=1).bool()
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x))


def make_env(world, seed):
    if world == "k4":
        return RichWorld(complexity=4, seed=seed)
    return NonStatWorld(grid=8, slip=0.1, lam=0.0, horizon=50, seed=seed,
                        regime=0)


def collect_episode(env, world):
    """单局采集：k4 镜像 run_long（guided 0.3）；rho0 镜像 iter9（fracA.3/fracB.15）。"""
    r = env.rng.random()
    if world == "k4":
        roll = guided_rollout if r < 0.3 else random_rollout
        return roll(env)
    if r < 0.3:
        return _rollout_toward(env, (2, 3))
    if r < 0.45:
        return _rollout_toward(env, (4, 5))
    return random_rollout(env)


def build_eval_buffer(world, seed):
    """固定 eval 集：k4=seed+900 随机局 10 条（run_long 镜像）；rho0=seed+950 混合 12。"""
    ev = make_env(world, seed + 900) if world == "k4" \
        else make_env(world, seed + 950)
    ebuf = EpisodeBuffer(capacity=99)
    n = 10 if world == "k4" else 12
    for i in range(n):
        if world != "k4" and i % 3 != 0:
            ebuf.add_episode(*_rollout_toward(ev, (2, 3) if i % 3 == 1 else (4, 5)))
        else:
            ebuf.add_episode(*random_rollout(ev))
    return ebuf


def eval_nextobs(som, ar, ebuf, device, world, batch=32, t_win=8, seed=4242):
    """同口径解码 R²（教师强制输入 rollout，预测 o_{t+1}）。RNG 存档/恢复。"""
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    b = ebuf.sample_windows(batch, t_win, rng=random.Random(seed))
    o = b["obs"].to(device)
    a = b["act"].to(device)
    on = b["obs_next"].to(device)
    ovar = float(on.reshape(-1, on.shape[-1]).var(dim=0).mean())
    copy_mse = float(((o - on) ** 2).mean())
    som.eval()
    ar.eval()
    with torch.no_grad():
        h = som.substrate.spawn(o.shape[0]).h
        dec_som = 0.0
        for t in range(t_win):
            x = som._x(o[:, t], a[:, t])
            s, _, _, _ = som.taiji.mixed_step_parts(LatentState(h), x)
            h = s.h
            dec_som += float(((som.decode_obs(h) - on[:, t]) ** 2).mean())
        dec_ar = float(((ar(o, a) - on) ** 2).mean())
    som.train()
    ar.train()
    torch.set_rng_state(state)
    if ovar <= 0:
        nan = float("nan")
        return {"dec_r2_som": nan, "dec_r2_ar": nan, "copy_r2": nan,
                "dec_mse_som": nan, "dec_mse_ar": nan, "obs_var": ovar}
    dec_som /= t_win
    return {"dec_r2_som": round(100 * (1 - dec_som / ovar), 2),
            "dec_r2_ar": round(100 * (1 - dec_ar / ovar), 2),
            "copy_r2": round(100 * (1 - copy_mse / ovar), 2),
            "dec_mse_som": round(dec_som, 5),
            "dec_mse_ar": round(dec_ar, 5),
            "obs_var": round(ovar, 5)}


def run_r4(seed, device, world="k4", cycles=2400, eval_every=100, batch=32,
           t_win=8, lr=1e-3, hidden=32, ep_per_cycle=8, wake_batches=10,
           clip=1.0, tag_suffix=""):
    torch.manual_seed(seed)
    random.seed(seed)
    env = make_env(world, seed)
    som = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=64,
                     hidden=hidden, norm_kind="ln_post").to(device)
    ar = ARTransformer(d_obs=env.d_obs, n_actions=env.n_actions).to(device)
    som_opt = torch.optim.Adam(som.parameters(), lr=lr)
    ar_opt = torch.optim.Adam(ar.parameters(), lr=lr)
    buffer = EpisodeBuffer(capacity=2000)
    ebuf = build_eval_buffer(world, seed)

    p_som = sum(p.numel() for p in som.parameters())
    p_ar = sum(p.numel() for p in ar.parameters())
    gap = 100 * (p_ar - p_som) / p_som

    rows = []
    for cycle in range(1, cycles + 1):
        for _ in range(ep_per_cycle):
            buffer.add_episode(*collect_episode(env, world))
        for i in range(wake_batches):
            b = buffer.sample_windows_mixed(
                batch, t_win,
                rng=random.Random(seed * 1000 + cycle * 100 + i),
                only_last=ep_per_cycle, recent_frac=0.5)
            loss_s, _ = som.loss_on_batch(b["obs"], b["act"], b["obs_next"],
                                          b["rew"])
            som_opt.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(som.parameters(), clip)
            som_opt.step()
            pred = ar(b["obs"].to(device), b["act"].to(device))
            loss_a = F.mse_loss(pred, b["obs_next"].to(device))
            ar_opt.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), clip)
            ar_opt.step()
        if cycle == 1 or cycle % eval_every == 0:
            r = eval_nextobs(som, ar, ebuf, device, world)
            rows.append({"cycle": cycle, **r})
            if cycle % 200 == 0 or cycle == cycles:
                print(f"[r4 {world} s{seed}] c{cycle}/{cycles} "
                      f"decR2 som={r['dec_r2_som']} ar={r['dec_r2_ar']} "
                      f"(copy={r['copy_r2']})", flush=True)

    def half(key):
        rs = [x[key] for x in rows if x["cycle"] <= cycles // 2]
        return rs[-1] if rs else None

    end = rows[-1]
    rec = {"seed": seed, "world": world, "d_h": 64, "cycles": cycles,
           "eval_every": eval_every,
           "params_som": p_som, "params_ar": p_ar,
           "param_gap_pct": round(gap, 2),
           "dec_r2_som_end": end["dec_r2_som"],
           "dec_r2_ar_end": end["dec_r2_ar"],
           "delta_end": round(end["dec_r2_som"] - end["dec_r2_ar"], 2),
           "dec_r2_som_half": half("dec_r2_som"),
           "dec_r2_ar_half": half("dec_r2_ar"),
           "copy_r2_end": end["copy_r2"],
           "obs_var_end": end["obs_var"]}
    name = f"r4_{world}_s{seed}{tag_suffix}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"{name}.final.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"R4 {world} s{seed}: {json.dumps(rec)}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", choices=["rho0", "k4"], default="k4")
    ap.add_argument("--cycles", type=int, default=2400)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device(a.device) if a.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    run_r4(a.seed, dev, world=a.world, cycles=a.cycles,
           eval_every=a.eval_every, tag_suffix=a.tag_suffix)


if __name__ == "__main__":
    main()
