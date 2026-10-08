"""★ planner.py 图上慢规划（必需）—— PRINCIPLES §5 / ROADMAP E2（A1）。

图规划（plan_graph）：先在卦图上价值迭代得 V（离线统计，零 rollout 开销），
再对每个候选动作想象一步 → 落卦 → 得分 = 即时奖励预测 + γ·V[卦]。
细粒度执行交回连续核心（一步想象走 WorldModel，长程走卦图）。

对照：
    free       自由潜空间 random-shooting（k×horizon 次 rollout）
    free_match 与图规划等 rollout 数的对照（k=n_actions, horizon=1）
"""
from __future__ import annotations

import torch

from .codebook import HexagramCodebook, value_iteration


def plan_graph(model, codebook: HexagramCodebook, obs: torch.Tensor,
               h: torch.Tensor, gamma: float = 0.9, depth: int = 3):
    """图上慢规划选动作。返回 (action int, 各动作得分 list)。"""
    V = value_iteration(codebook, gamma, depth)
    scores = []
    with torch.no_grad():
        for a in range(model.n_actions):
            at = torch.tensor([a], dtype=torch.long)
            h_a, _ = model.rollout_step(h.view(1, -1), obs.view(1, -1), at)
            idx = int(codebook.assign(h_a)[0])
            r_hat = float(model.predict_reward(h_a)[0])
            scores.append(r_hat + gamma * float(V[idx]))
    best = max(range(len(scores)), key=scores.__getitem__)
    return best, scores


def plan_free(model, obs, h, horizon: int = 6, k: int = 64):
    """自由潜空间 random-shooting（core.WorldModel.plan 的薄封装）。"""
    a, _ = model.plan(obs, h, horizon=horizon, k=k)
    return a


def plan_free_match(model, obs, h):
    """等 rollout 数对照：k=n_actions, horizon=1（与图规划同样只想象 n 步）。"""
    a, _ = model.plan(obs, h, horizon=1, k=model.n_actions)
    return a


def plan_shaped(model, codebook: HexagramCodebook, obs: torch.Tensor,
                h: torch.Tensor, horizon: int = 6, k: int = 48,
                gamma: float = 0.9, depth: int = 10, lam: float = 1.0):
    """图引导整形搜索（P0-2 修正实现，2026-10-08）。

    随机射击 + 卦价值电位整形：轨迹得分 = Σr̂ + λ·Σ(γV(s′)−V(s))。
    电位整形（Ng et al. 1999）在真 MDP 中不改变最优策略——卦图只引导
    搜索偏好，不替代连续核。λ=0 精确退化为 plan_free（A1 的干净对照）。
    返回 (action, R)。
    """
    V = value_iteration(codebook, gamma, depth)
    with torch.no_grad():
        k_obs = obs.unsqueeze(0).expand(k, -1)
        h_b = h.unsqueeze(0).expand(k, -1)
        acts = torch.randint(0, model.n_actions, (k, horizon))
        idx_prev = codebook.assign(h_b)
        R = torch.zeros(k)
        for t in range(horizon):
            h_b, _ = model.rollout_step(h_b, k_obs, acts[:, t])
            R += model.predict_reward(h_b)
            if lam > 0:
                idx_new = codebook.assign(h_b)
                R += lam * (gamma * V[idx_new] - V[idx_prev])
                idx_prev = idx_new
            k_obs = model.decode_obs(h_b)
        best = int(torch.argmax(R).item())
    return int(acts[best, 0].item()), R
