#!/usr/bin/env python
"""Somnium MCP server · stdio JSON-RPC（协议 2025-06-18）。

把已发表的 Somnium 循环世界模型（论文 v1.5，Zenodo 10.5281/zenodo.23479114）
包装成本地 agent 可拉起的工具服务器：观测 → O(1) 潜态 → 解码/想象/梦巩固。

形态与纪律（与 Financial Watch 的 propose-only MCP 同一族）：
  - 零第三方依赖（仅 torch + 仓库 src/），stdio ndjson JSON-RPC；
  - 只读世界 + 本地计算：无网络、无文件写出、无密钥；
  - somnium_dream 会更新服务器内存中的模型权重（梦后模型≠梦前，如实披露），
    可用 persist=false 做只读试梦。

启动（任意装有 torch≥2.0 的 python）：
    python mcp/server.py
    # checkpoint 默认取 mcp/somnium-canonical-dh64.pt，可用环境变量 SOMNIUM_CKPT 覆盖
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core import WorldModel  # noqa: E402
from src.dream import DreamConfig, DreamScheduler, EpisodeBuffer  # noqa: E402
from src.envs.latent_grid import LatentGrid, random_rollout  # noqa: E402

CKPT = Path(os.environ.get("SOMNIUM_CKPT", str(REPO / "mcp" / "somnium-canonical-dh64.pt")))
MAX_IMAGINE_STEPS = 32
MAX_EPISODES = 8

_MODEL = None
_META = {}


def load_model():
    global _MODEL, _META
    if _MODEL is not None:
        return _MODEL, _META
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    mc = state["extra"]["model_config"]
    m = WorldModel(d_obs=mc["d_obs"], n_actions=mc["n_actions"],
                   d_h=mc["d_h"], norm_kind=mc.get("norm_kind", "none"))
    m.load_state_dict(state["model"])
    m.eval()
    _MODEL = m
    _META = {
        "checkpoint": CKPT.name,
        "d_obs": mc["d_obs"], "d_h": mc["d_h"], "n_actions": mc["n_actions"],
        "norm_kind": mc.get("norm_kind", "none"),
        "params": state["extra"].get("params"),
        "trained_cycles": state.get("cycle"),
        "recipe": state["extra"].get("recipe"),
        "final_eval": state["extra"].get("final"),
        "provenance": state["extra"].get("provenance"),
    }
    return _MODEL, _META


def _vec(x, d_obs, name):
    """校验并转换一条观测为 float 张量。"""
    try:
        v = [float(t) for t in x]
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是 {d_obs} 个数字")
    if len(v) != d_obs:
        raise ValueError(f"{name} 长度必须为 {d_obs}（收到 {len(v)}）")
    return torch.tensor(v, dtype=torch.float32)


def _seq(x, d_obs, name, min_len=2, max_len=256):
    if not isinstance(x, list) or len(x) < min_len:
        raise ValueError(f"{name} 必须是 ≥{min_len} 条观测的数组")
    if len(x) > max_len:
        raise ValueError(f"{name} 最长 {max_len} 条")
    return torch.stack([_vec(o, d_obs, f"{name}[{i}]") for i, o in enumerate(x)])


def tool_info(args=None):
    m, meta = load_model()
    return meta


def tool_observe(args):
    """喂一段真实观测序列（teacher-forced），报告下一步预测质量与终态潜态摘要。"""
    m, meta = load_model()
    d_obs = meta["d_obs"]
    seq = _seq(args.get("seq"), d_obs, "seq", min_len=2)
    T = seq.shape[0]
    acts = args.get("actions")
    if acts is None:
        g = torch.Generator().manual_seed(0)
        act = torch.randint(0, meta["n_actions"], (T - 1,), generator=g)
    else:
        if not isinstance(acts, list) or len(acts) != T - 1:
            raise ValueError(f"actions 长度必须为 {T - 1}")
        act = torch.tensor([int(a) for a in acts], dtype=torch.long)
        if (act < 0).any() or (act >= meta["n_actions"]).any():
            raise ValueError(f"actions 取值必须在 [0,{meta['n_actions'] - 1}]")
    with torch.no_grad():
        h = m.ground(seq[0])
        som_sq, copy_sq = 0.0, 0.0
        for t in range(T - 1):
            h, _ = m.rollout_step(h.unsqueeze(0), seq[t].unsqueeze(0), act[t:t + 1])
            h = h[0]
            pred = m.decode_obs(h)
            tgt = seq[t + 1]
            som_sq += ((pred - tgt) ** 2).sum().item()
            copy_sq += ((seq[t] - tgt) ** 2).sum().item()
        som_mse = som_sq / ((T - 1) * d_obs)
        copy_mse = copy_sq / ((T - 1) * d_obs)
        r2 = 1.0 - som_mse / copy_mse if copy_mse > 1e-12 else None
        rew = m.predict_reward(h).item()
        unc = m.predict_uncertainty(h).item()
        hnorm = h.norm().item()
    return {"steps": T - 1, "somnium_mse": round(som_mse, 6),
            "copy_frame_mse": round(copy_mse, 6),
            "r2_vs_copy": None if r2 is None else round(r2, 4),
            "final_latent": {"h_norm": round(hnorm, 4),
                             "predicted_reward": round(rew, 4),
                             "uncertainty_sigma": round(unc, 4)},
            "note": "r2_vs_copy>0 表示优于复制上一帧的平凡基线（论文 §5.4 协议）"}


def tool_imagine(args):
    """闭环想象：从 ground(obs) 起步，自身解码回流，滚动 k 步并逐步解码。"""
    m, meta = load_model()
    d_obs = meta["d_obs"]
    o0 = _vec(args.get("obs"), d_obs, "obs")
    steps = int(args.get("steps", 8))
    if not 1 <= steps <= MAX_IMAGINE_STEPS:
        raise ValueError(f"steps 必须在 [1,{MAX_IMAGINE_STEPS}]")
    seed = int(args.get("seed", 0))
    acts = args.get("actions")
    act_list = None
    if acts is not None:
        if not isinstance(acts, list) or len(acts) < steps:
            raise ValueError(f"actions 长度必须 ≥ steps")
        act_list = [int(a) for a in acts[:steps]]
        if any(a < 0 or a >= meta["n_actions"] for a in act_list):
            raise ValueError(f"actions 取值必须在 [0,{meta['n_actions'] - 1}]")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    traj, uncs = [], []
    with torch.no_grad():
        h = m.ground(o0)
        for t in range(steps):
            a = act_list[t] if act_list is not None else rng.randrange(meta["n_actions"])
            o_t = m.decode_obs(h)            # 闭环：自身解码回流（R4b 协议）
            h, _ = m.rollout_step(h.unsqueeze(0), o_t.unsqueeze(0),
                                  torch.tensor([a], dtype=torch.long))
            h = h[0]
            o_next = m.decode_obs(h)
            traj.append([round(float(v), 4) for v in o_next])
            uncs.append(round(m.predict_uncertainty(h).item(), 4))
    return {"steps": steps, "imagined_obs": traj, "uncertainty_sigma": uncs,
            "note": "闭环想象=模型吃自己的解码（无真值锚定）；误差随 steps 复合"
                    "（论文 §5.4：本尺度 AR 基线开环更强，Somnium 残迹=复合更慢）"}


def tool_episode(args):
    """生成一条真实 grid8 世界情节（训练同分布），供 observe/dream 直接使用。"""
    _, meta = load_model()
    steps = min(max(int(args.get("steps", 20)), 4), 200)
    seed = int(args.get("seed", 42))
    slip = float(args.get("slip", 0.1))
    if not 0.0 <= slip <= 0.9:
        raise ValueError("slip 必须在 [0, 0.9]")
    env = LatentGrid(grid=8, slip=slip, p_threat_move=0.02, horizon=steps, seed=seed)
    obs, act, rew, _ = random_rollout(env)
    return {"steps": int(act.shape[0]), "slip": slip,
            "obs": [[round(float(v), 4) for v in row] for row in obs],
            "actions": [int(a) for a in act],
            "note": "随机策略情节；slip=动作被替换为随机动作的概率（0.1=易/0.45=难 regime）"}


def tool_dream(args):
    """对提交的情节做只巩固梦期（imagination 关，consolidation 开）。

    默认 persist=true：梦会更新服务器内存里的模型（下次 observe/imagine 用梦后权重）。
    persist=false 为只读试梦（测完即回滚到 checkpoint 权重）。
    """
    m, meta = load_model()
    d_obs = meta["d_obs"]
    eps = args.get("episodes")
    if not isinstance(eps, list) or not 1 <= len(eps) <= MAX_EPISODES:
        raise ValueError(f"episodes 必须是 1–{MAX_EPISODES} 条情节")
    batches = min(max(int(args.get("batches", 10)), 1), 50)
    seed = int(args.get("seed", 0))
    persist = bool(args.get("persist", True))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    buf = EpisodeBuffer(capacity=9999)
    min_len = 10 ** 9
    for i, ep in enumerate(eps):
        obs = _seq(ep, d_obs, f"episodes[{i}]", min_len=4, max_len=256)
        min_len = min(min_len, obs.shape[0])
        act = torch.randint(0, meta["n_actions"], (obs.shape[0] - 1,), generator=torch.Generator().manual_seed(seed + i))
        rew = torch.zeros(obs.shape[0] - 1)
        rare = torch.zeros(obs.shape[0] - 1, dtype=torch.bool)
        buf.add_episode(obs, act, rew, rare)
    t_win = max(2, min(8, min_len - 1))  # 窗口按最短情节自适应
    # 梦前/梦后用同一固定窗口评估（只巩固损失）
    ev_rng = random.Random(4242)
    b = buf.sample_windows(16, t_win, rng=ev_rng)
    with torch.no_grad():
        _, parts_before = m.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
    before = parts_before["pred"]
    sched = DreamScheduler(m, buf, DreamConfig(batch=16, t_win=t_win,
                                               imagination=False, consolidation=True,
                                               reverse_learning=False, rehearsal=False,
                                               read_replay=True),
                           rng=rng)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)  # 梦是真学习：权重必须实际更新
    for _ in range(batches):
        sched.dream_phase(opt, batches_per_component=1)
    m.eval()
    with torch.no_grad():
        _, parts_after = m.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
    after = parts_after["pred"]
    if not persist:
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(state["model"])
    return {"batches": batches, "consolidation_loss_before": round(before, 6),
            "consolidation_loss_after": round(after, 6),
            "delta": round(after - before, 6),
            "model_persisted": persist,
            "note": "delta<0=梦期巩固降低了所提交情节上的预测损失；"
                    "persist=true 时模型权重已在内存中更新（梦后模型≠梦前）"}


TOOLS = [
    {"name": "somnium_info",
     "description": "模型卡：已发表的 Somnium 循环世界模型（16K 参数档）的维度、配方与训练终评。",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}}},
    {"name": "somnium_observe",
     "description": "喂一段真实观测序列（每条 8 维浮点向量），报告 O(1) 潜态的下一步预测 "
                    "MSE/R²（对拷贝上一帧的平凡基线）与终态潜态摘要（奖励预测/不确定度）。",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
         "seq": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                 "description": "观测序列，≥2 条，每条 8 维"},
         "actions": {"type": "array", "items": {"type": "integer"},
                     "description": "可选：每步动作 0–3（缺省种子化随机）"}},
         "required": ["seq"]}},
    {"name": "somnium_imagine",
     "description": "闭环想象：从给定观测接地起步，模型吃自己的解码滚动 k 步，逐步返回想象的观测"
                    "与不确定度（梦/白日梦的内容生成器）。",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
         "obs": {"type": "array", "items": {"type": "number"}, "description": "8 维起始观测"},
         "steps": {"type": "integer", "minimum": 1, "maximum": 32},
         "seed": {"type": "integer", "description": "缺省 0；同种子同输出"},
         "actions": {"type": "array", "items": {"type": "integer"},
                     "description": "可选：显式动作序列（长度 ≥ steps）"}},
         "required": ["obs", "steps"]}},
    {"name": "somnium_episode",
     "description": "生成一条真实 grid8 世界情节（与训练同分布的随机策略 rollout），"
                    "返回 obs 序列与动作，可直接喂给 somnium_observe / somnium_dream。",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
         "steps": {"type": "integer", "minimum": 4, "maximum": 200, "description": "缺省 20"},
         "seed": {"type": "integer", "description": "缺省 42；同种子同情节"},
         "slip": {"type": "number", "minimum": 0, "maximum": 0.9,
                  "description": "动作随机替换率：0.1=易 regime / 0.45=难 regime"}}}},
    {"name": "somnium_dream",
     "description": "对提交的情节（1–8 条观测序列）做只巩固梦期并返回梦前/梦后损失。"
                    "默认会更新服务器内存中的模型权重（梦后模型≠梦前）；persist=false 只读试梦。",
     "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
         "episodes": {"type": "array", "items": {"type": "array", "items": {"type": "array",
                     "items": {"type": "number"}}}, "description": "情节列表，每条为 ≥4 步的观测序列"},
         "batches": {"type": "integer", "minimum": 1, "maximum": 50, "description": "梦期批数，缺省 10"},
         "seed": {"type": "integer"},
         "persist": {"type": "boolean", "description": "缺省 true=梦后权重保留在内存"}},
         "required": ["episodes"]}},
]


def dispatch(msg):
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    try:
        if method == "initialize":
            return ok({"protocolVersion": params.get("protocolVersion") or "2025-06-18",
                       "capabilities": {"tools": {}},
                       "serverInfo": {"name": "somnium-mcp", "version": "1.0.0",
                                      "title": "Somnium (recurrent world model)"}})
        if method in ("notifications/initialized", "notifications/cancelled"):
            return None
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            fns = {"somnium_info": tool_info, "somnium_observe": tool_observe,
                   "somnium_imagine": tool_imagine, "somnium_dream": tool_dream,
                   "somnium_episode": tool_episode}
            if name not in fns:
                return err(-32602, f"unknown tool: {name}")
            try:
                out = fns[name](args)
            except ValueError as e:
                return ok({"content": [{"type": "text", "text": f"参数错误: {e}"}],
                           "isError": True})
            return ok({"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})
        if mid is not None:
            return err(-32601, "Method not found: " + str(method))
        return None
    except Exception as e:  # 服务器级兜底：绝不让一行坏输入杀死 stdio 循环
        if mid is None:
            return None
        return err(-32603, f"internal error: {type(e).__name__}: {e}"[:200])


def main():
    load_model()  # 启动即加载：坏 checkpoint 尽早暴露
    sys.stderr.write(f"somnium-mcp ready (stdio; ckpt={CKPT.name}, "
                     f"params={_META.get('params')})\n")
    for line in iter(sys.stdin.readline, ""):  # readline 循环：管道下禁用迭代预读（防死锁）
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = dispatch(msg)
        if out is not None:
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
