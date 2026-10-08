"""★ checkpoint.py 断点存续（阶段 1 工程前置，Spot 实例必需）。

保存/恢复：模型、优化器、循环位置、全部随机态（torch/python/环境 env.rng）、
情景缓冲内容。原子写（tmp+replace）防半截档。
信任边界：仅加载自己写的档（weights_only=False 用于 RNG 元组）。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(path, *, model, optimizer=None, cycle: int = 0,
                    extra: Optional[dict] = None, env_rng=None, buffer=None) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "cycle": int(cycle),
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
        "env_rng": env_rng.getstate() if env_rng is not None else None,
        "buffer": ([dict(ep) for ep in buffer.episodes] if buffer is not None else None),
        "extra": dict(extra or {}),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    torch.save(state, tmp)
    Path(tmp).replace(p)


def load_checkpoint(path, *, model, optimizer=None, env_rng=None, buffer=None):
    """恢复到保存时状态。返回 (cycle, extra)。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if env_rng is not None and state["env_rng"] is not None:
        env_rng.setstate(state["env_rng"])
    if buffer is not None and state["buffer"] is not None:
        buffer.episodes = state["buffer"]
        if len(buffer.episodes) > buffer.capacity:
            buffer.episodes = buffer.episodes[-buffer.capacity:]
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["py_rng"])
    return state["cycle"], state["extra"]
