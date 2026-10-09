"""★ growth.py 梦长模型：函数保持的宽度扩展（R5 载具）——用户设计 2026-10-09。

"参数每随一次梦境加深扩大一次"：不预先定死容量，每次梦期加深后把潜态宽度
扩一档——梦养大梦者，醒着的世界喂养长大的身体。生长必须**函数保持**
（Net2Net 式），否则"生长"="重置"。

方案（修正版，2026-10-09 深夜——首版复制消费列会重复计入旧维贡献，已废弃）：
  - **生产者复制**：新潜态维的产出权重（encoder 末层行、cell 输出行）复制
    随机源维 + 仅新增行加噪声（打破对称）；
  - **消费者零列**：一切读取 h/enc 的权重（cell 输入、energy、门、各头）的
    新增列置零 → 新维是源维的"影子"，不改变任何输出；
  - noise=0 时严格逐位保持（影子与源永久同化）；noise>0 时偏差 O(ε)，
    消费者列开始收到梯度、影子逐渐"醒来"。

注意：生长后 Adam 矩与参数形状失配，须重建优化器（矩清零，文档化行为）。
◇ 可扩展：剪枝回缩（梦醒的"呼吸"）、growth 时 gate 重置联动（iter4 棘轮）。
"""
from __future__ import annotations

import random
from typing import Optional

import torch
from torch import nn

from .core import WorldModel


def _dup_index(old: int, new: int, rng: random.Random, device) -> torch.Tensor:
    """长度 new 的旧维索引表：前 old 个为自身，其余随机复制源。"""
    idx = list(range(old)) + [rng.randrange(old) for _ in range(new - old)]
    return torch.tensor(idx, device=device)


def widen_world_model(model: WorldModel, new_d_h: int,
                      rng: Optional[random.Random] = None,
                      noise: float = 0.0) -> None:
    """就地函数保持扩宽：d_h: m → new_d_h（生产者复制+消费者零列）。"""
    rng = rng or random.Random()
    m = model.substrate.d_h
    n = model.n_actions
    if new_d_h <= m:
        raise ValueError(f"new_d_h({new_d_h}) 必须大于当前({m})")
    dev = model.substrate.cell.weight.device
    idx = _dup_index(m, new_d_h, rng, dev)          # 新维 → 源维

    def copy_rows(lin: nn.Linear) -> None:           # 生产者：输出维复制+噪声
        w = lin.weight.data[idx, :].clone()
        if noise > 0:
            w[m:] += noise * torch.randn_like(w[m:])
        lin.weight = nn.Parameter(w)
        lin.bias = nn.Parameter(lin.bias.data[idx].clone())

    def zero_cols(lin: nn.Linear, layout: str) -> None:  # 消费者：新列置零
        w, b = lin.weight.data, lin.bias.data
        gap = new_d_h - m
        if layout == "h":
            w2 = torch.cat([w, torch.zeros(w.shape[0], gap, device=dev)], dim=1)
        elif layout == "h_x":                        # [h(m') | enc(m') | onehot]
            z = torch.zeros(w.shape[0], gap, device=dev)
            w2 = torch.cat([w[:, :m], z, w[:, m:2 * m], z, w[:, 2 * m:]], dim=1)
        else:
            raise ValueError(layout)
        lin.weight = nn.Parameter(w2)
        lin.bias = nn.Parameter(b.clone())

    # 生产者：enc 维与 h 维的产出（e0 兼消费者：行复制+列清零，形状两侧对齐）
    copy_rows(model.encoder[-1])
    copy_rows(model.substrate.cell)
    copy_rows(model.substrate.energy_head[0])
    # 消费者：一切读取 h / enc 的输入
    zero_cols(model.substrate.cell, "h_x")
    zero_cols(model.substrate.energy_head[0], "h")
    zero_cols(model.substrate.energy_head[2], "h")
    zero_cols(model.taiji.gate_net, "h_x")
    for head in (model.reward_head, model.obs_head, model.var_head):
        zero_cols(head, "h")
    model.substrate.d_h = new_d_h
    model.substrate.d_x = new_d_h + n
