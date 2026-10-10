"""★ dream.py 梦期调度器（必需）—— PRINCIPLES §4 / ROADMAP E0、E3。

四开关（独立启停 = A4 消融的实验面）：
    imagination      内生想象（视网膜波式）：spawn 潜态 + 伪观测 rollout，
                     优化潜态方差下限（反塌缩）+ 能量平滑（Lipschitz-lite）。
                     不碰真实数据、不需要 buffer —— E0-b"先梦后醒"的引擎。
    consolidation    巩固：情景缓冲回放蒸馏（episodic→parametric，标准 replay）。
    reverse_learning 反向学习（Crick-Mitchison "梦以遗忘" v0）：噪声增广回放，
                     输入加噪、目标干净 = 去噪化，剪除对噪声的伪敏感。
    rehearsal        排练（Revonsuo 威胁模拟 v0）：稀有事件(rare)过采样回放。

DreamLog：梦 log（能量/α/潜态方差/各开关损失）——"梦报表"与 A5 读梦的数据源。
    ◇ 卦码字段占位（codebook 接入后填）。

公平性注记：梦期是额外离线计算。E0 的比较基准 = 真实环境交互步数，不是算力对齐。

◇ 可扩展：EMA 目标、能量自适应反向学习强度、actor-critic 想象（Dreamer 式）
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .core import WorldModel


@dataclass
class DreamConfig:
    """梦期四开关 + 超参。全部可独立消融（A4）。"""

    imagination: bool = True
    consolidation: bool = True
    reverse_learning: bool = True
    rehearsal: bool = True
    noise: float = 0.1          # 反向学习：输入噪声幅度
    rare_bias: float = 3.0      # 排练：rare 转移采样权重倍率
    # 注（迭代#2 实测）：spawn 潜态方差 ≈ σ²=1.0 > div_target=0.3，此地板在
    # 全部正式 run 中恒为松弛（从未激活）——想象收益实际全部来自能量平滑项
    # （更贴近 Crick-Mitchison/Tononi 的"睡眠调节能量地形"理论）。若将来把
    # div_target 提到激活区间，机制改变，需重跑 E0/E3 再下结论。
    div_target: float = 0.3     # 想象：潜态方差下限（反塌缩；见上注：正式 run 中未激活）
    lam_smooth: float = 0.1     # 想象：能量平滑权重
    pseudo_scale: float = 1.0   # 想象：伪观测尺度
    batch: int = 16
    t_win: int = 8
    imagine_len: int = 8
    read_replay: bool = False  # 读梦 v2：巩固回放按覆盖逆权重偏置（盲区过采样）


class EpisodeBuffer:
    """情景缓冲：整条 episode 存储，窗口采样。"""

    def __init__(self, capacity: int = 500) -> None:
        self.capacity = capacity
        self.episodes: List[Dict[str, torch.Tensor]] = []

    def add_episode(self, obs: torch.Tensor, act: torch.Tensor,
                    rew: torch.Tensor, rare: torch.Tensor) -> None:
        """obs: (T+1, d_obs)；act: (T,) long；rew: (T,)；rare: (T,) bool。"""
        self.episodes.append({"obs": obs, "act": act, "rew": rew, "rare": rare})
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_windows(
        self,
        n: int,
        t_win: int,
        rare_bias: float = 0.0,
        rng: Optional[random.Random] = None,
        only_last: Optional[int] = None,
        weight_fn=None,
    ) -> Dict[str, torch.Tensor]:
        """采样 n 个长 t_win 的窗口 → dict(obs, act, obs_next, rew, rare_frac)。

        rare_bias > 0：按 episode 稀有率加权（排练）。only_last：仅最新 k 条
        episode（醒期新鲜数据路径，保证各日程真实数据消耗一致）。
        weight_fn：外部基础权重函数 ep→w（读梦 v2 的盲区逆权重）。
        """
        rng = rng or random.Random()
        eps = self.episodes[-only_last:] if only_last else self.episodes
        if not eps:
            raise IndexError("buffer 为空")
        # 只从足够长的 episode 采定长窗口，避免长短混叠（早终局的短局被跳过）
        long_eps = [ep for ep in eps if ep["act"].shape[0] >= t_win]
        if long_eps:
            eps, w = long_eps, t_win
        else:
            w = min(t_win, max(ep["act"].shape[0] for ep in eps))
            eps = [ep for ep in eps if ep["act"].shape[0] >= w]
        if rare_bias > 0:
            weights = [1.0 + rare_bias * float(ep["rare"].float().mean()) for ep in eps]
        else:
            weights = [1.0] * len(eps)
        if weight_fn is not None:
            weights = [wt * weight_fn(ep) for wt, ep in zip(weights, eps)]
        if all(wt == weights[0] for wt in weights):
            weights = None
        idxs = rng.choices(range(len(eps)), weights=weights, k=n)
        obs_l, act_l, obs_n_l, rew_l = [], [], [], []
        rare_frac = 0.0
        for i in idxs:
            ep = eps[i]
            T = ep["act"].shape[0]
            s = rng.randrange(T - w + 1)
            obs_l.append(ep["obs"][s: s + w])
            act_l.append(ep["act"][s: s + w])
            obs_n_l.append(ep["obs"][s + 1: s + 1 + w])
            rew_l.append(ep["rew"][s: s + w])
            rare_frac += float(ep["rare"][s: s + w].float().mean())
        return {
            "obs": torch.stack(obs_l),
            "act": torch.stack(act_l),
            "obs_next": torch.stack(obs_n_l),
            "rew": torch.stack(rew_l),
            "rare_frac": rare_frac / n,
        }


class ObsCoverage:
    """obs 空间覆盖统计（读梦 A5 重设版：不依赖已降级的卦码）。

    - 维护 coverage_dims 两维的 G×G 访问直方图（醒期数据驱动）+ 全维边际统计；
    - 盲区 = 低访问格位；pseudo_obs 合成"未梦之梦"：盲区格位 + 其余维度按
      醒期边际分布采样。
    - blind=False 时退化为均匀格位（对照臂用：同分布、无读梦偏置）。
    """

    def __init__(self, d_obs: int, dims=(0, 1), bins: int = 6,
                 lo: float = 0.0, hi: float = 1.0) -> None:
        self.d_obs = d_obs
        self.dims = tuple(dims)
        self.bins = bins
        self.lo, self.hi = lo, hi
        self.blind = True  # 读梦开关：True=盲区偏置，False=均匀格位（对照臂）
        self.counts = torch.zeros(bins, bins)
        self.n = 0
        self.mean = torch.zeros(d_obs)
        self.m2 = torch.zeros(d_obs)
        self.std = torch.ones(d_obs)

    def update(self, obs: torch.Tensor) -> None:
        """obs: (..., d_obs)——醒期数据喂入，更新直方图与边际统计。"""
        x = obs.reshape(-1, self.d_obs).detach()
        self.n += x.shape[0]
        delta = x - self.mean
        self.mean = self.mean + delta.sum(0) / self.n
        self.m2 = self.m2 + ((x - self.mean) ** 2).sum(0)
        self.std = (self.m2 / max(self.n - 1, 1)).sqrt().clamp(min=1e-3)
        a = ((x[:, self.dims[0]] - self.lo) / (self.hi - self.lo) * self.bins) \
            .long().clamp(0, self.bins - 1)
        b = ((x[:, self.dims[1]] - self.lo) / (self.hi - self.lo) * self.bins) \
            .long().clamp(0, self.bins - 1)
        flat = a * self.bins + b
        self.counts.view(-1).scatter_add_(
            0, flat, torch.ones_like(flat, dtype=self.counts.dtype))

    def _cells(self, n: int, blind: bool):
        w = (1.0 / (1.0 + self.counts)).flatten() if blind \
            else torch.ones(self.bins * self.bins)
        idx = torch.multinomial(w, n, replacement=True)
        a, b = idx // self.bins, idx % self.bins
        ca = (a.float() + 0.5) / self.bins * (self.hi - self.lo) + self.lo
        cb = (b.float() + 0.5) / self.bins * (self.hi - self.lo) + self.lo
        return ca, cb

    def pseudo_obs(self, n: int, blind: Optional[bool] = None) -> torch.Tensor:
        """合成伪观测：格位（盲区或均匀）+ 其余维度 ~ 醒期边际。

        blind 缺省取实例开关 self.blind（读梦臂 True / 对照臂 False）。
        """
        use_blind = self.blind if blind is None else blind
        ca, cb = self._cells(n, use_blind)
        out = self.mean + self.std * torch.randn(n, self.d_obs)
        out[:, self.dims[0]] = ca
        out[:, self.dims[1]] = cb
        return out

    def covered_frac(self) -> float:
        return float((self.counts > 0).float().mean())

    def episode_weight(self, obs: torch.Tensor) -> float:
        """episode 的读梦权重 = 其各时刻所在格位访问数的逆均值（盲区高权）。"""
        x = obs.reshape(-1, self.d_obs).detach()
        a = ((x[:, self.dims[0]] - self.lo) / (self.hi - self.lo) * self.bins) \
            .long().clamp(0, self.bins - 1)
        b = ((x[:, self.dims[1]] - self.lo) / (self.hi - self.lo) * self.bins) \
            .long().clamp(0, self.bins - 1)
        return float((1.0 / (1.0 + self.counts[a, b])).mean())


class DreamLog:
    """梦 log：一行一梦期。未来"读梦"（A5）从此取数。"""

    def __init__(self) -> None:
        self.entries: List[Dict] = []

    def append(self, entry: Dict) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def last(self) -> Dict:
        return self.entries[-1] if self.entries else {}

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.entries.clear()


class DreamScheduler:
    """梦期调度器：四开关组件 + 醒期统一入口。"""

    _NEED_BUFFER = ("consolidation", "reverse_learning", "rehearsal")

    def __init__(
        self,
        model: WorldModel,
        buffer: EpisodeBuffer,
        cfg: Optional[DreamConfig] = None,
        log: Optional[DreamLog] = None,
        rng: Optional[random.Random] = None,
        coverage: Optional[ObsCoverage] = None,
    ) -> None:
        self.model = model
        self.buffer = buffer
        self.cfg = cfg or DreamConfig()
        self.log = log or DreamLog()
        self.rng = rng or random.Random()
        self.coverage = coverage  # 读梦反馈环：醒期喂覆盖 → 想象偏置盲区
        self.t = 0

    # ---- 组件：内生想象（不需要 buffer） ----
    def imagine_loss(self):
        """spawn 潜态 + 伪观测随机动作 rollout：反塌缩 + 能量平滑。

        读梦开启（self.coverage 给定）时，伪观测 = 覆盖器合成的盲区样本
        （多梦未梦之梦）；否则 N(0,1)·scale。
        """
        c, m = self.cfg, self.model
        B, L = c.batch, c.imagine_len
        h = m.substrate.spawn(B).h
        hs = []
        alpha_sum = 0.0
        for t in range(L):
            pseudo = (self.coverage.pseudo_obs(B).to(h.device) if self.coverage is not None
                      else torch.randn(B, m.d_obs, device=h.device) * c.pseudo_scale)
            acts = torch.randint(0, m.n_actions, (B,))
            h, a = m.rollout_step(h, pseudo, acts)
            hs.append(h)
            alpha_sum += float(a.mean().detach())
        h_last = hs[-1]
        div = torch.relu(c.div_target - h_last.var(dim=0)).mean()
        eps = 0.1 * torch.randn_like(h_last)
        smooth = (
            (m.substrate.energy_of(h_last + eps) - m.substrate.energy_of(h_last)) ** 2
        ).mean()
        loss = div + c.lam_smooth * smooth
        stats = {
            "div": float(div.detach()),
            "smooth": float(smooth.detach()),
            "latent_std": float(h_last.detach().std()),
            "alpha": alpha_sum / L,
        }
        return loss, stats

    def _replay_loss(self, noise: float = 0.0, rare_bias: float = 0.0):
        # 读梦 v2：巩固回放（noise=0）按覆盖逆权重偏置——盲区真实转移过采样。
        # v1（伪观测盲区偏置）已被证空转：imagine 目标内容不敏感，记录在案。
        wf = None
        if (noise == 0.0 and getattr(self.cfg, "read_replay", False)
                and self.coverage is not None):
            wf = lambda ep: self.coverage.episode_weight(ep["obs"])
        b = self.buffer.sample_windows(
            self.cfg.batch, self.cfg.t_win, rare_bias=rare_bias, rng=self.rng,
            weight_fn=wf,
        )
        loss, parts = self.model.loss_on_batch(
            b["obs"], b["act"], b["obs_next"], b["rew"], input_noise=noise
        )
        return loss, parts, b

    # ---- 梦期主入口 ----
    def dream_phase(self, optimizer, batches_per_component: int = 1,
                    clip: float = 0.0) -> Dict:
        """执行一轮梦期：每个启用组件 backward+step。返回摘要并写梦 log。
        clip>0 时对梯度做范数裁剪（夜二教训：长训练发散，v2.2 加）。"""
        import torch.nn.utils as tnu
        summary: Dict[str, Dict] = {}
        enabled = [
            name
            for name in ("imagination", "consolidation", "reverse_learning", "rehearsal")
            if getattr(self.cfg, name)
        ]
        for name in enabled:
            if name in self._NEED_BUFFER and len(self.buffer) == 0:
                summary[name] = {"skipped": "empty_buffer"}
                continue
            for _ in range(batches_per_component):
                if name == "imagination":
                    loss, st = self.imagine_loss()
                elif name == "consolidation":
                    loss, st, _ = self._replay_loss()
                elif name == "reverse_learning":
                    loss, st, _ = self._replay_loss(noise=self.cfg.noise)
                else:  # rehearsal
                    loss, st, b = self._replay_loss(rare_bias=self.cfg.rare_bias)
                    st = dict(st, rare_frac=b["rare_frac"])
                optimizer.zero_grad()
                loss.backward()
                if clip > 0:
                    tnu.clip_grad_norm_(self.model.parameters(), clip)
                optimizer.step()
                summary[name] = st
        with torch.no_grad():
            probe = self.model.substrate.spawn(64)
            energy = float(self.model.substrate.energy_of(probe.h).mean())
        self.log.append(
            {
                "t": self.t,
                "components": summary,
                "energy_probe": energy,
                "coverage": (self.coverage.covered_frac()
                             if self.coverage is not None else None),
                "hexagram": None,  # ◇ L3 codebook 接入后填
            }
        )
        self.t += 1
        return summary

    # ---- 醒期统一入口（各日程共用，保证真实数据消耗一致） ----
    def wake_update(self, optimizer, batch: Dict[str, torch.Tensor],
                    clip: float = 0.0) -> Dict:
        loss, parts = self.model.loss_on_batch(
            batch["obs"], batch["act"], batch["obs_next"], batch["rew"]
        )
        optimizer.zero_grad()
        loss.backward()
        if clip > 0:
            import torch.nn.utils as tnu
            tnu.clip_grad_norm_(self.model.parameters(), clip)
        optimizer.step()
        if self.coverage is not None:
            self.coverage.update(batch["obs"])  # 读梦：醒期喂覆盖统计
        return parts
