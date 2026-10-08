"""★ core.py 世界模型组装层（必需）—— ROADMAP §1.0 模型底座。

组装：encoder（观测→潜输入）+ substrate（共享底座）+ taiji（阴阳门控核心）
     + reward/obs 两个接地头（奖励头必带；obs 头小权重，仅供想象闭环与可解释 ◇）。

JEPA 式训练目标（潜空间预测，不重建像素；PRINCIPLES §7）：
    L = MSE(h_{t+1}, sg·enc(o_{t+1})) + λr·MSE(r̂, r) + λd·MSE(ô, o_{t+1})
    每批从混沌出生（substrate.spawn）——序列起点即 L0。

MPC 探头：random-shooting；想象闭环用 obs 头回投（ô_{t+1} = obs_head(h_{t+1})）。

◇ 可扩展：EMA 目标编码器、actor-critic、卦码头（codebook 接入）、d_h 放大至 1–8M
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .substrate import LatentState, Substrate
from .yang import YangMode
from .yin import YinMode
from .taiji import TaijiGate


class WorldModel(nn.Module):
    """潜空间世界模型：底座之上的最小组装。"""

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        d_h: int = 64,
        hidden: int = 32,
        lam_rew: float = 1.0,
        lam_dec: float = 0.2,
        init_scale: float = 0.1,
        gate_mode: str = "learned",
        extra_relax: int = 0,
        codebook=None,
        lam_cb: float = 0.1,
        lam_var: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_obs = d_obs
        self.n_actions = n_actions
        self.lam_rew = lam_rew
        self.lam_dec = lam_dec
        # 递归弛豫深度（E1 深度-性能曲线）：每步混合更新后追加 n 次阴弛豫。
        # 纯阳对照（fixed_yang）应保持 0。
        self.extra_relax = int(extra_relax)
        # L3 态势码本（E2/A1）：传入 HexagramCodebook 即激活——训练时加 VQ
        # commitment 损失并观测转移统计；None 则模型与 L3 无耦合。
        self.codebook = codebook
        self.lam_cb = lam_cb
        self.lam_var = lam_var
        # 编码器直接映射到潜态空间（d_h）——JEPA 同空间预测，无需投影头
        self.encoder = nn.Sequential(
            nn.Linear(d_obs, hidden), nn.GELU(), nn.Linear(hidden, d_h)
        )
        self.substrate = Substrate(d_x=d_h + n_actions, d_h=d_h, init_scale=init_scale)
        self.yang = YangMode(self.substrate)
        self.yin = YinMode(self.substrate)
        self.taiji = TaijiGate(self.substrate, self.yang, self.yin, gate_mode=gate_mode)
        self.reward_head = nn.Linear(d_h, 1)
        self.obs_head = nn.Linear(d_h, d_obs)
        # L5 智层第一粒种子（E3b/A5 引入）：奖励预测的异方差不确定度 log σ²。
        # 校准（知不知）与选择性弃权（知止）的度量基础。
        self.var_head = nn.Linear(d_h, 1)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def ground(self, obs: torch.Tensor) -> torch.Tensor:
        """取象于当下：行动用的接地初始潜态（编码器直出）。

        训练序列从混沌出生（substrate.spawn，保多样性）；行动/规划时
        从当前观测接地初始化，避免随机潜态污染首步决策。
        """
        return self.encoder(obs)

    def _x(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = F.one_hot(action.long(), self.n_actions).to(obs.dtype)
        return torch.cat([self.embed(obs), a], dim=-1)

    def rollout_step(self, h: torch.Tensor, obs: torch.Tensor, action: torch.Tensor):
        """一步混合更新 (h_t, o_t, a_t) → h_{t+1}。action: long 张量。返回 (h_next, α)。
        extra_relax > 0 时在混合步后追加阴弛豫（E1 深度消融）。"""
        s, a = self.taiji.mixed_step(LatentState(h), self._x(obs, action))
        if self.extra_relax > 0:
            s = self.yin.relax(s, self.extra_relax)
        return s.h, a

    def predict_reward(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward_head(h).squeeze(-1)

    def predict_uncertainty(self, h: torch.Tensor) -> torch.Tensor:
        """奖励预测的 σ（L5 校准用，与 predict_reward 同一潜态）。"""
        return self.var_head(h).squeeze(-1).mul(0.5).exp()

    def decode_obs(self, h: torch.Tensor) -> torch.Tensor:
        return self.obs_head(h)

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor):
        """(B,T,d_obs),(B,T)long → (h_T (B,d_h), α (B,T,1))。"""
        B, T, _ = obs_seq.shape
        h = self.substrate.spawn(B).h
        alphas = []
        for t in range(T):
            h, a = self.rollout_step(h, obs_seq[:, t], act_seq[:, t])
            alphas.append(a)
        return h, torch.stack(alphas, dim=1)

    def loss_on_batch(
        self,
        obs_seq: torch.Tensor,
        act_seq: torch.Tensor,
        obs_next_seq: torch.Tensor,
        rew_seq: torch.Tensor,
        input_noise: float = 0.0,
    ):
        """潜态预测 + 奖励/obs 接地。

        input_noise > 0：输入加噪、目标保持干净——反向学习的去噪回放入口
        （dream.py 用）。返回 (total_loss, parts dict)。
        """
        B, T, _ = obs_seq.shape
        obs_in = (
            obs_seq + input_noise * torch.randn_like(obs_seq)
            if input_noise > 0
            else obs_seq
        )
        h = self.substrate.spawn(B).h
        l_pred = l_rew = l_dec = l_cb = l_var = 0.0
        alpha_sum = 0.0
        idxs = []
        hs = []
        for t in range(T):
            h, a = self.rollout_step(h, obs_in[:, t], act_seq[:, t])
            alpha_sum = alpha_sum + a.mean()
            with torch.no_grad():
                target = self.embed(obs_next_seq[:, t])
            l_pred = l_pred + F.mse_loss(h, target)
            r_hat = self.predict_reward(h)
            l_rew = l_rew + F.mse_loss(r_hat, rew_seq[:, t])
            # 异方差 NLL：0.5(log σ² + (r−μ)²/σ²)——σ 学"哪里不知道"
            logvar = self.var_head(h).squeeze(-1).clamp(-6.0, 4.0)
            l_var = l_var + 0.5 * (
                logvar + (rew_seq[:, t] - r_hat).detach() ** 2 / logvar.exp()
            ).mean()
            l_dec = l_dec + F.mse_loss(self.decode_obs(h), obs_next_seq[:, t])
            if self.codebook is not None:
                cb_t, idx_t = self.codebook.vq_loss(h)
                l_cb = l_cb + cb_t
                idxs.append(idx_t)
                hs.append(h)
        l_pred, l_rew, l_dec = l_pred / T, l_rew / T, l_dec / T
        l_var = l_var / T
        total = (l_pred + self.lam_rew * l_rew + self.lam_dec * l_dec
                 + self.lam_var * l_var)
        parts = {
            "pred": float(l_pred.detach()),
            "rew": float(l_rew.detach()),
            "dec": float(l_dec.detach()),
            "alpha": float(alpha_sum.detach() / T),
            "var_nll": float((l_var / T).detach()),
        }
        if self.codebook is not None:
            l_cb = l_cb / T
            total = total + self.lam_cb * l_cb
            parts["cb"] = float(l_cb.detach())
            if self.training and idxs:
                idx_seq = torch.stack(idxs, dim=1)          # (B,T)
                self.codebook.observe(idx_seq, rew_seq)     # 转移图 + 卦德统计
                self.codebook.reseed_dead(torch.stack(hs, dim=1).reshape(-1, self.substrate.d_h))
        return total, parts

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, h: torch.Tensor, horizon: int = 6, k: int = 64):
        """MPC random-shooting：返回 (最优首动作 int, 各轨迹预测回报 (k,))。"""
        k_obs = obs.unsqueeze(0).expand(k, -1)
        h_b = h.unsqueeze(0).expand(k, -1)
        acts = torch.randint(0, self.n_actions, (k, horizon))
        R = torch.zeros(k)
        for t in range(horizon):
            h_b, _ = self.rollout_step(h_b, k_obs, acts[:, t])
            R += self.predict_reward(h_b)
            k_obs = self.decode_obs(h_b)  # 想象闭环
        best = int(torch.argmax(R).item())
        return int(acts[best, 0].item()), R
