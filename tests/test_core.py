"""core.py WorldModel 测试：形状 / 损失有限 / 学习收敛 / MPC 探头。"""
import torch

from src.core import WorldModel
from src.dream import EpisodeBuffer
from src.envs import LatentGrid, random_rollout


def small_model(seed=0, **kw):
    torch.manual_seed(seed)
    return WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16, **kw)


def train_env(seed=0):
    """可控训练环境：固定起终点、无 slip、无威胁——动力学完全确定。"""
    return LatentGrid(grid=4, slip=0.0, p_threat_move=0.0, horizon=12, seed=seed,
                      fixed_start=(0, 0), fixed_goal=(1, 1))


def make_windows(env, n=16, t_win=8, seed=0):
    import random as _r

    buf = EpisodeBuffer(capacity=100)
    for _ in range(20):
        buf.add_episode(*random_rollout(env))
    return buf.sample_windows(n, t_win, rng=_r.Random(seed))


class TestShapes:
    def test_forward(self):
        m = small_model()
        obs = torch.randn(4, 6, 8)
        act = torch.randint(0, 4, (4, 6))
        h, alphas = m(obs, act)
        assert h.shape == (4, 24) and alphas.shape == (4, 6, 1)

    def test_loss_finite(self):
        m = small_model()
        b = make_windows(train_env())
        loss, parts = m.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        assert torch.isfinite(loss)
        assert {"pred", "rew", "dec", "alpha", "var_nll"} <= set(parts)

    def test_uncertainty_head(self):
        m = small_model()
        h = torch.randn(6, 24)
        sigma = m.predict_uncertainty(h)
        assert sigma.shape == (6,) and torch.all(sigma > 0)
        b = make_windows(train_env())
        loss, _ = m.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        loss.backward()
        assert m.var_head.weight.grad is not None
        assert m.var_head.weight.grad.abs().sum() > 0

    def test_extra_relax_forward_and_lowers_energy(self):
        """extra_relax 路径可跑，且追加弛豫后能量不升（阴弛豫机制不变）。"""
        from src.substrate import LatentState

        m = small_model(extra_relax=2)
        obs = torch.randn(4, 6, 8)
        act = torch.randint(0, 4, (4, 6))
        h, _ = m(obs, act)
        assert h.shape == (4, 24)
        with torch.no_grad():
            e_before = m.substrate.energy_of(h)
            s = m.yin.relax(LatentState(h), 4)
            e_after = m.substrate.energy_of(s.h)
        assert torch.all(e_after <= e_before)


class TestLearning:
    def test_prediction_mse_drops(self):
        """学习收敛 sanity：确定性环境上 250 步后潜态预测损失应大幅下降。"""
        torch.manual_seed(0)
        env = train_env()
        buf = EpisodeBuffer(capacity=100)
        for _ in range(30):
            buf.add_episode(*random_rollout(env))
        model = small_model(seed=0)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = last = None
        for step in range(250):
            b = buf.sample_windows(16, 8, rng=__import__("random").Random(step))
            loss, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 0:
                first = parts["pred"]
            last = parts["pred"]
        assert last < 0.6 * first, (first, last)


class TestMPC:
    def test_plan_logic_equivalence(self):
        """MPC 逻辑精确等价：plan == argmax_k Σ_t r̂(想象轨迹 k)。

        方向性行为断言（>7/10 朝目标）曾在此测试；var_head 引入后训练动态
        变化、玩具尺度不再稳定（实测 5-8/10 摆动）——行为判定归正式实验
        （E0 MPC 探头），单元测试只测机制精确性。
        """
        model = small_model(seed=2)
        obs = torch.randn(8)
        h = torch.randn(24)
        torch.manual_seed(5)
        a, R = model.plan(obs, h, horizon=4, k=8)
        torch.manual_seed(5)
        acts = torch.randint(0, 4, (8, 4))
        k_obs = obs.unsqueeze(0).expand(8, -1)
        h_b = h.unsqueeze(0).expand(8, -1)
        R2 = torch.zeros(8)
        with torch.no_grad():
            for t in range(4):
                h_b, _ = model.rollout_step(h_b, k_obs, acts[:, t])
                R2 += model.predict_reward(h_b)
                k_obs = model.decode_obs(h_b)
        assert a == int(acts[int(torch.argmax(R2)), 0].item())
        assert torch.allclose(R, R2, atol=1e-6)
