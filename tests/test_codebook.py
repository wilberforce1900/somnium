"""codebook/planner 测试：量化 / VQ 梯度 / 转移统计 / 价值迭代 / 先天卦序 / 图规划。"""
import torch

from src.codebook import HexagramCodebook, value_iteration
from src.core import WorldModel
from src.envs import LatentGrid, random_rollout
from src.dream import EpisodeBuffer
from src.planner import plan_free, plan_free_match, plan_graph


class TestQuantize:
    def test_assign_nearest(self):
        cb = HexagramCodebook(d_h=4, k=8)
        with torch.no_grad():
            cb.codes.zero_()
            cb.codes[3] = 10.0  # 远离原点的唯一码字
        h = torch.zeros(5, 4)
        h[2] = 8.0
        idx = cb.assign(h)
        assert idx[2].item() == 3 and idx[0].item() != 3

    def test_vq_loss_grads_both_sides(self):
        cb = HexagramCodebook(d_h=4, k=8)
        h = torch.randn(6, 4, requires_grad=True)
        loss, idx = cb.vq_loss(h)
        loss.backward()
        assert h.grad.abs().sum() > 0
        assert cb.codes.grad.abs().sum() > 0


class TestStats:
    def test_observe_transitions_and_rewards(self):
        cb = HexagramCodebook(d_h=4, k=8)
        idx_seq = torch.tensor([[0, 1, 1], [2, 2, 0]])
        rew = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        cb.observe(idx_seq, rew)
        assert cb.counts[0, 1].item() == 1 and cb.counts[1, 1].item() == 1
        assert cb.counts[2, 2].item() == 1 and cb.counts[2, 0].item() == 1
        assert cb.visits[0].item() == 2 and cb.visits[1].item() == 2 and cb.visits[2].item() == 2
        assert abs(cb.reward_sum[0].item() - 1.0) < 1e-6
        assert abs(cb.reward_sum[2].item() - 1.0) < 1e-6

    def test_transition_probs_rows(self):
        cb = HexagramCodebook(d_h=4, k=4)
        idx_seq = torch.tensor([[0, 1, 1]])
        cb.observe(idx_seq)
        T = cb.transition_probs()
        assert abs(T[0].sum().item() - 1.0) < 1e-6
        assert T[0, 1].item() == 1.0 and T[2].sum().item() == 0.0  # 未观测行不传播

    def test_mean_reward_unvisited_zero(self):
        cb = HexagramCodebook(d_h=4, k=4)
        idx_seq = torch.tensor([[0, 0]])
        rew = torch.tensor([[2.0, 4.0]])
        cb.observe(idx_seq, rew)
        mr = cb.mean_reward()
        assert abs(mr[0].item() - 3.0) < 1e-6 and mr[1].item() == 0.0


class TestValueIteration:
    def test_propagates_along_graph(self):
        """0 →(r=0)→ 1 →(r=0)→ 2(r=1)：V[0] 应因传播而 > 0。"""
        cb = HexagramCodebook(d_h=4, k=3)
        idx_seq = torch.tensor([[0, 1, 2]])
        rew = torch.tensor([[0.0, 0.0, 1.0]])
        cb.observe(idx_seq, rew)
        V = value_iteration(cb, gamma=0.9, depth=3)
        assert V[2].item() == 1.0 and V[1].item() > 0.8 and V[0].item() > 0.6


class TestHexagram:
    def test_xiantian_binary(self):
        cb = HexagramCodebook(d_h=4, k=64)
        assert cb.to_hexagram(0) == "000000"
        assert cb.to_hexagram(1) == "000001"  # 低位=初爻
        assert cb.to_hexagram(63) == "111111"
        cb32 = HexagramCodebook(d_h=4, k=32)
        assert cb32.to_hexagram(31) == "11111"  # 5-bit


class TestModelWithCodebook:
    def _setup(self, seed=0, k=16):
        torch.manual_seed(seed)
        env = LatentGrid(grid=4, slip=0.0, p_threat_move=0.0, horizon=12,
                         seed=seed, fixed_start=(0, 0), fixed_goal=(1, 1))
        cb = HexagramCodebook(d_h=24, k=k)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16, codebook=cb)
        buf = EpisodeBuffer(capacity=100)
        for _ in range(15):
            buf.add_episode(*random_rollout(env))
        return model, cb, buf

    def test_loss_includes_cb_and_observes(self):
        model, cb, buf = self._setup()
        model.train()
        b = buf.sample_windows(8, 6, rng=__import__("random").Random(0))
        before = cb.counts.sum().item()
        loss, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        assert "cb" in parts and torch.isfinite(loss)
        assert cb.counts.sum().item() > before  # 训练态自动观象
        loss.backward()
        assert cb.codes.grad.abs().sum() > 0

    def test_eval_does_not_pollute_stats(self):
        model, cb, buf = self._setup()
        model.eval()
        b = buf.sample_windows(8, 6, rng=__import__("random").Random(0))
        before = cb.counts.sum().item()
        with torch.no_grad():
            model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        assert cb.counts.sum().item() == before

    def test_commitment_pulls_latents(self):
        """训练几步后 h 到码字的距离应下降（VQ 收敛 sanity）。"""
        model, cb, buf = self._setup(seed=1, k=8)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        import random as _r

        def dist():
            b = buf.sample_windows(16, 6, rng=_r.Random(99))
            with torch.no_grad():
                h, _ = model(b["obs"], b["act"])
                d = (h - cb.codes[cb.assign(h)]).pow(2).mean()
            return float(d)

        d0 = dist()
        for i in range(150):
            b = buf.sample_windows(16, 6, rng=_r.Random(i))
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        assert dist() < d0, (d0, dist())


class TestPlanners:
    def _trained(self, seed=0):
        torch.manual_seed(seed)
        env = LatentGrid(grid=4, slip=0.0, p_threat_move=0.0, horizon=12,
                         seed=seed, fixed_start=(0, 0), fixed_goal=(1, 1))
        cb = HexagramCodebook(d_h=24, k=16)
        model = WorldModel(d_obs=8, n_actions=4, d_h=24, hidden=16, codebook=cb)
        buf = EpisodeBuffer(capacity=200)
        for _ in range(30):
            buf.add_episode(*random_rollout(env))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        import random as _r
        for i in range(250):
            b = buf.sample_windows(16, 6, rng=_r.Random(i))
            loss, _ = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        return model, cb, env

    def test_all_planners_return_valid_action(self):
        model, cb, env = self._trained()
        obs = env.reset()
        h = model.ground(obs)
        assert plan_graph(model, cb, obs, h)[0] in range(4)
        assert plan_free(model, obs, h, horizon=4, k=16) in range(4)
        assert plan_free_match(model, obs, h) in range(4)

    def test_graph_score_equivalence(self):
        """planner 逻辑精确等价：plan_graph == argmax_a (r̂(h_a) + γ·V[idx_a])。

        行为层面"图规划 vs 自由规划"的优劣是 E2 正式实验的预注册问题
        （ROADMAP §1.3 +5pp 阈值），不在单元测试里下结论。
        """
        model, cb, env = self._trained()
        obs = env.reset()
        h = model.ground(obs)
        a, scores = plan_graph(model, cb, obs, h, gamma=0.9, depth=3)
        V = value_iteration(cb, gamma=0.9, depth=3)
        manual = []
        with torch.no_grad():
            for act in range(4):
                at = torch.tensor([act], dtype=torch.long)
                h_a, _ = model.rollout_step(h.view(1, -1), obs.view(1, -1), at)
                idx = int(cb.assign(h_a)[0])
                manual.append(float(model.predict_reward(h_a)[0]) + 0.9 * float(V[idx]))
        assert a == max(range(4), key=manual.__getitem__)
        assert all(abs(s - m) < 1e-6 for s, m in zip(scores, manual))
