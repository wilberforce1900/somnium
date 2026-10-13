"""k5/6 扩展与 ENV-E v2 世界测试：有序目标/巡逻 hazard/regime 角色翻转/
全局 regime 持续性/λ 统计/确定性。"""
import random

import torch

from src.envs import NonStatWorld, RichWorld, random_rollout


class TestRichWorldK5:
    def test_k5_shape_and_params(self):
        e = RichWorld(complexity=5, seed=0)
        assert e.G == 12 and e.H == 260 and e.n_goals == 4 and e.n_keys == 3
        assert e.ordered_goals and not e.patrol_hazard
        assert e.d_obs == 2 + 25 + 2 + 2  # 位置2+视野25+进度2+时钟常量2
        o = e.reset()
        assert o.shape == (e.d_obs,)

    def test_k5_ordered_goals_enforced(self):
        e = RichWorld(complexity=5, seed=3)
        e.reset()
        # 跳序：站在第 2 个目标上走一步（第 1 个未收）——不登记
        g1 = e.goals_pos[1]
        e._agent = (g1[0] - 1, g1[1]) if g1[0] > 0 else (g1[0] + 1, g1[1])
        e.step(1 if g1[0] > e._agent[0] else 3)
        assert e._goals_done == set() or e.goals_pos[0] in e._goals_done or \
            e._agent != g1, "（几何兜底：若被墙挡则跳过本断言）"
        # 顺序：从第 1 个目标相邻格走上它——登记
        g0 = e.goals_pos[0]
        done = False
        for (dx, dy, a) in ((1, 0, 1), (-1, 0, 3), (0, 1, 0), (0, -1, 2)):
            e._agent = (g0[0] - dx, g0[1] - dy)
            before = len(e._goals_done)
            _, _, _, info = e.step(a)
            if e._agent == g0:
                assert len(e._goals_done) == before + 1, "按序收集第 1 个应登记"
                assert info["progress"] == 1
                done = True
                break
        assert done, "四个方向至少一个可达第 1 目标"

    def test_k5_deterministic_and_rollout(self):
        e1, e2 = RichWorld(complexity=5, seed=7), RichWorld(complexity=5, seed=7)
        o1, o2 = e1.reset(), e2.reset()
        assert torch.equal(o1, o2)
        obs, act, rew, rare = random_rollout(RichWorld(complexity=5, seed=1))
        assert obs.shape[1] == RichWorld(complexity=5, seed=1).d_obs
        assert act.shape[0] > 0


class TestRichWorldK6:
    def test_k6_patrol_hazard_approaches(self):
        e = RichWorld(complexity=6, seed=0)
        e.reset()
        d0 = abs(e.hazard[0] - e._agent[0]) + abs(e.hazard[1] - e._agent[1])
        # 走 25 步（含 5 个巡逻 tick），agent 不动太多（贴墙走或随机）
        closer_seen = False
        for _ in range(25):
            e.step(random.randrange(4))
            d = abs(e.hazard[0] - e._agent[0]) + abs(e.hazard[1] - e._agent[1])
            if d < d0:
                closer_seen = True
                break
        assert closer_seen, "巡逻 hazard 必须在逼近 agent"
        assert e.patrol_hazard

    def test_k6_params(self):
        e = RichWorld(complexity=6, seed=0)
        assert e.G == 12 and e.H == 300 and e.patrol_hazard and e.ordered_goals

    def test_k56_rejected_k7(self):
        try:
            RichWorld(complexity=7)
        except ValueError:
            pass
        else:
            raise AssertionError("k=7 应拒绝")


class TestNonStatWorld:
    def test_role_flip_swaps_goal_hazard(self):
        e = NonStatWorld(grid=8, slip=0.0, lam=0.0, horizon=50, seed=5)
        e.reset()
        e._agent = (e._a[0] - 1, e._a[1]) if e._a[0] > 0 else (e._a[0] + 1, e._a[1])
        # 走向 A：ρ0 得 +1，ρ1 得 −1
        e.regime = 0
        _, r0, d0, _ = e.step(1 if e._a[0] > e._agent[0] else 3)
        assert r0 == 1.0 and d0
        e2 = NonStatWorld(grid=8, slip=0.0, lam=0.0, horizon=50, seed=5)
        e2.reset()
        e2._agent = (e2._a[0] - 1, e2._a[1]) if e2._a[0] > 0 else (e2._a[0] + 1, e2._a[1])
        e2.regime = 1
        _, r1, d1, _ = e2.step(1 if e2._a[0] > e2._agent[0] else 3)
        assert r1 == -1.0 and d1

    def test_regime_persists_across_reset(self):
        e = NonStatWorld(grid=8, lam=0.0, seed=1, regime=1)
        e.reset()
        e.reset()
        assert e.regime == 1, "regime 是全局持续过程，episode 边界不重抽"

    def test_lam_statistics(self):
        e = NonStatWorld(grid=8, lam=0.0, horizon=200, seed=2)
        for _ in range(3):
            obs, act, rew, rare = random_rollout(e)
        assert e.switches == 0, "lam=0 永不切换"
        e2 = NonStatWorld(grid=8, lam=1.0, horizon=50, seed=3)
        e2.reset()
        for _ in range(100):
            e2.step(random.randrange(4))
        assert e2.switches >= 90, "lam=1 每步必切"

    def test_obs_schema(self):
        e = NonStatWorld(grid=8, seed=4)
        o = e.reset()
        assert o.shape == (8,) and o.abs().max() <= 1.5

    def test_guided_rollout_runs(self):
        from src.envs import guided_rollout
        e = NonStatWorld(grid=6, slip=0.1, lam=0.01, horizon=20, seed=6)
        obs, act, rew, rare = guided_rollout(e)
        assert obs.shape == (act.shape[0] + 1, 8)
