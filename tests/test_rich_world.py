"""ENV-D rich_world 测试：确定性 / 形状协议 / 钥匙-门 / 多目标 / 推箱 / 非平稳 / k=0 对照。"""
import torch

from src.envs.rich_world import RichWorld
from src.envs.latent_grid import DIRS


def make(k, seed=0):
    return RichWorld(complexity=k, seed=seed)


class TestProtocol:
    def test_k0_delegates_to_grid_protocol(self):
        env = make(0)
        assert env.d_obs == 8 and env.G == 6
        obs = env.reset()
        assert obs.shape == (8,)
        o, r, done, info = env.step(0)
        assert "rare" in info

    def test_obs_shapes_by_k(self):
        # 布局：位置2 + 视野view² + [k≥2 进度2] + 时钟1 + 常量1
        expect = {1: 2 + 9 + 2, 2: 2 + 25 + 2 + 2, 3: 2 + 25 + 2 + 2, 4: 2 + 25 + 2 + 2}
        for k, d in expect.items():
            env = make(k)
            obs = env.reset()
            assert obs.shape == (d,), (k, obs.shape, d)

    def test_obs_prefix_position(self):
        """约定：obs[:2] = 位置归一化（coverage dims=(0,1) 依赖）。"""
        env = make(2, seed=1)
        obs = env.reset()
        assert 0 <= obs[0] <= 1 and 0 <= obs[1] <= 1

    def test_deterministic_same_seed(self):
        a, b = make(3, seed=7), make(3, seed=7)
        oa, ob = a.reset(), b.reset()
        assert torch.equal(oa, ob)
        ta = tb = 0
        for i in range(50):
            act = i % 4
            r1 = a.step(act)
            r2 = b.step(act)
            assert torch.equal(r1[0], r2[0]) and r1[1] == r2[1]
            ta += r1[1]; tb += r2[1]
        assert ta == tb

    def test_obs_values_bounded(self):
        for k in range(1, 5):
            env = make(k, seed=k)
            obs = env.reset()
            assert torch.all((obs >= 0) & (obs <= 1.0001)), k


class TestKeyDoor:
    def test_key_pickup_flags_rare(self):
        env = make(1, seed=0)
        env.reset()
        assert env.keys_pos, "k=1 应有钥匙"
        kx, ky = env.keys_pos[0]
        for dx, dy in DIRS:  # 找一个相邻格走过去
            px, py = kx - dx, ky - dy
            if 0 <= px < env.G and 0 <= py < env.G and (px, py) not in env.walls:
                env._agent = (px, py)
                break
        before = len(env._keys_held)
        move = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}[(dx, dy)]
        o, r, done, info = env.step(move)
        assert len(env._keys_held) == before + 1 and info["rare"] is True

    def test_k1_exit_gate_blocks_last_goal_without_key(self):
        """k=1 终点门：最后一个目标在无钥时不可登记（钥匙的存在意义）。"""
        env = make(1, seed=0)
        env.reset()
        # 先完成除最后一个外的所有目标（走到而非传送）
        for g in env.goals_pos[:-1]:
            gx, gy = g
            px, py = (gx - 1, gy) if gx > 0 else (gx + 1, gy)
            if (px, py) in env.walls or (px, py) in env.doors:
                px, py = (gx, gy - 1) if gy > 0 else (gx, gy + 1)
            env._agent = (px, py)
            move = 1 if px == gx - 1 else (3 if px == gx + 1 else (0 if py == gy - 1 else 2))
            env.step(move)
        assert len(env._goals_done) == env.n_goals - 1, env._goals_done
        # 无钥走向最后一个目标 → 不登记
        last = env.goals_pos[-1]
        lx, ly = last
        px, py = (lx - 1, ly) if lx > 0 and (lx - 1, ly) not in env.walls else (lx, ly - 1)
        env._agent = (px, py)
        env._keys_held.clear()
        move = 1 if px == lx - 1 else 0
        o, r, done, info = env.step(move)
        assert len(env._goals_done) == env.n_goals - 1 and not done
        # 持钥再来 → 登记 + 完成
        env._keys_held.add(env.keys_pos[0])
        env._agent = (px, py)
        o, r, done, info = env.step(move)
        assert done and r == 1.0

    def test_locked_door_blocks_without_key(self):
        """k=2 双房间：无钥撞门不动。"""
        env = make(2, seed=0)
        env.reset()
        assert env.doors, "k=2 应有门"
        dx, dy = env.doors[0]
        # 站到门的左侧邻格（若在墙内则试右侧）
        side = (dx - 1, dy) if dx > 0 and (dx - 1, dy) not in env.walls else (dx + 1, dy)
        env._agent = side
        env._keys_held.clear()
        move = 1 if side == (dx - 1, dy) else 3
        o, r, done, info = env.step(move)
        assert env._agent != (dx, dy) or env._doors_open[0]  # 没穿（或碰巧持钥开门）
        assert len(env._keys_held) == 0

    def test_key_opens_door(self):
        env = make(2, seed=0)
        env.reset()
        di = 0
        dx, dy = env.doors[di]
        key = env.door_keys[di]
        env._keys_held.add(key)          # 持钥
        side = (dx - 1, dy) if dx > 0 and (dx - 1, dy) not in env.walls else (dx + 1, dy)
        env._agent = side
        move = 1 if side == (dx - 1, dy) else 3
        o, r, done, info = env.step(move)
        assert env._doors_open[di] is True and info["rare"] is True


class TestGoalsAndBoxes:
    def test_multi_goal_completion(self):
        env = make(2, seed=1)
        env.reset()
        assert env.n_goals == 2
        # 依次"走到"每个目标：把 agent 放到目标左侧一格，向右走一步踩上
        for g in env.goals_pos:
            gx, gy = g
            px, py = (gx - 1, gy) if gx > 0 else (gx + 1, gy)
            assert (px, py) not in env.walls and (px, py) not in env.doors
            env._agent = (px, py)
            move = 1 if px == gx - 1 else 3
            o, r, done, info = env.step(move)  # 走上目标格 → 登记
        assert len(env._goals_done) == env.n_goals
        # 已完成的状态下：env.step 会因 goal 全完成而终局 +1（设计如此）——
        # 验证的是 done/r，而非"重复扣步"。重置后验证不重复登记。
        env.reset(seed=1)
        for g in env.goals_pos:
            gx, gy = g
            px, py = (gx - 1, gy) if gx > 0 else (gx + 1, gy)
            env._agent = (px, py)
            env.step(1 if px == gx - 1 else 3)
        assert len(env._goals_done) == env.n_goals
        o, r, done, info = env.step(0)
        assert done and r == 1.0  # 终局后 step 仍安全返回终局值

    def test_box_push_and_block(self):
        env = make(3, seed=0)
        env.reset()
        assert env.push_boxes and len(env._boxes) == 2
        bx, by = env._boxes[0]
        # 把 agent 放到箱子左侧向右推（右侧留空才推得动）
        env._agent = (bx - 1, by)
        free = ((bx + 1, by) not in env.walls and (bx + 1, by) not in env.doors
                and (bx + 1, by) not in env._boxes[1:] and (bx + 1, by) != env.hazard
                and bx + 1 < env.G)
        o, r, done, info = env.step(1)
        if free:
            assert (bx + 1, by) in env._boxes
        # 无论推没推动，不该崩
        assert env._t >= 1


class TestNonstationary:
    def test_doors_oscillate_at_k4(self):
        env = make(4, seed=0)
        env.reset()
        before = dict(env._doors_open)
        for _ in range(20):
            env.step(0)
        assert env._doors_open != before, "k=4 门应周期翻转"

    def test_k3_doors_do_not_oscillate(self):
        env = make(3, seed=0)
        env.reset()
        before = dict(env._doors_open)
        for _ in range(25):
            env.step(0)
        assert env._doors_open == before


class TestPartialObs:
    def test_partial_obs_masks_outside(self):
        env = make(3, seed=0)  # k≥3 部分可观测
        obs = env.reset()
        # 视野 5×5=25 格全部可见（中心），遮蔽不产生越界值——
        # 具体验证：把 agent 放角落，视野出界格子应=1.0（边界当墙）
        env._agent = (0, 0)
        obs = env._obs()
        vals = obs[4:4 + 25]  # k≥2: [0:2]位置 [2:4]进度 [4:29]视野
        corner = vals[0].item()  # 视野左上 = (-2,-2) 出界 → 1.0
        assert corner == 1.0
