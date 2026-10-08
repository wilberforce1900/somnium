"""阶段 1 工程前置测试：checkpoint 断点续跑 / device 接线（cpu 路径）。"""
import random

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.core import WorldModel
from src.dream import DreamConfig, DreamLog, DreamScheduler, EpisodeBuffer
from src.envs import LatentGrid, random_rollout


def make_setup(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    env = LatentGrid(grid=4, slip=0.1, p_threat_move=0.05, horizon=15, seed=seed)
    model = WorldModel(d_obs=env.d_obs, n_actions=env.n_actions, d_h=24, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = EpisodeBuffer(capacity=50)
    for _ in range(5):
        buffer.add_episode(*random_rollout(env))
    sched = DreamScheduler(model, buffer,
                           DreamConfig(batch=8, t_win=6, imagine_len=4),
                           log=DreamLog(), rng=random.Random(seed))
    return env, model, opt, buffer, sched


class TestCheckpoint:
    def test_roundtrip_restores_model_opt_buffer_rng(self, tmp_path):
        env, model, opt, buffer, sched = make_setup()
        for i in range(3):
            b = buffer.sample_windows(8, 6, rng=random.Random(i))
            sched.wake_update(opt, b)
        p = tmp_path / "ckpt.pt"
        save_checkpoint(p, model=model, optimizer=opt, cycle=3,
                        extra={"env_steps": 123}, env_rng=env.rng, buffer=buffer)
        # 固定批，训练后模型的输出
        b = buffer.sample_windows(8, 6, rng=random.Random(42))
        torch.manual_seed(1)
        h1, _ = model(b["obs"], b["act"])
        # 新模型恢复
        _, model2, opt2, buffer2, _ = make_setup()
        cycle, extra = load_checkpoint(p, model=model2, optimizer=opt2,
                                       env_rng=env.rng, buffer=buffer2)
        assert cycle == 3 and extra["env_steps"] == 123
        assert len(buffer2) == len(buffer)
        torch.manual_seed(1)
        h2, _ = model2(b["obs"], b["act"])
        assert torch.equal(h1, h2)  # 权重逐位一致
        # 优化器状态恢复（Adam 动量逐位一致）
        assert torch.equal(opt.state_dict()["state"][0]["exp_avg"],
                           opt2.state_dict()["state"][0]["exp_avg"])
        # 随机态恢复：保存（消费 expected 后的状态）→破坏→恢复→后续流一致
        torch.manual_seed(9)
        _expected = torch.rand(4)
        save_checkpoint(tmp_path / "y.pt", model=model)
        torch.manual_seed(123)  # 破坏随机态
        load_checkpoint(tmp_path / "y.pt", model=model)
        followup_restored = torch.rand(4)
        torch.manual_seed(9)
        torch.rand(4)  # 消费掉 expected，回到保存时状态
        followup_direct = torch.rand(4)
        assert torch.equal(followup_restored, followup_direct)

    def test_atomic_no_tmp_leftover(self, tmp_path):
        env, model, opt, buffer, sched = make_setup()
        p = tmp_path / "sub" / "ckpt.pt"
        save_checkpoint(p, model=model, optimizer=opt, cycle=1,
                        env_rng=env.rng, buffer=buffer)
        assert p.exists() and not (tmp_path / "sub" / "ckpt.pt.tmp").exists()


class TestDeviceWiring:
    def test_explicit_cpu_and_offdevice_inputs(self):
        """cpu 设备路径 + 输入自动收敛（模拟异构输入，cpu 上等价 no-op）。"""
        env, model, opt, buffer, sched = make_setup(seed=1)
        model.to("cpu")
        b = buffer.sample_windows(8, 6, rng=random.Random(0))
        loss, parts = model.loss_on_batch(b["obs"], b["act"], b["obs_next"], b["rew"])
        assert torch.isfinite(loss)
        obs = env.reset()
        h = model.ground(obs)
        assert h.device == model.substrate.cell.weight.device
        a, _ = model.plan(obs, h, horizon=3, k=8)
        assert a in range(4)

    def test_imagine_on_model_device(self):
        """梦期想象与模型同设备（cpu 上验证代码路径不炸）。"""
        env, model, opt, buffer, sched = make_setup(seed=2)
        model.to("cpu")
        summary = sched.dream_phase(opt)
        assert "imagination" in summary
