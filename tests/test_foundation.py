"""公共底座冒烟与纪律测试。

覆盖：形状 / 混沌初始化 / 阳步数学 / 阴能量下降 / 互根梯度流 /
门控消融模式（A2）/ 确定性。全部 CPU、秒级。
"""
import torch

from src import Substrate, TaijiGate, YangMode, YinMode


def make(seed=0, batch=4, T=6, d_x=8, d_h=16):
    """统一装配：一个底座 + 阴阳两算子 + 门控。"""
    torch.manual_seed(seed)
    sub = Substrate(d_x=d_x, d_h=d_h)
    yang = YangMode(sub)
    yin = YinMode(sub)
    gate = TaijiGate(sub, yang, yin)
    s0 = sub.spawn(batch)
    xs = torch.randn(T, batch, d_x)
    return sub, yang, yin, gate, s0, xs


class TestShapes:
    def test_stream(self):
        _, yang, _, _, s0, xs = make()
        s = yang.stream(s0, xs)
        assert s.h.shape == s0.h.shape
        assert s.t == xs.shape[0]

    def test_relax(self):
        _, _, yin, _, s0, _ = make()
        s = yin.relax(s0, n_steps=4)
        assert s.h.shape == s0.h.shape
        assert s.t == 4

    def test_mixed(self):
        _, _, _, gate, s0, xs = make()
        s = s0
        for x in xs:
            s, a = gate.mixed_step(s, x)
        assert s.h.shape == s0.h.shape
        assert s.t == xs.shape[0]
        assert a.shape == (s0.h.shape[0], 1)


class TestChaos:  # L0：混沌出生
    def test_spawn_distribution(self):
        torch.manual_seed(0)
        sub = Substrate(d_x=8, d_h=16)
        h = sub.spawn(8192).h
        assert abs(h.std().item() - 1.0) < 0.05
        assert abs(h.mean().item()) < 0.05

    def test_spawn_reproducible(self):
        torch.manual_seed(1)
        a = Substrate().spawn(8).h
        torch.manual_seed(1)
        b = Substrate().spawn(8).h
        assert torch.equal(a, b)

    def test_inject(self):
        torch.manual_seed(2)
        sub = Substrate()
        h = torch.zeros(64, sub.d_h)
        h2 = sub.inject(h, level=1.0)
        assert abs(h2.std().item() - 1.0) < 0.05


class TestYangMath:  # 阳步逐位对公式
    def test_step_formula(self):
        sub, yang, _, _, s0, xs = make()
        with torch.no_grad():
            got = yang.step(s0, xs[0]).h
            exp = s0.h + yang.dt * torch.tanh(
                sub.cell(torch.cat([s0.h, xs[0]], dim=-1))
            )
        assert torch.allclose(got, exp, atol=1e-6)


class TestYinEnergy:  # 阴：能量必降
    def test_relax_decreases_energy(self):
        sub, _, yin, _, s0, _ = make(seed=3)
        with torch.no_grad():
            e0 = sub.energy_of(s0.h)
        s1 = yin.relax(s0, n_steps=8)
        with torch.no_grad():
            e1 = sub.energy_of(s1.h)
        assert torch.all(e1 < e0)


class TestMutualRooting:  # 互根：参数同体、梯度互通
    def test_shared_parameter_identity(self):
        _, yang, yin, _, _, _ = make()
        assert yang.substrate.cell is yin.substrate.cell
        assert yang.substrate.energy_head is yin.substrate.energy_head

    def test_yang_grads_reach_cell(self):
        sub, yang, _, _, s0, xs = make()
        loss = yang.stream(s0, xs).h.pow(2).mean()
        loss.backward()
        assert sub.cell.weight.grad is not None
        assert sub.cell.weight.grad.abs().sum().item() > 0

    def test_yin_grads_reach_energy_head(self):
        sub, yang, yin, _, s0, xs = make()
        s = yang.stream(s0, xs)
        loss = sub.energy_of(yin.relax(s, 3).h).sum()
        loss.backward()
        g = sub.energy_head[0].weight.grad
        assert g is not None
        assert g.abs().sum().item() > 0

    def test_learned_gate_trains_everything(self):
        sub, _, _, gate, s0, xs = make()
        s = s0
        for x in xs:
            s, _ = gate.mixed_step(s, x)
        s.h.sum().backward()
        for name, p in (
            ("cell", sub.cell.weight),
            ("energy", sub.energy_head[0].weight),
            ("gate", gate.gate_net.weight),
        ):
            assert p.grad is not None, name
            assert p.grad.abs().sum().item() > 0, name


class TestGateAblation:  # A2 消融开关（PRINCIPLES §11）
    def test_fixed_yang_equals_pure_yang(self):
        sub, yang, yin, _, s0, xs = make(seed=5)
        g = TaijiGate(sub, yang, yin, gate_mode="fixed_yang")
        s_m, a = g.mixed_step(s0, xs[0])
        assert torch.allclose(a, torch.ones_like(a))
        assert torch.allclose(s_m.h, yang.step(s0, xs[0]).h, atol=1e-6)

    def test_fixed_yin_equals_pure_yin(self):
        sub, yang, yin, _, s0, xs = make(seed=5)
        g = TaijiGate(sub, yang, yin, gate_mode="fixed_yin")
        s_m, a = g.mixed_step(s0, xs[0])
        assert torch.allclose(a, torch.zeros_like(a))
        assert torch.allclose(s_m.h, yin.step(s0).h, atol=1e-6)

    def test_schedule_mode(self):
        sub, yang, yin, _, s0, xs = make(seed=5)
        g = TaijiGate(sub, yang, yin, gate_mode="schedule", schedule=lambda t: 0.9)
        _, a = g.mixed_step(s0, xs[0])
        assert torch.allclose(a, torch.full_like(a, 0.9))

    def test_invalid_mode_rejected(self):
        sub, yang, yin, _, _, _ = make()
        try:
            TaijiGate(sub, yang, yin, gate_mode="harmony")
        except ValueError:
            pass
        else:
            raise AssertionError("非法 gate_mode 应被拒绝")


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        def run(seed):
            _, _, _, gate, s0, xs = make(seed=seed)
            s = s0
            for x in xs:
                s, _ = gate.mixed_step(s, x)
            return s.h

        assert torch.equal(run(7), run(7))
        assert not torch.allclose(run(7), run(8))
