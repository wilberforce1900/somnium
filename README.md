# mission_yin_yang

以中国传统哲学（阴阳 / 易经 / 数术）为归纳偏置来源的自组织认知架构研究项目。

核心主张：模型发育**从做梦开始**——混沌（噪声底质）→ 梦（内生潜空间滚动）→ 醒（现实锚定）
→ 反思（元认知读梦）；思想在连续潜空间进行，语言仅为 I/O 外设。

## 文档

- [PRINCIPLES.md](PRINCIPLES.md) — 原理→机制映射、两轴架构、术语纪律、消融登记册（A1–A6）
- [ROADMAP.md](ROADMAP.md) — 阶段 0 机制验证实验（E0–E4）、预注册阈值、决策门 G0
- [ARCHITECTURE.md](ARCHITECTURE.md) — 代码底座结构、必需（★）/可扩展（◇）/规划（○）标识

## 代码底座（v0.2）

单底座 + 双模式：`src/substrate.py`（共享潜态与参数，互根）+ `src/yang.py`（阳：流式发散）
+ `src/yin.py`（阴：能量弛豫）+ `src/taiji.py`（太极门控，A2 消融开关内置）；
上叠 `src/core.py`（世界模型 + MPC + ground 接地）、`src/dream.py`（梦期四开关调度）、
`src/codebook.py`（64 卦 VQ 码本 + 转移图 + 价值迭代）、`src/planner.py`（图上慢规划）、
`src/envs/`（ENV-A grid / ENV-C maze）、`experiments/`（E0/E1/E2 三实验入口）。

    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python experiments/e0_dream_first/run_e0.py --tag smoke   # E0
    .venv/bin/python experiments/e1_yinyang_gate/run_e1.py --task maze --tag smoke  # E1
    .venv/bin/python experiments/e2_hexagram_codebook/run_e2.py --k 64 --planner graph --tag smoke  # E2

## 当前状态

底座 + E0/E1/E2 三条实验流水线就绪（66 测试绿，smoke 均登记 `results/registry.md`）。
下一步：E0/E1/E2 正式全量跑（预注册阈值已锁定于 ROADMAP）；ENV-B（ARC）待网络。
三大消融入口全部可跑：A3 梦（E0）、A2 阴阳（E1）、A1 六十四卦（E2）。

## 纪律

哲学组件不绑定可测消融不得进入技术文档；负结果照记 `results/registry.md`；
不作 AGI 时间表承诺。
