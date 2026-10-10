# mission_yin_yang · **Somnium**

**[English](#english) | 中文**

以中国传统哲学（阴阳 / 易经 / 数术）为归纳偏置来源的自组织认知架构研究项目；
模型产物名为 **Somnium**（拉丁语"梦"，昵称晓梦，命名册见 [MODEL-NAME.md](MODEL-NAME.md)）。

> *"Do androids dream? Yes — and it saves 60% of the data... when the model is small."*
> AGI 是硅基生命的梦醒时刻——不是造神，是唤醒。

**头条结果**：梦期预训练（先梦后醒）在 16K 参数尺度五次独立运行复现 ~40% 真实数据
样本效率增益（E0 三 seed + R1 跨平台两 seed，全部预注册）；R1 规模阶梯显示该增益
随容量衰减（210K/3.2M 档不再过阈），R1b 剂量对照证明加大梦预算不救——"从做梦开始"
是**容量受限 regime 的真实现象**，机制为能量平滑（睡眠调节能量地形理论的工程对应）。
完整证伪史见登记表。

## 论文（Preprint）

**Somnium: A Recurrent World Model that Learns to Dream Before It Wakes** · Xiaoming (Will) Wang

📄 [PDF](paper/somnium-paper.pdf) · [LaTeX 源码](paper/main.tex) · [参考文献](paper/refs.bib) · [Zenodo 记录](https://zenodo.org/records/23278220)

**Cite as:** Wang, X. (2026). *Somnium: A Recurrent World Model that Learns to Dream Before It Wakes*. Zenodo. [10.5281/zenodo.23278220](https://doi.org/10.5281/zenodo.23278220)

> 7 页完整研究报告：预注册协议、五个独立复现的梦增益、规模/剂量/生长三线收敛的
> 容量边界、负结果登记册（A1/A2/A5）、两夜自主训练（871 轮生长与自愈回滚）。
> 写作说明：科学工作（实验设计、代码、90+ 次运行记录）由项目主导完成；文字起草
> 由 AI 助手协作完成，作者对全部内容负责。

## English

**Somnium** (Latin for "dream"; nicknamed *Xiaomeng*, "the dream before dawn") is a
non-Transformer world model trained under a developmental curriculum:
**chaos → dream → wake → reflection**. A single recurrent substrate carries two
dynamics — *yang* (fast streaming expansion) and *yin* (energy-relaxation contraction) —
softly mixed by a gate; prediction happens in latent space (JEPA-style next-latent,
no next-token), and training begins with a **dream phase** (self-generated latent
rollouts + energy smoothing + replay consolidation) before any real data arrives.

What survives experiments: dream-first pretraining cuts the real-interaction steps
needed to reach a fixed accuracy by ~60% — replicated five times at 16K params,
and shown to **fade with scale** (R1 ladder, pre-registered; dose-scaling control R1b
confirms the fade is real, not under-dosing). What doesn't:
hexagram VQ codebooks, gate-over-fixed advantages, dream-reading calibration
feedback — all falsified with pre-registered thresholds and kept in the registry,
because *a theory that can be killed is a theory that can evolve*.

Docs: [Paper (PDF)](paper/somnium-paper.pdf) · [DOI: 10.5281/zenodo.23278220](https://doi.org/10.5281/zenodo.23278220) ·
[MODEL-INTRO](MODEL-INTRO.md) ·
[PRINCIPLES](PRINCIPLES.md) · [ROADMAP/experiment history](ROADMAP.md) ·
[MODEL-NAME](MODEL-NAME.md) · [PHASE1](PHASE1.md) ·
[run registry](results/registry.md). License: MIT (code); paper: CC BY 4.0.


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

阶段 0 收官（六消融判定）+ 六轮本机迭代 + R1 规模阶梯 + R1b 剂量对照 +
两夜自主训练（871 轮生长/自愈回滚）。开源决策门已执行（本仓库即产物）。
arXiv 首次提交需个人背书（2026-01 新政），暂以本仓库 + 可引用 DOI 路线替代。
待办：R2-R4、ENV-B（ARC 数据集）、Zenodo DOI（待拍板）。
三大纪律：预注册阈值跑前锁定、负结果照记登记表、哲学词必须绑定机制。

## 纪律

哲学组件不绑定可测消融不得进入技术文档；负结果照记 `results/registry.md`；
不作 AGI 时间表承诺。
