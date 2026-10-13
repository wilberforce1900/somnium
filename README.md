# mission_yin_yang · **Somnium**

**[English](#english) | 中文**

以中国传统哲学（阴阳 / 易经 / 数术）为归纳偏置来源的自组织认知架构研究项目；
模型产物名为 **Somnium**（拉丁语"梦"，昵称晓梦，命名册见 [MODEL-NAME.md](MODEL-NAME.md)）。

> *"Do androids dream? Yes — and it saves 60% of the data... when the model is small."*
> AGI 是硅基生命的梦醒时刻——不是造神，是唤醒。

**头条结果**：梦期预训练（先梦后醒）在 16K 参数尺度五次独立运行复现 ~40% 真实数据
样本效率增益（E0 三 seed + R1 跨平台两 seed，全部预注册）；R1 规模阶梯显示该增益
随容量衰减，R1b 剂量对照证明加大梦预算不救。**v1.2 更新（2026-10）**：五连灭排除链
+ 四臂归一化对照证明宽潜态病态（d_h≥256）是**可修复的优化几何缺陷**（出口 LayerNorm
治愈：1024 档 1.03→0.17，"d_h≤256 天花板" 论撤销）；仪器三修（解码 R² 替代移动标尺、
收敛协议、混合回放止漂 0.415→0.011）后 A3 重测——**容量受限结论在健康仪器上更干净
地重新成立**：梦增益只存活于 wake 自身容量受限的 64 档（0.56–0.69），256/1024 归零
（0.74–0.98/未达标），且全梦电池不优于"带回放的醒"——**回放是载荷组件**。
完整证伪史见登记表。

## 论文（Preprint）

**Somnium: A Recurrent World Model that Learns to Dream Before It Wakes** · Xiaoming (Will) Wang

📄 [PDF v1.2](paper/somnium-paper-v1.2.pdf) · [LaTeX 源码](paper/main.tex) · [参考文献](paper/refs.bib) · [Zenodo 记录](https://zenodo.org/records/23333007)

**Cite as:** Wang, X. (2026). *Somnium: A Recurrent World Model that Learns to Dream Before It Wakes*. Zenodo. [DOI 10.5281/zenodo.23278220](https://doi.org/10.5281/zenodo.23278220)（概念 DOI；v1.2 版本 DOI [10.5281/zenodo.23333007](https://doi.org/10.5281/zenodo.23333007)）

> 9 页完整研究报告：预注册协议、五个独立复现的梦增益、容量边界三线证据、
> **宽潜态病态的排除链与治愈（含对自家 v1.1"表达力天花板"结论的公开撤回）、
> 移动标尺与仪器修复、健康仪器上的 A3 重测**、负结果登记册（A1/A2/A5）、
> 两夜自主训练（871 轮生长与自愈回滚）。
> 写作说明：科学工作（实验设计、代码、150+ 次运行记录）由项目主导完成；文字起草
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
and shown to **fade with scale** (R1 ladder; dose control R1b confirms the fade
is real). **v1.2:** the wide-model pathology turned out to be a *fixable
optimization defect* — output LayerNorm restores wide models to small-model
accuracy, retracting the earlier "expressivity ceiling" — and with the
measurement instruments repaired (scale-invariant decode fidelity, converged
protocols, mixed replay), the capacity-bound conclusion returns in cleaner
form: the dream advantage lives only where wake itself is capacity-limited,
and *replay* is the load-bearing component. What doesn't survive:
hexagram VQ codebooks, gate-over-fixed advantages, dream-reading calibration
feedback, and our own v1.1 ceiling claim — all falsified with pre-registered
thresholds and kept in the registry, because *a theory that can be killed is
a theory that can evolve*.

Docs: [Paper v1.2 (PDF)](paper/somnium-paper-v1.2.pdf) · [DOI: 10.5281/zenodo.23278220](https://doi.org/10.5281/zenodo.23278220) ·
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

**iter8 仪器修复轮收官（2026-10-13）**：阶段 0 判定齐——P0-A 混合回放治愈近期化
漂移（9000 轮 0.011，门限 19× 下方）；P0-B A3 重测（18 run）：容量受限衰减在
健康仪器上重新成立、回放=载荷组件；P0-C ENV-E 校时钟 pilot：slip-only 预测层太
温和（预切换已泛化）→ 原 48-run 矩阵否决，v2 方向=策略层 regime 差异。
此前：iter7 (b) 线（出口归一化治愈，[ITER7-WORLD-FIRST.md](ITER7-WORLD-FIRST.md) §2.5）、
夜三 62 run（富世界长训/协议桥/种子复现，[NIGHT3-PLAN.md](NIGHT3-PLAN.md)）、
[ITER8-PLAN.md](ITER8-PLAN.md) 全链预注册。
**下一步（待拍板排序）**：ENV-E v2（策略层非平稳）、ENV-D k5/6 结构轴
（R5 门复活=收敛协议下 256→1024 改善 ≥10%）、器官线（梦调节器——触发条件
=非平稳矩阵对调度敏感；"副手不模拟世界，它模拟做梦者"）。
三大纪律：预注册阈值跑前锁定、负结果照记登记表、哲学词必须绑定机制。

## 纪律

哲学组件不绑定可测消融不得进入技术文档；负结果照记 `results/registry.md`；
不作 AGI 时间表承诺。
