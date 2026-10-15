# ITER12-PLAN · ENV-E 重开第二站（Stage 2）：λ 阶梯矩阵——非平稳维护度与梦调度

> **性质**：预登记（阈值与决策树跑前锁定；锁后不改，改动另立文件）
> **前身**：ITER11 Stage 1（单翻转恢复）：恢复快一个数量级（τ50 25-70、τ90≈150）
>   且**对调度不敏感**（r_τ50=0.89）——梦=出生优惠非再读优惠。Stage 2 触发条件
>   （恢复存在门）已满足。
> **时钟已到手**：dwell 阶梯由 τ 反推（ENV-E 评审惯例 dwell∈{5τ, τ, τ/3}）：
>   **{750, 150, 50}**（=5τ90 / τ90 / ≈τ50；λ = {0.00133, 0.00667, 0.02}——
>   竟与原提案的市场刻度 {0.002, 0.01, 0.05} 同带）。
> **反幻觉**：本矩阵测**语义维护度**（readout），不判行为层梦增益；
>   不新增组件/剂量/分阶（各有已答/已存档入口）。

## 0. 问题

建立之后进入**连续随机切换**（NonStatWorld lam 机制：每步以概率 λ 翻转全局 regime）——
（Q1）切换多快时语义层开始跟丢？维护度随 dwell 的退化曲线是什么形状？
（Q2）dream-first 的"出生优惠"（PRE 建立 ~1.4×、稳压）是否延续为**切换期的维护优势**？
（先验注记（跑前登记的预期，非判据）：ITER11 已示恢复对调度不敏感 → Q2 预期 ≈0。）

## 1. 设计（单变元=λ（dwell）；协议逐字复用 ITER9/10/11 冻结件）

- **协议**：PRE 2400@ρ0（建立，同 ITER11：采集 fracA.3+fracB.15+55%随机 / mixed /
  clip1.0 / eval 每 5 锁 4245）→ **切换相位 2400 周期（env.lam=1/dwell，连续随机
  翻转，不作手动翻转）**。梦臂=E0 标准前奏（dream_phase×200，同 ITER10/11）。
- **读数的"当前 regime"口径**：切换期每次 eval 同时评 ev0/ev1；**acc_sw(t) = 当前
  env.regime 对应的那半**（in-regime 维护度）。主读数 **acc_sw_end = 末十均值**；
  次读数 acc_sw 全相位均值、acc0/acc1 双曲线、switches 计数（≈2400/dwell）。
- **矩阵**：{wake, dream} × dwell{750, 150, 50} × seed{0,1} = **12 run**（4800 周期，
  ~71min/run）。λ→0 锚（单翻转）已由 ITER11 入库，不重复。
- **自带锚点**：PRE 段（每个 (schedule, seed) 的三个 dwell 臂 PRE 相同）应与归档
  **逐位一致**——wake ≡ iter9 `term_scale_dh64_lam1_s{0,1}`、dream ≡ iter10
  `dream_sem_dh64_s{0,1}`（c≤2400）。等价性测试守（tests/test_switch.py）。

## 2. 预注册判据（跑前锁定）

1. **有效性门**：12/12 PRE 过门（C_pre <2400）；PRE 行与归档逐位（偏差>1e-9
   → 停机查因，结果作废）。
2. **维持带**（wake 臂、双 seed、按 dwell 逐判）：acc_sw_end ≥0.80 = 维护；
   [0.60, 0.80) = 部分恢复平衡；<0.60 = 失守。
3. **退化曲线门**：acc_sw_end(dwell50) ≤ acc_sw_end(dwell750) − 0.10（双 seed 一致）
   → "退化曲线实存"（Q1 正结果）；否则"全档维持"（系统比 τ 预测的更稳）。
4. **调度门（主，Q2）**：Δ_sw(λ) = mean over seeds [acc_sw_end(dream) − acc_sw_end(wake)]：
   ≥ +0.08 梦改善维护 / ≤ −0.08 梦损害维护 / 之间 无差异；按 λ 逐报 + 汇总。
5. **副（报告不判）**：PRE 建立比 C_pre 复核（ITER10 ~1.4× 的第四次独立复现）；
   acc0/acc1 双曲线形态（切换期的相位滞后）；switches 实测计数核对（泊松带内）。
6. **先验登记**（非判据）：dwell 750 预期近满维护；150 为边界带；50 预期部分失衡；
   Δ_sw 预期 ≈0。

## 3. 不授权项

- 行为层梦增益、梦剂量、梦分阶（staging）——各有已答/已存档入口，本矩阵不碰；
- 高容量档（ITER10 已示 256/1024 无调度效应）；
- Stage 3（退化档加码/上容量）由本矩阵结果决定，另立预登记。

## 4. 运行与成本

- 双链并行（各 6 run）：A=wake dwell750→150→50 (s0) → (s1)；B=dream 同序。
  墙钟 ≈8.5-9h；≈$7。
- 输出：`experiments/env_e_v3/switch_{schedule}_dwell{D}_s{seed}.{jsonl,final.json}`
  （dream 臂另出 `.dream.jsonl`）。
- 部署前置：VM 依赖件 md5 对齐（dream.py/pilot_terminal.py/pilot_terminal_scale.py/
  pilot_dream_semantics.py/pilot_recovery.py）。
