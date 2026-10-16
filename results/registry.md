# Run 登记表

一行一 run，负结果照记（ROADMAP §0）。

公平性注记：梦日程（dream_first/alternate）含额外离线梯度步；E0 比较基准=真实环境交互步数（PRINCIPLES §11 A3）。
辅助指标列：E0=MPC 回报；E1=精度(acc)+门控 α+α-难度相关（maze 墙位准确率 / grid 容差命中率）。
E1 预注册偏离（2026-10-08）：A2 判定任务集 ENV-B+ENV-C → ENV-A+ENV-C（ENV-B 外网不可达）。

**2026-10-08 预注册修订（全量正式跑前锁定，细则 ROADMAP §1.7）**：E0/E2 训练混入 30% 引导局；eval 前锁 torch seed 4242；E0=3日程×3seed(9)，E1=learned/fixed_yang×grid/maze×2seed(8)，E2=6配置×2seed(12)；E0 判定=steps_to_target ≤ 0.6×wake_only 总步数；E2 图搜索=plan_shaped（v1 graph 与 v2 shaped 均未过 p0check，A1 预记实现效力警告）。

| 日期 | 实验 | 日程/配置 | seed | 真实交互步数 | eval 潜态预测 MSE | 辅助指标 | 结论 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-08 | E0 | wake_only | 0 | 254 | 0.947154 | 0.1467 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E0 | dream_first | 0 | 254 | 0.899283 | 0.19 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E0 | alternate | 0 | 254 | 0.871319 | 0.19 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E1 | grid/learned+r0 | 0 | 254 | 0.93762 | acc=0.019531 α=0.398375 corr=0.197228 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E1 | grid/fixed_yang+r0 | 0 | 254 | 0.988658 | acc=0.03125 α=1.0 corr=0.0 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E1 | grid/fixed_yin+r0 | 0 | 254 | 0.98642 | acc=0.011719 α=0.0 corr=0.0 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E1 | maze/learned+r0 | 0 | 1432 | 0.714555 | acc=0.506836 α=0.405535 corr=-0.320916 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E1 | maze/fixed_yang+r0 | 0 | 1432 | 0.579881 | acc=0.485352 α=1.0 corr=0.0 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E1 | maze/learned+r2 | 0 | 1432 | 0.67479 | acc=0.503906 α=0.405769 corr=-0.312844 | 流水线验证 | smoke/smoke |
| 2026-10-08 | E2 | grid_long/k0/free | 0 | 1972 | 0.783966 | succ=0.0 ret=-0.6 rand=0.00 | 作废：env 误含 hazard，重跑见 smoke2 |
| 2026-10-08 | E2 | grid_long/k64/graph | 0 | 1972 | 0.971853 | succ=0.0 ret=-0.6 rand=0.00 | 作废：env 误含 hazard，重跑见 smoke2 |
| 2026-10-08 | E2 | grid_long/k64/free | 0 | 1972 | 0.917984 | succ=0.0 ret=-0.76 rand=0.00 | 作废：env 误含 hazard，重跑见 smoke2 |
| 2026-10-08 | E2 | grid_long/k64/free_match | 0 | 1972 | 0.929482 | succ=0.0 ret=-0.685 rand=0.00 | 作废：env 误含 hazard，重跑见 smoke2 |
| 2026-10-08 | E2 | grid_long/k32/graph | 0 | 1972 | 0.915057 | succ=0.0 ret=-0.6 rand=0.00 | 作废：env 误含 hazard，重跑见 smoke2 |
| 2026-10-08 | E2 | grid_long/k128/graph | 0 | 1972 | 0.818912 | succ=0.0 ret=-0.6 rand=0.00 | 作废：env 误含 hazard，重跑见 smoke2 |
| 2026-10-08 | E2 | grid_long/k0/free | 0 | 2400 | 0.772165 | succ=0.0 ret=-0.6 rand=0.00 | 流水线验证 | smoke/smoke2 |
| 2026-10-08 | E2 | grid_long/k64/graph | 0 | 2400 | 0.837141 | succ=0.0 ret=-0.6 rand=0.00 | 流水线验证 | smoke/smoke2 |
| 2026-10-08 | E2 | grid_long/k64/free | 0 | 2400 | 0.887296 | succ=0.1667 ret=-0.4317 rand=0.00 | 流水线验证 | smoke/smoke2 |
| 2026-10-08 | E2 | grid_long/k64/free_match | 0 | 2400 | 0.942391 | succ=0.0 ret=-0.6 rand=0.00 | 流水线验证 | smoke/smoke2 |
| 2026-10-08 | E2 | grid_long/k32/graph | 0 | 2400 | 0.938542 | succ=0.0 ret=-0.6 rand=0.00 | 流水线验证 | smoke/smoke2 |
| 2026-10-08 | E2 | grid_long/k128/graph | 0 | 2400 | 0.857418 | succ=0.0 ret=-0.6 rand=0.00 | 流水线验证 | smoke/smoke2 |
| 2026-10-08 | E0 | wake_only+g0.3 | 0 | 5225 | 0.932016 | -0.202 | 基线（target 来源） | full/formal |
| 2026-10-08 | E0 | dream_first+g0.3 | 0 | 5225 | 0.24583 | -0.5 | ✅ A3 正信号：steps_to_target 0.39/0.39/0.42 ≤ 0.6 | full/formal |
| 2026-10-08 | E0 | alternate+g0.3 | 0 | 5225 | 0.838689 | -0.202 | 方向一致未达阈：0.77/0.79/0.76 | full/formal |
| 2026-10-08 | E0 | wake_only+g0.3 | 1 | 5381 | 0.883469 | -0.5 | 基线（target 来源） | full/formal |
| 2026-10-08 | E0 | dream_first+g0.3 | 1 | 5381 | 0.241631 | -0.26 | ✅ A3 正信号：steps_to_target 0.39/0.39/0.42 ≤ 0.6 | full/formal |
| 2026-10-08 | E0 | alternate+g0.3 | 1 | 5381 | 0.720234 | -0.5 | 方向一致未达阈：0.77/0.79/0.76 | full/formal |
| 2026-10-08 | E0 | wake_only+g0.3 | 2 | 5363 | 0.898631 | -0.61 | 基线（target 来源） | full/formal |
| 2026-10-08 | E0 | dream_first+g0.3 | 2 | 5363 | 0.218092 | -0.708 | ✅ A3 正信号：steps_to_target 0.39/0.39/0.42 ≤ 0.6 | full/formal |
| 2026-10-08 | E0 | alternate+g0.3 | 2 | 5363 | 0.778517 | -0.63 | 方向一致未达阈：0.77/0.79/0.76 | full/formal |
| 2026-10-08 | E1 | grid/learned+r0 | 0 | 6891 | 0.989094 | acc=0.021484 α=0.245359 corr=0.242035 | ❌ A2 未达 +2pp（+0.9pp） | full/formal |
| 2026-10-08 | E1 | grid/fixed_yang+r0 | 0 | 6891 | 1.035356 | acc=0.015625 α=1.0 corr=0.0 | 基线 | full/formal |
| 2026-10-08 | E1 | maze/learned+r0 | 0 | 32000 | 0.811128 | acc=0.507812 α=0.347634 corr=0.027939 | ❌ A2 未达 +2pp（-0.8pp） | full/formal |
| 2026-10-08 | E1 | maze/fixed_yang+r0 | 0 | 32000 | 0.651703 | acc=0.520508 α=1.0 corr=0.0 | 基线 | full/formal |
| 2026-10-08 | E1 | grid/learned+r0 | 1 | 6887 | 0.970816 | acc=0.021484 α=0.286672 corr=-0.025302 | ❌ A2 未达 +2pp（+0.9pp） | full/formal |
| 2026-10-08 | E1 | grid/fixed_yang+r0 | 1 | 6887 | 0.902099 | acc=0.009766 α=1.0 corr=0.0 | 基线 | full/formal |
| 2026-10-08 | E1 | maze/learned+r0 | 1 | 31915 | 0.827122 | acc=0.490723 α=0.322559 corr=-0.27135 | ❌ A2 未达 +2pp（-0.8pp） | full/formal |
| 2026-10-08 | E1 | maze/fixed_yang+r0 | 1 | 31915 | 0.648073 | acc=0.494629 α=1.0 corr=0.0 | 基线 | full/formal |
| 2026-10-08 | E2 | grid_long/k0/free+g0.3 | 0 | 11159 | 0.889495 | succ=0.8125 ret=0.4706 rand=0.00 | 基线 succ=0.406 | full/formal |
| 2026-10-08 | E2 | grid_long/k64/shaped+g0.3 | 0 | 11159 | 0.958394 | succ=0.0 ret=-0.6 rand=0.00 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k64/free+g0.3 | 0 | 11159 | 0.958394 | succ=0.0 ret=-0.6 rand=0.00 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k64/free_match+g0.3 | 0 | 11159 | 0.8315 | succ=0.0 ret=-0.6 rand=0.00 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k32/shaped+g0.3 | 0 | 11159 | 0.896969 | succ=0.0625 ret=-0.5169 rand=0.00 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k128/shaped+g0.3 | 0 | 11159 | 0.900995 | succ=0.1875 ret=-0.3519 rand=0.00 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k0/free+g0.3 | 1 | 11270 | 0.781501 | succ=0.0 ret=-0.6 rand=0.06 | 基线 succ=0.406 | full/formal |
| 2026-10-08 | E2 | grid_long/k64/shaped+g0.3 | 1 | 11270 | 0.922392 | succ=0.0 ret=-0.6 rand=0.06 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k64/free+g0.3 | 1 | 11270 | 0.903094 | succ=0.0 ret=-0.6 rand=0.06 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k64/free_match+g0.3 | 1 | 11270 | 0.847496 | succ=0.0 ret=-0.6 rand=0.06 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k32/shaped+g0.3 | 1 | 11270 | 0.87705 | succ=0.0 ret=-0.6 rand=0.06 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E2 | grid_long/k128/shaped+g0.3 | 1 | 11270 | 0.894089 | succ=0.0 ret=-0.6 rand=0.06 | ❌ A1 负结果：码本在训练中伤及连续模型 → 易层降级 | full/formal |
| 2026-10-08 | E3 | dream_full | 0 | 850 | 0.7600 | ret12=1.2166 rob=0.98 | 待判定 | smoke/smoke |
| 2026-10-08 | E3 | dream_full | 0 | 1844 | 0.2896 | ret12=1.5829 rob=1.00 | ✅ A4 基准：保持率 +1.57（vs none +1.34，差 23.1pp ≥10pp） | full/formal |
| 2026-10-08 | E3 | dream_none | 0 | 1844 | 0.5569 | ret12=1.3598 rob=0.97 | 无梦基线（正迁移存在，梦在其上加 23pp） | full/formal |
| 2026-10-08 | E3 | dream_no_imagination | 0 | 1844 | 0.3158 | ret12=1.5637 rob=0.92 | 想象对保持率无独立贡献（其增益在 E0 样本效率） | full/formal |
| 2026-10-08 | E3 | dream_no_consolidation | 0 | 1844 | 0.3512 | ret12=1.5298 rob=0.97 | ✅ consolidation 有独立贡献（关掉降 ≥5pp） | full/formal |
| 2026-10-08 | E3 | dream_no_reverse_learning | 0 | 1844 | 0.3465 | ret12=1.5395 rob=0.97 | ✅ reverse_learning 有独立贡献（关掉降 ≥5pp） | full/formal |
| 2026-10-08 | E3 | dream_no_rehearsal | 0 | 1844 | 0.3468 | ret12=1.5351 rob=0.97 | ✅ rehearsal 有独立贡献（关掉降 ≥5pp） | full/formal |
| 2026-10-08 | E3 | dream_full | 1 | 1827 | 0.3353 | ret12=1.556 rob=1.01 | ✅ A4 基准：保持率 +1.57（vs none +1.34，差 23.1pp ≥10pp） | full/formal |
| 2026-10-08 | E3 | dream_none | 1 | 1827 | 0.6140 | ret12=1.3172 rob=1.00 | 无梦基线（正迁移存在，梦在其上加 23pp） | full/formal |
| 2026-10-08 | E3 | dream_no_imagination | 1 | 1827 | 0.3576 | ret12=1.5404 rob=0.98 | 想象对保持率无独立贡献（其增益在 E0 样本效率） | full/formal |
| 2026-10-08 | E3 | dream_no_consolidation | 1 | 1827 | 0.4065 | ret12=1.4869 rob=0.97 | ✅ consolidation 有独立贡献（关掉降 ≥5pp） | full/formal |
| 2026-10-08 | E3 | dream_no_reverse_learning | 1 | 1827 | 0.4170 | ret12=1.4788 rob=0.97 | ✅ reverse_learning 有独立贡献（关掉降 ≥5pp） | full/formal |
| 2026-10-08 | E3 | dream_no_rehearsal | 1 | 1827 | 0.3998 | ret12=1.4974 rob=0.97 | ✅ rehearsal 有独立贡献（关掉降 ≥5pp） | full/formal |
| 2026-10-08 | E3b | read_dream_read | 0 | - | 0.040328 | cal=0.034466 sel=0.00166 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/smoke |
| 2026-10-08 | E3b | read_dream_read | 0 | - | 0.040328 | cal=0.034466 sel=0.00166 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/formal |
| 2026-10-08 | E3b | read_dream_read | 1 | - | 0.015386 | cal=0.048058 sel=0.004539 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/formal |
| 2026-10-08 | E3b | read_dream_random | 0 | - | 0.040327 | cal=0.034466 sel=0.001661 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/formal |
| 2026-10-08 | E3b | read_dream_random | 1 | - | 0.015387 | cal=0.048058 sel=0.00454 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/formal |
| 2026-10-08 | E3b | read_dream_none | 0 | - | 0.074125 | cal=0.037767 sel=0.010347 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/formal |
| 2026-10-08 | E3b | read_dream_none | 1 | - | 0.031219 | cal=0.048058 sel=0.005107 | v1 空转：imagine 目标内容不敏感，伪观测盲区偏置无效果 | full/formal |
| 2026-10-08 | E3b | read_dream_read | 0 | - | 0.040857 | cal=0.034466 sel=0.002798 | ❌ A5 未达阈：v2 机制生效但 cal_err 相对 random 降幅 0%（<20%） | full/formal2 |
| 2026-10-08 | E3b | read_dream_read | 1 | - | 0.015319 | cal=0.048058 sel=0.006267 | ❌ A5 未达阈：v2 机制生效但 cal_err 相对 random 降幅 0%（<20%） | full/formal2 |
| 2026-10-08 | E3b | read_dream_random | 0 | - | 0.040327 | cal=0.034466 sel=0.001661 | 对照（均匀回放） | full/formal2 |
| 2026-10-08 | E3b | read_dream_random | 1 | - | 0.015387 | cal=0.048058 sel=0.00454 | 对照（均匀回放） | full/formal2 |
| 2026-10-08 | E3b | read_dream_none | 0 | - | 0.074125 | cal=0.037767 sel=0.010347 | 无梦参考：var_head 校准即达 95–97%（无余量可补） | full/formal2 |
| 2026-10-08 | E3b | read_dream_none | 1 | - | 0.031219 | cal=0.048058 sel=0.005107 | 无梦参考：var_head 校准即达 95–97%（无余量可补） | full/formal2 |
| 2026-10-08 | E0 | dream_first+g0.3 | 7 | 280 | 0.882604 | -1.0333 | 流水线验证 | smoke/infratest |
| 2026-10-08 | E0 | dream_first+g0.3 | 7 | 280 | -1 | -1 | 流水线验证 | smoke/infratest |
| 2026-10-08 | E0 | wake_only+g0.3 | 0 | 5225 | 0.939097 | -0.638 | 流水线验证 | full/bench |
| 2026-10-08 | E0 | wake_only+g0.3 | 0 | 5225 | 0.9995 | -0.338 | 流水线验证 | full/bench |
| 2026-10-09 | E0 | wake_only+g0.3 | 0 | 5225 | 0.923355 | -0.236 | R1 ❌ 比值 0.84/0.79 | full·cuda/r1-dh64 |
| 2026-10-09 | E0 | wake_only+g0.3 | 1 | 5381 | 0.892143 | 0.358 | R1 ❌ 比值 0.84/0.79 | full·cuda/r1-dh64 |
| 2026-10-09 | E0 | dream_first+g0.3 | 0 | 5225 | 0.290452 | -0.5 | R1 ❌ 比值 0.84/0.79 | full·cuda/r1-dh64 |
| 2026-10-09 | E0 | dream_first+g0.3 | 1 | 5381 | 0.311721 | -0.27 | R1 ❌ 比值 0.84/0.79 | full·cuda/r1-dh64 |
| 2026-10-09 | E0 | alternate+g0.3 | 0 | 5225 | 0.849588 | -0.688 | R1 ❌ 比值 0.84/0.79 | full·cuda/r1-dh64 |
| 2026-10-09 | E0 | alternate+g0.3 | 1 | 5381 | 0.779755 | -0.308 | R1 ❌ 比值 0.84/0.79 | full·cuda/r1-dh64 |
| 2026-10-09 | E0 | wake_only+g0.3 | 0 | 5225 | 0.981218 | -0.5 | R1 ❌ 比值 0.84/0.68 | full·cuda/r1-dh256 |
| 2026-10-09 | E0 | wake_only+g0.3 | 1 | 5381 | 0.987766 | -0.346 | R1 ❌ 比值 0.84/0.68 | full·cuda/r1-dh256 |
| 2026-10-09 | E0 | dream_first+g0.3 | 0 | 5225 | 0.965057 | -0.21 | R1 ❌ 比值 0.84/0.68 | full·cuda/r1-dh256 |
| 2026-10-09 | E0 | dream_first+g0.3 | 1 | 5381 | 0.973607 | 0.076 | R1 ❌ 比值 0.84/0.68 | full·cuda/r1-dh256 |
| 2026-10-09 | E0 | alternate+g0.3 | 0 | 5225 | 0.954778 | -0.5 | R1 ❌ 比值 0.84/0.68 | full·cuda/r1-dh256 |
| 2026-10-09 | E0 | alternate+g0.3 | 1 | 5381 | 0.96086 | 0.234 | R1 ❌ 比值 0.84/0.68 | full·cuda/r1-dh256 |
| 2026-10-09 | E0 | wake_only+g0.3 | 0 | 5225 | 1.001823 | -0.5 | R1 ❌ 比值 0.84/0.53 | full·cuda/r1-dh1024 |
| 2026-10-09 | E0 | wake_only+g0.3 | 1 | 5381 | 1.015225 | -0.228 | R1 ❌ 比值 0.84/0.53 | full·cuda/r1-dh1024 |
| 2026-10-09 | E0 | dream_first+g0.3 | 0 | 5225 | 0.993221 | -0.212 | R1 ❌ 比值 0.84/0.53 | full·cuda/r1-dh1024 |
| 2026-10-09 | E0 | dream_first+g0.3 | 1 | 5381 | 0.996155 | -0.368 | R1 ❌ 比值 0.84/0.53 | full·cuda/r1-dh1024 |
| 2026-10-09 | E0 | alternate+g0.3 | 0 | 5225 | 0.98764 | -0.216 | R1 ❌ 比值 0.84/0.53 | full·cuda/r1-dh1024 |
| 2026-10-09 | E0 | alternate+g0.3 | 1 | 5381 | 0.998236 | -0.88 | R1 ❌ 比值 0.84/0.53 | full·cuda/r1-dh1024 |
| 2026-10-09 | E0 | dream_first+g0.3 | 0 | 5225 | 0.290452 | -0.5 | R1b ✅ 0.41/0.39 复现（同剂量确定性验证） | full/r1b-dh64 |
| 2026-10-09 | E0 | dream_first+g0.3 | 1 | 5381 | 0.311721 | -0.27 | R1b ✅ 0.41/0.39 复现（同剂量确定性验证） | full/r1b-dh64 |
| 2026-10-09 | E0 | dream_first+g0.3 | 0 | 5225 | 0.959861 | -0.44 | R1b ❌ 0.88/0.79 剂量×4 无改善 | full/r1b-dh256 |
| 2026-10-09 | E0 | dream_first+g0.3 | 1 | 5381 | 0.968525 | -0.088 | R1b ❌ 0.88/0.79 剂量×4 无改善 | full/r1b-dh256 |
| 2026-10-09 | E0 | dream_first+g0.3 | 0 | 5225 | 1.00992 | -0.62 | R1b ❌ 剂量×16 全程未达标（更差） | full/r1b-dh1024 |
| 2026-10-09 | E0 | dream_first+g0.3 | 1 | 5381 | 1.029312 | 0.174 | R1b ❌ 剂量×16 全程未达标（更差） | full/r1b-dh1024 |

登记说明（2026-10-11 起）：本表登记新增实验；(a)/(b) 之前的 R5-cal 全链（v1 30cyc 15 份、
v2.2 阶梯、排除链五大假设）以 ITER7-WORLD-FIRST.md §2.1–2.4 表格 + experiments/r5_world_calibration/
全量 json（40 份）为准，不逐行补登。日期=run 的 UTC 日。

| 2026-10-11 | R5-cal(a) | k0/dh1024/linear 60cyc | 0 | ~2.5K | 1.02962 | 三 cell 对照参照 | ❌ (a) 线：cell 变体全灭（最佳改善 1.2%） | full·cuda/celltest s0 |
| 2026-10-11 | R5-cal(a) | k0/dh1024/gated 60cyc | 0 | ~2.9K | 1.08062 | — | ❌ (a) 对照（更差 +5%） | full·cuda/celltest s0 |
| 2026-10-11 | R5-cal(a) | k0/dh1024/twolayer 60cyc | 0 | ~1.9K | 1.01732 | — | ❌ (a) 对照（+1.2% 远未治愈） | full·cuda/celltest s0 |
| 2026-10-11 | R5-cal(b) | k0/dh1024/none 120cyc | 0 | ~2.5K | 1.02962 | α→0.016 门坍缩；hN→1.00 冻结 | ❌ (b) 参照（与 (a) 逐位一致，确定性 ✓） | full·cuda/b-line s0 |
| 2026-10-11 | R5-cal(b) | k0/dh1024/ln_post 120cyc | 0 | 14.8K | 0.16566 | α 谷→0.999；hN 1.00→0.65 | ✅ 治愈（≤0.30；间隙闭合 96%） | full·cuda/b-line s0 |
| 2026-10-11 | R5-cal(b) | k0/dh1024/rms_post 120cyc | 0 | 14.8K | 0.19745 | α≈0.2–0.8；阴步幅 0.008→0.56 | ✅ 治愈（≤0.30） | full·cuda/b-line s0 |
| 2026-10-11 | R5-cal(b) | k0/dh1024/ln_pre 120cyc | 0 | ~2.0K | 1.03551 | α→0.016（同 none 坍缩） | ❌ 无效→病根在写侧非 cell 输入 | full·cuda/b-line s0 |
| 2026-10-11 | R5-cal(b) | k0/dh256/ln_post 120cyc | 0 | 14.8K | 0.11465 | — | ✅ ladder 悬崖压平（none 0.957→0.115） | full·cuda/b-ladder s0 |
| 2026-10-11 | R5-cal(b) | k0/dh64/ln_post 120cyc | 0 | 14.8K | 0.17508 | — | ◐ 健康档轻损（none 0.134→0.175 同量级） | full·cuda/b-ladder s0 |
| 2026-10-11 | R5-cal(b) | k0/dh256/rms_post 120cyc | 0 | 14.8K | 0.13315 | — | ✅ 交叉验证 ≈64 档 | full·cuda/b-ladder s0 |
| 2026-10-11 | R5-cal(b) | k0/dh1024/ln_post seed1 | 1 | 15.0K | 0.17040 | — | ✅ 复现稳健（s0 0.16566，轨迹逐段重合） | full·cuda/b-rep |
| 2026-10-11 | R5-cal(b) | k0/dh1024/rms_post seed1 | 1 | 15.0K | 0.37153 | s1 掉回坍缩陷阱（α 谷 0.004 ~70cyc）晚期自救 | ◐ 种子敏感（s0 0.19745）——主解=ln_post | full·cuda/b-rep |

登记说明（2026-10-12 夜三/P2/P4/P3）：长训 run 的逐 cycle 记录以
experiments/night3_long/*.jsonl 为权威（每行含 mse/α/hN/阴阳步幅/sat），
下表只登终值与判定。**量纲警告：长训收敛后编码器目标尺度跨 run 差 11×，
原始 MSE 跨配置不可直比；判定以解码 R²（obs 空间）为主标尺。**

| 2026-10-12 | E-长训 | k4/dh1024/ln_post 9000cyc | 0 | 12.18M | 0.01680 | α→1.0 hN 0.17 sat 0.86 | 富世界海量余量：120cyc"平台"0.138 的 8× 下方 | full·cuda/night3-E |
| 2026-10-12 | P2 | k0/dh64 9000cyc | 0 | 12.2M | 0.07642 | dec R² 84.1%（拷贝参照 89.6%） | 简单世界：64 弱于 256 | full·cuda/p2 |
| 2026-10-12 | P2 | k0/dh256 9000cyc | 0 | 12.2M | 0.02457 | dec R² 89.0% | 简单世界 256 稳 | full·cuda/p2 |
| 2026-10-12 | P2 | k0/dh1024 9000cyc | 0 | 1.11M | 0.41502 | c500 曾 0.118；hN 0.30→0.64 | ❌ 系统性近期化漂移（s1 复现 0.373 同带）→ only_last=8 协议 | full·cuda/p2 |
| 2026-10-12 | P2 | k2/dh256 9000cyc | 0 | 7.5M | 0.00947 | — | 全矩阵最低 MSE | full·cuda/p2 |
| 2026-10-12 | P2 | k2/dh1024 9000cyc | 0 | 12.2M | 0.01605 | — | 与 256 同水平 | full·cuda/p2 |
| 2026-10-12 | P2 | k4/dh64 9000cyc | 0 | 12.2M | 0.02084 | — | 富世界 64 亦可用 | full·cuda/p2 |
| 2026-10-12 | P2 | k4/dh256 9000cyc | 0 | 12.18M | 0.01196 | **dec R² 92.6%**（拷贝 29.8%） | 主档；s1 复现 1.01× | full·cuda/p2 |
| 2026-10-12 | P2 | k4/dh1024 9000cyc | 0 | 12.18M | 0.01680 | dec R² 93.2% | 与 256 同水平（s1 0.73×） | full·cuda/night3-E |
| 2026-10-12 | P2 | k4/dh2048 9000cyc | 0 | 12.18M | 0.02187 | dec R² 90.2%；c100 0.40→c500 0.0094 | 2048=慢启动非陷阱；略逊 256/1024 | full·cuda/p2 |
| 2026-10-12 | P4 | 桥 k4/256 c120/c1000/c4000 | 0 | 0.15M/1.25M/5.0M | 0.087/0.019/0.011 | R² −48/64/57%；dec R² 51/87/92.7% | 解码保真 1000cyc 冲 87%、4000 饱和 93% | full·cuda/p4 |
| 2026-10-12 | P4 | 桥 k4/1024 c120 | 0 | 0.15M | 0.04990 | R² −104%；dec 60% | 120cyc=未站上目标流形 | full·cuda/p4 |
| 2026-10-12 | P4 | 桥 k0/256 c120 | 0 | 0.15M | 0.08032 | R² −140%；dec 44% | 同上（k0 拷贝参照即 89.6%，k0 桥意义有限） | full·cuda/p4 |
| 2026-10-12 | P3 | k4/dh256 s1 9000cyc | 1 | 12.18M | 0.01208 | — | ✅ 1.01× 复现 | full·cuda/p3 |
| 2026-10-12 | P3 | k4/dh1024 s1 9000cyc | 1 | 12.18M | 0.01222 | — | ✅ 0.73×（s0 尾段轻漂） | full·cuda/p3 |
| 2026-10-12 | P3 | k0/dh1024 s1 9000cyc | 1 | 1.11M | 0.37343 | 两种子同落 0.37–0.42 | 漂移系统性确认（协议问题非噪声） | full·cuda/p3 |
| 2026-10-12 | P3 | k2/dh256 s1 9000cyc | 1 | 7.48M | 0.02561 | — | ◐ 2.70× 种子敏感 | full·cuda/p3 |

登记说明（2026-10-13 P0）：iter8 仪器修复轮。A3 重做判定表见 ITER8-PLAN §6；
下表只登 wake 终值与梦比（门 ≤0.6）。

| 2026-10-13 | P0-A | k0/dh1024/mixed 9000cyc | 0 | 1.11M | 0.01055 | hN 0.136 无漂 | ✅ 混合回放治愈近期化漂移（门 ≤0.20 的 19× 下方；recent 协议 0.415） | full·cuda/p0a |
| 2026-10-13 | P0-B | dh64 wake/dream/alt ×2seed | 0,1 | 316K | 0.051/0.043 | 梦 0.58✅/0.69❌ alt 0.56✅/0.62❌ | ◐ 边界 regime（种子分裂，按 E0 惯例不判定） | full·cuda/a3redo |
| 2026-10-13 | P0-B | dh256 wake/dream/alt ×2seed | 0,1 | 316K | 0.018/0.019 | 梦 未达标/0.74❌ | ❌ 梦增益未恢复 | full·cuda/a3redo |
| 2026-10-13 | P0-B | dh1024 wake/dream/alt ×2seed | 0,1 | 316K | 0.018/0.016 | 梦 0.77❌/0.98❌ alt 均未达标 | ❌ 梦增益未恢复——**容量受限衰减在健康模型+收敛协议下重新成立** | full·cuda/a3redo |
| 2026-10-13 | P0-C | ENV-E/P0 pilot s0/s1 | 0,1 | 6.3M | τ50=120/312 | m1@switch≈收敛；ρ0 保持 −0.006/−0.012 | slip-only 预测层太温和→原 48-run 矩阵否决；ENV-E v2=hazard 布局+策略层 | full·cuda/env-e-p0 |
| 2026-10-13 | P0-D | pilot v1 MPC 回报 s0/s1 | 0,1 | - | m0@flip 0.07/0.11 | 翻转后 m0/m1 躺平 −0.5 | ❌ 仪器无效：MPC 从未形成稳健奔目标行为（行为层读数不适用 v0） | full·cuda/env-e-p0d |
| 2026-10-13 | P0-D | pilot v2 全转移 MAE s0/s1 | 0,1 | - | e1@flip 0.037/0.043 | e1_end 0.037/0.043（纹丝不动） | ❌ 仪器无效：终局 ±1 仅占 0.8%，信号被步进背景稀释（Δ0.008<噪声） | full·cuda/env-e-p0d |
| 2026-10-13 | P0-D | pilot v3 终局条件化 s0/s1 | 0,1 | - | acc1@flip 0.29/0.52 | 训练 regime 终局符号准确率仅 0.55–0.59 | ◐ 仪器成功但暴露上游瓶颈：±1 太稀疏，A/B 语义学不扎实——ENV-E 矩阵后延至垫高基础现象 | full·cuda/env-e-p0d |
| 2026-10-13 | K56 | k5 dh64/256/1024 9000cyc mixed | 0 | 13.6M×3 | dec R² 79.1/94.6/95.2% | rank90=9@1024 | ❌ 门未过（256→1024 +0.6pp）但 64→256 梯度 +15pp（深结构压小身体） | full·cuda/k56 |
| 2026-10-13 | K56 | k6 dh64/256/1024 9000cyc mixed | 0 | 13.6M×3 | dec R² 71.8/86.7/**90.1%** | 梯度单调放大 | ❌ 门未过（+3.4pp=三档最大残差）——对抗深度持续购买宽度价值，k7 候选；身体冻结维持 | full·cuda/k56 |
| 2026-10-14 | K7 | k7 dh64/256/1024 9000cyc mixed | 0 | 10.1M×3 | dec R² 45.7/84.3/87.4% | dh64 rank90=2 崩 | ❌ 门未过（+3.1pp）——四世界趋势饱和，ENV-D 宽度天花板判定；转 ENV-E | full·cuda/k7 |
| 2026-10-14 | P0-D4 | 垫高双引导 s0/s1（fracA 0.3/fracB 0.15） | 0,1 | - | acc0@flip 0.542/0.506 | 平衡曝露无效（vs 失衡 0.554） | ❌ 不可建分支生效——ENV-E 线降级存档；器官触发门阻断 | full·cuda/p0d4 |
| 2026-10-14 | ITER9 term-scale | 容量阶梯+λ_rew 五臂×2seed×2400cyc（+T64 升级臂 4800） | 0,1 | - | acc_end T64 .860/.849→升级 .925；W64 .957/.886；T256 .865/.892；W256 .866/.849；T1024 .877/.924 | 过门周期≈2123→1113(W64)/1008(T256)/1155(W256)/390(T1024)；锚点双 seed 逐位复刻 D4；H3 无偏置 | ✅ 预算假说确认——P0-D4"不可建"=600cyc/16K 读数天花板（同协议 d_h1024 @600 过门）；容量 5.4×/加权 2×@16K 两旋钮不叠加 | full·cuda/term_scale |
| 2026-10-15 | ITER10 dream-sem | dream-first×{64,256,1024}×s0/s1（+64 档 s2 扩展对） | 0,1,2 | - | 过门周期 dream/wake：64=1385-1540/1850-2395；256=860-955/930-1085；1024=380-515/365-415 | r(64)=0.71◐（三seed方向一致）；r(256)=0.91、r(1024)=1.16；梦稳压（64 跨155 vs 545） | ◐ H-A 部分成立——容量受限档加速方向成立（~1.4×、未达 E0 0.60 阈）、高容量档零效应；调度×容量交互存在、幅度弱；staging 门=候选排①后 | full·cuda/dream-sem |
| 2026-10-15 | ITER11 recovery | 单翻转恢复 {wake,dream}×s0/s1（PRE 2400 建立→翻转→POST 2400） | 0,1 | - | acc1_end 0.954/0.961/0.960/0.944；τ50 25/70/25/55；τ90 190/110/90/215 | 翻转点 acc1≈0.05-0.16（反号如预期）；PRE 段 481/481×4 与归档逐位锚定；保持面全 0.04-0.06 | ◐→实质无差异：**恢复快一个数量级（τ50 25-70 vs 建立 1400-2400）且对调度不敏感——梦=出生优惠非再读优惠**；r_τ50=0.89；Stage 2 λ 阶梯解锁（dwell{750,150,50}） | full·cuda/env-e-v3 |
| 2026-10-16 | ITER12a | λ 阶梯首矩阵（dwell 步制——单位错误版，12 run） | 0,1 | - | 全臂 acc_sw 0.49-0.59（chance）、无梯度 | 开关计数 585/2850/8451 → 校准 176-183 步/周期；PRE 锚定 12/12 满分 | ❌ 协议实现错误（dwell 单位=步 vs τ 单位=周期，差 ~180×）——矩阵作废留档；副产品=平凡域下限对照（dwell≪τ → chance 且 15× 范围平坦） | full·cuda/env-e-v3 |
