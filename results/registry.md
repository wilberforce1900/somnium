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
