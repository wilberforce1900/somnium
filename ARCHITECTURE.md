# 底座架构 · v0.2（2026-10-08）

最原始工程架构：**单底座 + 双模式算子 + 门控**，上叠世界模型组装层、梦期调度器与 E0 实验入口。
对应 PRINCIPLES §2（L0）、§3（L1）、§4（梦）、§7（L2 思）；ROADMAP §1.0。

## 设计铁律

阴阳 = 同一底座上的两种**动力学模式**，不是两个网络（TRM 教训）。
yin.py 与 yang.py 是两个模式算子模块，共享同一个 Substrate 实例：
参数互根（同体），软门混合（太极），绝不各自持有参数。
"两个独立子网络"的变体不丢弃——它是 A2 消融的对照组，已内建为
`taiji.gate_mode ∈ {fixed_yang, fixed_yin}` 开关。

## 结构图

```
              ┌────────────────────────────────┐
              │ substrate.py   ★ 公共底座      │
              │  LatentState(h, t)   潜态束    │
              │  cell      正向动力学核（阳用）│
              │  energy   能量地形（阴用）     │
              │  spawn/inject  混沌入口 (L0)   │
              └─────────┬──────────────────────┘
               共享引用 ↓          ↓ 共享引用
        ┌───────────────┐   ┌───────────────┐
        │ yang.py ★ 阳  │   │ yin.py  ★ 阴  │
        │ step/stream   │   │ step/relax    │
        │ 快·发散·流式  │   │ 慢·收敛·弛豫  │
        └──────┬────────┘   └────────┬──────┘
               └────────┬───────────┘
                  taiji.py ★ 太极门控
          h ← h + α·Δ阳 + (1−α)·Δ阴   （软混合：互根）
```

数据流（一步）：观测 x → 门控 α=σ(g([h,x])) → 并行算 Δ阳、Δ阴 → 软混合更新 h。
阴步经 `create_graph` 把梯度送回共享 energy_head——整条轨迹对全部共享参数可微（BPTT 全图）。

## 必需 / 可扩展标识

图例：**★ 必需**（v0 已实现并有测试）｜ **◇ 可扩展**（接口已留，未实现）｜ ○ 规划（未动工）

| 模块 | 职责 | PRINCIPLES | 状态 | 扩展点 |
|---|---|---|---|---|
| `src/substrate.py` | 潜态束 + 共享主干 + 混沌入口 | §2 L0, §3 | ★ | cell→线性注意力；潜态加卦码槽；sigma 课程化 ◇ |
| `src/yang.py` | 阳算子：流式发散展开 | §3 | ★ | 展开深度课程化；并行流式 ◇ |
| `src/yin.py` | 阴算子：能量弛豫收缩 | §3 | ★ | Hopfield 闭式吸引子；DEQ 隐式微分；早停 tol 已留 ◇ |
| `src/taiji.py` | 太极门控：软混合 + 消融开关 | §3, §11 | ★ | 梦醒宏节律 schedule；门控上下文加能量/卦码 ◇ |
| `src/core.py` | 世界模型组装：encoder+底座+门控+奖励/obs 接地头+MPC 探头；JEPA 式潜态预测（编码器直出潜态空间） | §7, ROADMAP §1.0 | ★ | EMA 目标；actor-critic；卦码头；d_h 放大 ◇ |
| `src/dream.py` | 梦期调度器（四开关 = A4 消融面）+ EpisodeBuffer + DreamLog | §4 | ★ | EMA 目标；能量自适应强度；卦码统计（log 已留字段）◇ |
| `src/envs/latent_grid.py` | ENV-A：可控动力学（slip）+ 稀有威胁事件 | ROADMAP §1.0 | ★ | 局部视野、多 hazard、固定起点课程 ◇ |
| `src/envs/maze.py` | ENV-C：完美迷宫（DFS 生成，墙位观测，BFS 难度锚） | ROADMAP §1.2 | ★ | 固定迷宫池课程、钥匙门 ◇ |
| `src/envs/` ENV-B | arc_lite：ARC-AGI-1 固定子集 | ROADMAP §1.0 | ○（网络受限暂缓） | ARC 数据集需外网，待镜像/手动导入 |
| `src/codebook.py` | L3 六十四卦码本：VQ 量化（先天卦序二进制）+ 转移图统计 + 死码重生 + 价值迭代 | §5 | ★ | 可学习转移网络；EMA 计数；King Wen 显示层；卦码检索记忆 ◇ |
| `src/planner.py` | 图上慢规划（价值迭代+单步想象落卦）+ free/free_match 对照 | §5 | ★ | beam/MCTS 深搜索；多步图路径投影 ◇ |
| `src/metrics.py` | 校准 / 保持率 / 选择性曲线 | §8 | ○ | |
| `experiments/e0_dream_first/` | E0 实验入口（三日程 + registry 登记） | ROADMAP §1.1 | ★（smoke 通过） | 正式全量跑（--full）待预注册锁定后 |
| `experiments/e1_yinyang_gate/` | E1 实验入口（2 任务 × 3 门控模式 × 弛豫深度；α-难度相关性分析） | ROADMAP §1.2 | ★（smoke 通过） | 正式 ≤8 run 待跑；fixed_yin 为机制对照非竞争基线 |
| `experiments/e2_hexagram_codebook/` | E2 实验入口（K∈{0,32,64,128} × planner∈{graph,free,free_match}；卦码轨迹 dump） | ROADMAP §1.3 | ★（smoke2 通过） | 正式 ≤12 run 待跑；训练日程暂 wake_only（待 E0 判定）；smoke 观察：free 单环 0.17 vs graph 0——单步落卦打分弱，正式跑量化 |
| `results/registry.md` | Run 登记表（负结果照记） | ROADMAP §0 | ★ | |

代码内同步标识：各模块 docstring 首部带 ★/◇ 标记，与本表一一对应。

## 消融开关（开箱即用）

A2（阴阳门控，PRINCIPLES §11）：`TaijiGate(gate_mode=...)`
`learned` vs `fixed_yang` vs `fixed_yin`——同一底座天然等参数量，无需配平。

## 螺旋进化接口（留白清单，2026-10-08）

本项目为迭代进化预留的窗口，分三类：

**1. 判定→重构回路（架构层面的螺旋，已在转动）**
消融登记册 A1–A6 + 预注册降级路径 = 架构的进化选择机制：假说被证伪时组件
降级而非删除（A2 阴阳→调度开关；A1 易→可解释层），被证实时升格为主线
（A3/A4 梦机制）。阶段 0 首圈已经转完一轮——这就是螺旋的第一个实体。

**2. 显式接口/开关（代码内 ◇ 留白）**
- `gate_mode="schedule"`：宏节律挂点（梦醒课程、未来发育时间表接入处）
- `substrate.inject/noise`：梦期噪声与反向学习的物理入口（强度课程化留白）
- `extra_relax`：阴弛豫深度（递归深度维度留白）
- `WorldModel.codebook`：码本可插拔（已降级为只读观察器，EMA/只读重审入口）
- `DreamLog.hexagram` 字段：卦码报表占位（可解释层复活入口）
- `WorldModel.var_head`：L5 智层第一粒种子（校准头，E3b 引入）
- `ObsCoverage` + `read_dream`：**首个闭环反馈**——梦的产物被读取，反过来
  改变下一晚的梦（读梦→盲区→偏置想象）。这是字面意义的螺旋。

**3. 未动的留白（阶段 1+）**
概念流放大（d_h→1–8M、EMA 目标编码器）、语言 I/O 编解码器、
多 agent 共享态势板、数术先验层（A6 未跑）、反向学习强度自适应。

**4. 进化目标约束（用户原则，2026-10-08）**
螺旋的轴不是"能力无限上旋"，而是**能力×匹配度**的共同进化——刚刚好
原则（PRINCIPLES §8.5）：进化终点不是让人仰望的神，是与交互对象认知
同步的伴侣。未来接口：partner-paced 课程、用户认知模型 agent（阶段 2）、
σ 门控的解释粒度调节。


## 环境（本机现实）

- Intel x86_64 macOS：torch 二进制上限 2.2.2（CPU），需 Python ≤ 3.12（brew python@3.12）。
- Apple Silicon 机器可放开到新 torch 并用 MPS（`substrate.default_device()` 已自动降级）。

## 运行测试

    cd /Users/willsmacbookpro/Documents/dsh/mission_yin_yang
    .venv/bin/python -m pytest tests/ -q
