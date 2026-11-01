# 规范打包模型 · 来源与配方（MODEL-PROVENANCE）

**checkpoint**：`mcp/somnium-canonical-dh64.pt`（84KB，16K 参数档，随仓库在档）
**训练入口**：`experiments/mcp_pack/train_canonical.py`
**性质**：打包产物生产（非预登记实验，不入 registry）；配方如实如下，全部可复现。

## 配方（E0 dream_first FULL 无-MPC 变体）

| 项 | 值 |
|---|---|
| 世界 | `LatentGrid(grid=8, slip=0.1, p_threat_move=0.02, horizon=50)`，d_obs=8，n_actions=4 |
| 模型 | `WorldModel(d_obs=8, n_actions=4, d_h=64, norm_kind=none)`，15,920 参数 |
| 日程 | 梦期先行：空缓冲内生想象 200 批 → 醒期 300 cycles（每 cycle 10 局 guided 0.3 + 10 wake 批，recent 采样） |
| 优化 | Adam lr=1e-3，batch=32，t_win=8，buffer_cap=2000 |
| seed | 0（环境/模型/批采样全种子化） |
| 评估 | 固定 eval 集（seed+999，20 局随机策略） |

与论文 E0 的两处工程差异（不影响配方公平性，见训练入口 docstring）：去掉 MPC 回放探头（CPU 打包场景非必需）；评估追加解码 R²。

## 终评（checkpoint extra 内嵌完整历史）

- 潜态预测 MSE（E0 主指标协议，锁 4242）：**0.0692**
- 解码 R²（对拷贝参照）：**0.872**（teacher-forced 末步重建）
- 总真实交互步数：79,480

## 边界（引用 MODEL-CARD 口径）

本 checkpoint 是 **grid8 符号世界的研究原型**：不适用视觉/文本/生产 RL/长记忆；一步观测预测在本尺度不敌参数匹配 AR（论文 §5.4 诚实双负）；梦的适用域=出生期（五线证据）。MCP 工具会如实报告这些边界（如 observe 的负 R²）。

## 复现

```bash
.venv/bin/python experiments/mcp_pack/train_canonical.py --cycles 300 --seed 0
# ≈2 分钟（CPU）；输出 mcp/somnium-canonical-dh64.pt（自描述：extra.model_config 可独立重建）
```
