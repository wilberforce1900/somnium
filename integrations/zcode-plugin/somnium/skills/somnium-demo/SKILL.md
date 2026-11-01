---
name: somnium-demo
description: 演示已发表的 Somnium 循环世界模型（somnium-mcp 五工具）：观测压缩进 O(1) 潜态、闭环想象（白日梦）、梦巩固真实更新权重，并如实报告模型边界。当用户想看/试 Somnium、做梦、想象、世界模型演示时使用。
---

# Somnium 演示脚本（3 分钟）

Somnium 是一篇公开预印本里的循环世界模型（O(1) 潜态、JEPA 式下一潜态预测、
"先梦后醒"日程）。本插件的 MCP 服务器把它变成五个可调用工具。演示按以下顺序
进行，每步向用户展示返回中的关键字段：

## 步骤

1. **somnium_info** — 模型卡：16K 参数、d_h=64、训练终评（解码 R²≈0.87）。
   指出 `recipe.provenance` 与论文 DOI（10.5281/zenodo.23479114）。
2. **somnium_episode**（steps=20, seed=42）— 生成一条真实 grid8 世界情节
   （与训练同分布）。展示 obs 是 8 维向量（agent/goal/hazard 坐标 + 时间）。
3. **somnium_observe**（seq=该情节, actions=其 actions）— O(1) 潜态逐步压缩。
   报告 `somnium_mse`、`copy_frame_mse`、`r2_vs_copy`、终态潜态摘要。
   **如实解读**：一步预测可能不敌拷贝基线（负 R²）——论文 §5.4 的诚实双负，
   工具照实报告，这本身是卖点。
4. **somnium_imagine**（obs=情节首帧, steps=16, seed=7）跑两次 — 两次输出
   逐位一致（确定性白日梦）；模型吃自己的解码滚动，无真值锚定。
5. **somnium_dream**（episodes=[该情节], batches=20, persist=true）—
   **梦真的更新权重**：对比梦前后 `consolidation_loss_before/after`（应下降），
   再跑一次 somnium_observe 看 `somnium_mse` 在梦过的情节上下降。
   提醒：persist=true 后服务器内存中的模型已改变（梦后模型≠梦前）。
6. **边界总结**（不要跳过）：本模型是 grid8 符号世界研究原型；一步预测本尺度
   不敌参数匹配 AR 基线；梦的适用域=出生期（五线证据，论文 §5.2–5.3）。

## 工具速查

| 工具 | 必填参数 | 返回要点 |
|---|---|---|
| somnium_info | — | params/d_h/recipe/final_eval |
| somnium_episode | steps≤200 | obs 序列 + actions |
| somnium_observe | seq（≥2 条 8 维） | mse/r2_vs_copy/final_latent |
| somnium_imagine | obs+steps≤32 | imagined_obs+uncertainty（同种子确定） |
| somnium_dream | episodes（1–8 条） | before/after/delta/model_persisted |

## 前置与排障

- 服务器由 `scripts/launch.sh` 拉起：需要 `SOMNIUM_REPO` 指向 Somnium 克隆
  （缺省 `~/somnium`），且该仓库 `.venv/bin/python` 装有 torch。缺失时 stderr
  会给出明确指引（harness 面板显示 MCP server failed 时先查这个）。
- 首次调用约 3–10s（torch 导入 + checkpoint 加载）。
