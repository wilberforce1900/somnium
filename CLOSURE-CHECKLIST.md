# Somnium 结项清单（CLOSURE-CHECKLIST）

> **目标**：ITER14 收割后按此清单走完"清理 → 定稿 → 封装 → 结项"阶段闭环。
> 状态标记：☐ 待办 / ◐ 进行中 / ☑ 完成。

## A. 实验收口

- ☑ ITER14 预登记+实现+测试全本机（a4423b5，179 测试绿）
- ☐ ITER14 两站执行（VM，run_14.sh，2026-10-19 09:14 UTC 点火）
  - ☐ 站 1 R4b 多步开环 ×2 seed → §1-R 判决（k16 Δ_r2 ±5pp 门）
  - ☐ 站 2 读梦 v2 三臂 ×2 seed → §2-R 判决（G1 +0.20 / G2 +0.10 配对门）
  - ☐ registry 2 行 + 数据入库 + 提交推送
- ☐ staging（DREAM-STAGING-NOTE）：结项裁定=永久暂缓（三连空证据），笔记留档

## B. VM 清理（收割+验证后）

- ☐ 数据对账：VM jsonl/final 与本地+GitHub 逐一对账（ckpt/种子库=确定性可重生成）
- ☐ 删除：`experiments/night3_long/ckpts`（17G）+ `overnight_autodriver/out`（3.5G）
  → 磁盘 91% → ~55%
- ☐ 保留：repo / venvs / 日志 / tectonic（v1.5 编译用）
- ☐ v1.5 发布后：最终形态建议=删机（代码唯一权威源=GitHub）

## C. 论文 v1.5（终版定稿）

- ☐ 内容：R4 外基线 + R4b 开环终局 + R2 校准余量/读梦 v2 终局 + R3 遗忘三节浓缩；
  参数公式勘误（15,912→16,048，+0.85%）；"实验总计行"更新（run 数/GPU 时/成本）
- ☐ 编译：VM tectonic；预计 14-15 页
- ☐ 视觉验收：visual-judge（配额 10-19 16:48 后恢复）全卷独立复验
- ☐ 发布链：GitHub 提交+Release 资产/notes → Zenodo v1.5 上传（用户）→ 版本 DOI 回填
- ☐ README 终版（含结项说明+完整版本链 v1.0→v1.5）

## D. 封装与开源面

- ☑ MODEL-CARD.md（参数表/已测边界/适用/不适用/局限/引用）
- ☐ 模型卡复核：数字与 registry 终版一致（ITER14 后如有更新）
- ☐ 卫生终检：test_repo_hygiene + tarball 个人标识 grep=0（沿 v1.4 流程）
- ☐ GitHub 公开面终检：README/MODEL-CARD/Release 资产完整可导航

## E. 结项归档

- ☐ 结项说明（README"结项"节或独立 CLOSURE.md）：完成弧线（E0→ITER14，
  ~60 预登记 run 族）+ 未完成项及理由：
  - 记忆器官（触发门未亮——世界轴先饱和）
  - 梦调节器（触发门阻断——ENV-E 调度不敏感）
  - staging 分阶（三连空，永久暂缓）
  - 网格外环境/更大尺度（$150 预算研判：留储备不烧）
  - R4b 若负：多步开环升级臂（登记不自动开）
- ☐ 记忆终版：项目记忆+索引行改为"🏁 结项"态（保留复活触发器描述）
- ☐ VM 删机（用户操作，B 完成后）
- ☐ （可选）作品集整合：tabghach org 展示页挂 somnium

## F. 复活触发器（结项后重启项目的条件）

- 任何单条满足即值得重开：①记忆器官门亮（长 horizon 任务 dec R² 显著劣化）
  ②出现调度敏感的新读数 ③网格外环境（MiniGrid 级）预算到位 ④论文外部反馈
  需要补实验。
