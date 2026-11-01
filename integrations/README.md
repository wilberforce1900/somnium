# Somnium MCP · 跨 harness 接入

**一个服务器，六个 harness。** 所有主流本地 agent harness 都支持 stdio MCP——不需要为每个 harness 写插件，只需要各自的配置片段。核心服务器见 [mcp/server.py](../mcp/server.py)（协议 2025-06-18，零第三方依赖，仅 torch）。

## 0. 通用前置（所有 harness 共用）

```bash
git clone https://github.com/wilberforce1900/somnium.git
cd somnium
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python mcp/server.py   # 启动自检：stderr 应输出 somnium-mcp ready
```

下述配置中的 `<REPO>` 一律替换为你的克隆路径（绝对路径）。checkpoint 随仓库在档（`mcp/somnium-canonical-dh64.pt`，84KB，d_h=64，16K 参数档）。

## 1. 各 harness 配置

### ZCode（✅ 原生插件化进行中）
设置 → MCP Servers，或 `~/.zcode/cli/config.json` 的 `mcp.servers`：

```json
{ "somnium": { "command": "<REPO>/.venv/bin/python", "args": ["<REPO>/mcp/server.py"] } }
```

验证：会话内应出现 `mcp__somnium__somnium_info` 等五个工具；调用 `somnium_info` 返回模型卡即通。
（正式 ZCode plugin 包装——skill/命令/manifest——见 roadmap，用 plugin-creator 构建。）

### Codex CLI（OpenAI）
`~/.codex/config.toml` 追加：

```toml
[mcp_servers.somnium]
command = "<REPO>/.venv/bin/python"
args = ["<REPO>/mcp/server.py"]
```

验证：`codex` 会话内要求调用 somnium_info；首次加载约 3–10s（torch import）。

### Cursor
`~/.cursor/mcp.json`（全局）或项目 `.cursor/mcp.json`：

```json
{ "mcpServers": { "somnium": { "command": "<REPO>/.venv/bin/python",
                               "args": ["<REPO>/mcp/server.py"] } } }
```

验证：Cursor 设置 → MCP 面板显示 somnium 绿点 + 5 tools。

### OpenCode
`opencode.json`（项目根或 `~/.config/opencode/`）：

```json
{ "mcp": { "somnium": { "type": "local",
                        "command": ["<REPO>/.venv/bin/python", "<REPO>/mcp/server.py"],
                        "enabled": true } } }
```

### DeepSeek DS Harness
按已验证的 `b_mcp_*` 条目模式，在 harness 的 cordis patch 清单中插入：

```yaml
- insert:
    - id: b_mcp_somnium
      command: <REPO>/.venv/bin/python
      args: [<REPO>/mcp/server.py]
```

> ⚠️ 端到端待核实：字段名以你版本 harness 的 dsh-mcp-client 实际 schema 为准（同机 Financial Watch 的 `b_mcp_*` 条目已验证此拉起模式可行）。

### WorkBuddy
无公开插件规范可核对（待核实）——若其支持 stdio MCP，用与 Cursor 相同的 JSON（`{"mcpServers": …}`）或其设置界面的 "Custom MCP / Add MCP Server" 入口填 command+args。验证标准与上同：能看到 5 个 `somnium_*` 工具。

## 2. 五个工具速览

| 工具 | 输入 | 返回 | 演示话术 |
|---|---|---|---|
| `somnium_info` | — | 模型卡（16K 参数、d_h64、训练终评） | "这是我论文里那个模型的身份证" |
| `somnium_episode` | steps/seed/slip | 真实 grid8 情节（同训练分布） | "先造一段这个世界里的真实经验" |
| `somnium_observe` | seq(+actions) | 逐步预测 MSE、对拷贝基线的 R²、终态潜态摘要 | "O(1) 状态一边看一边压缩成 64 维" |
| `somnium_imagine` | obs/steps/seed | 闭环想象滚动（自身解码回流）+ 不确定度 | "它在做白日梦——吃自己的解码想象未来" |
| `somnium_dream` | episodes/batches | 梦前/梦后巩固损失；默认更新内存权重 | "把今天的经验喂给它睡一觉，权重真的变了" |

## 3. 面向演示的推荐脚本（3 分钟）

1. `somnium_info` —— 模型来自一篇公开预印本（DOI 在模型卡里）；
2. `somnium_episode` → `somnium_observe` —— 展示 R² 与潜态摘要；
3. `somnium_imagine` steps=16 同 seed 两次 —— 确定性白日梦；
4. `somnium_dream` persist=true —— 对比梦前后 `somnium_observe` 的 MSE（梦后应下降）；
5. 诚实边界：一步预测可能不如平凡拷贝基线（论文 §5.4：本尺度 AR 更强）——工具会如实报告负 R²，这本身是卖点（测量诚实性）。

## 4. 状态表

| Harness | 配置就绪 | 端到端验证 |
|---|---|---|
| ZCode | ✅ | 待插件正式包装后跑通 |
| Codex | ✅ 片段 | 待用户实机 |
| Cursor | ✅ 片段 | 待用户实机 |
| OpenCode | ✅ 片段 | 待用户实机 |
| DS Harness (DeepSeek) | ⚠️ 模式已验证、字段待核对 | 待核实 |
| WorkBuddy | ⚠️ 通用 stdio | 待核实（无公开规范） |
| Claude Code | 暂缓（用户拍板） | — |
