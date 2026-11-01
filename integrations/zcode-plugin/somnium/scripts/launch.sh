#!/bin/bash
# Somnium MCP 启动器（ZCode 插件用）：解析 Somnium 仓库位置并拉起 mcp/server.py。
# 仓库定位优先级：SOMNIUM_REPO 环境变量 > ~/somnium。要求该仓库 .venv 含 torch
# （见仓库 integrations/README.md §0 前置）。
set -u
REPO="${SOMNIUM_REPO:-$HOME/somnium}"
PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || true)"; fi
if [ -z "${PY:-}" ] || [ ! -f "$REPO/mcp/server.py" ]; then
  echo "somnium plugin: 未找到 Somnium 仓库（设 SOMNIUM_REPO=<克隆路径>，" \
       "并确保 .venv/bin/python 含 torch）。当前尝试: $REPO" >&2
  exit 1
fi
exec "$PY" "$REPO/mcp/server.py"
