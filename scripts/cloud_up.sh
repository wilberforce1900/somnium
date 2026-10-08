#!/bin/sh
# 上行：本机仓库 → 云 VM（排除环境/缓存/内部档）。用法：sh scripts/cloud_up.sh <user@host>
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
rsync -avz \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
  --exclude '*.pt' --exclude '*.tmp' \
  "$R/" "$1":~/mission_yin_yang/
