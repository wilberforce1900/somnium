#!/bin/sh
# 下行：云 VM 的实验结果 → 本机合并。用法：sh scripts/cloud_down.sh <user@host>
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
rsync -avz "$1":~/mission_yin_yang/experiments/ "$R/experiments/" \
  --include '*/' --include 'details_*' --include '*.json' --include '*.jsonl' --exclude '*'
rsync -avz "$1":~/mission_yin_yang/results/ "$R/results/"
echo "已回传。注意：registry.md 在云端追加的行需手工核对合并（避免覆盖）。"
