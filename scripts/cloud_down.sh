#!/bin/sh
# 下行：云 VM 的实验结果 → 本机合并。用法：sh scripts/cloud_down.sh <user@host>
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
rsync -avz "$1":~/mission_yin_yang/experiments/ "$R/experiments/" \
  --include '*/' --include 'details_*' --include '*.json' --include '*.jsonl' --exclude '*'
# registry 单独拉成副本防覆盖本地历史，行手工核对后合并
rsync -avz "$1":~/mission_yin_yang/results/registry.md "$R/results/registry_cloud.md"
echo "已回传。registry_cloud.md 为云端新增行，核对后合并进 registry.md。"
