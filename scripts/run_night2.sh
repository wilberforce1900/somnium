#!/bin/bash
# Somnium 夜二发射脚本（2026-10-10）：driver v2.1，完整 12 小时周期
# 用法（VM 上）: nohup bash ~/mission_yin_yang/scripts/run_night2.sh \
#                > ~/mission_yin_yang/results/night2.log 2>&1 &
PY=$HOME/venvs/torch/bin/python
cd ~/mission_yin_yang
END_UTC=$(date -u -d "+12 hours" +%Y-%m-%dT%H:%M:%S)
echo "=== NIGHT2 START $(date) 硬停 $END_UTC ==="
echo "磁盘: $(df -h / | tail -1)"
# 保护夜一种子库（已回传本地入库，VM 侧另存不覆盖）
if [ -d experiments/overnight_autodriver/out ]; then
  mv experiments/overnight_autodriver/out experiments/overnight_autodriver/out_night1_20261009
fi
mkdir -p experiments/overnight_autodriver/out
$PY experiments/overnight_autodriver/run_driver.py --end-utc "$END_UTC" --cycles 6
echo "=== NIGHT2 ALL DONE $(date) ==="
df -h / | tail -1
