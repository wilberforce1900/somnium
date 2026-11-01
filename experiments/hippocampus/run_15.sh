#!/bin/bash
# ITER15 · 外部海马体三臂 × 3 seeds（预登记 ITER15-PLAN.md，门限跑前锁定）
# 用法（VM）：git pull 后  nohup bash experiments/hippocampus/run_15.sh > ~/iter15.log 2>&1 &
# 预计 9 runs ≈ 2–3h（L4）。smoke 先行：SMOKE=1 bash run_15.sh（tiny 配置自检）。
set -u
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-.venv/bin/python}
VARIANT=${VARIANT:-a}   # a=ITER15（ρ 交替）；b=ITER15b（镜像冲突）
SUFFIX=""; [ "$VARIANT" = "b" ] && SUFFIX="_b"

if [ "${SMOKE:-0}" = "1" ]; then
  for arm in W WR WS; do
    $PY experiments/hippocampus/pilot_hippocampus.py --arm "$arm" --seed 0 --variant "$VARIANT" --smoke || exit 1
  done
  echo "[run_15] smoke OK (variant $VARIANT)"
  exit 0
fi

for seed in 0 1 2; do
  for arm in W WR WS; do
    echo "=== $(date -u '+%H:%M:%S') arm=$arm seed=$seed variant=$VARIANT ==="
    $PY experiments/hippocampus/pilot_hippocampus.py --arm "$arm" --seed "$seed" --variant "$VARIANT" || exit 1
  done
done
echo "[run_15] all 9 runs done (variant $VARIANT)"
