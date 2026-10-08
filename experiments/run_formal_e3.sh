#!/bin/sh
# E3a 正式跑（A4 梦功能分解）：6 变体 × 2 seed = 12 run。口径见 run_e3.py docstring。
R=/Users/willsmacbookpro/Documents/dsh/mission_yin_yang
PY=$R/.venv/bin/python
cd "$R" || exit 1
echo "=== E3 FORMAL START $(date) ==="
for seed in 0 1; do
  for v in full none no_imagination no_consolidation no_reverse_learning no_rehearsal; do
    echo ">>> E3 $v seed=$seed $(date +%H:%M:%S)"
    $PY experiments/e3_dream_ablation/run_e3.py --variant "$v" --seed "$seed" --tag formal \
      || echo "!!! E3 $v seed=$seed FAILED"
  done
done
echo "=== E3 FORMAL DONE $(date) ==="
$PY experiments/summarize.py --exp E3
