#!/bin/sh
# mission_yin_yang 全量正式跑（2026-10-08，口径见 ROADMAP §1.7）
# E0(9) → E1(8) → E2(12)，串行。日志：results/formal_run.log
R=/Users/willsmacbookpro/Documents/dsh/mission_yin_yang
PY=$R/.venv/bin/python
cd "$R" || exit 1
echo "=== FORMAL RUN START $(date) ==="

echo "--- E0 formal: 3 schedules x 3 seeds ---"
for seed in 0 1 2; do
  for sched in wake_only dream_first alternate; do
    echo ">>> E0 $sched seed=$seed $(date +%H:%M:%S)"
    $PY experiments/e0_dream_first/run_e0.py --schedule "$sched" --seed "$seed" --full --tag formal || echo "!!! E0 $sched seed=$seed FAILED"
  done
done

echo "--- E1 formal: {learned,fixed_yang} x {grid,maze} x 2 seeds ---"
for seed in 0 1; do
  for task in grid maze; do
    for mode in learned fixed_yang; do
      echo ">>> E1 $task $mode seed=$seed $(date +%H:%M:%S)"
      $PY experiments/e1_yinyang_gate/run_e1.py --task "$task" --gate_mode "$mode" --seed "$seed" --full --tag formal || echo "!!! E1 $task $mode seed=$seed FAILED"
    done
  done
done

echo "--- E2 formal: 6 configs x 2 seeds ---"
for seed in 0 1; do
  for cfg in "0 free" "64 shaped" "64 free" "64 free_match" "32 shaped" "128 shaped"; do
    k=$(echo $cfg | cut -d' ' -f1)
    p=$(echo $cfg | cut -d' ' -f2)
    echo ">>> E2 k=$k $p seed=$seed $(date +%H:%M:%S)"
    $PY experiments/e2_hexagram_codebook/run_e2.py --k "$k" --planner "$p" --seed "$seed" --full --tag formal || echo "!!! E2 k=$k $p seed=$seed FAILED"
  done
done

echo "=== FORMAL RUN DONE $(date) ==="
echo "--- 汇总 ---"
$PY experiments/summarize.py
