#!/usr/bin/env python
"""夜三结果快析（本地只读）：R5 门矩阵 / D·A3 重测 / E 长训曲线。

背景：D 阶段 18 run 共用 --tag night3（文件名不含 d_h），dh64/256 的 details
被 dh1024 覆盖——night3.log 的逐 cycle 打印是权威数据源（含 steps/mse/mpc）。
教训：多 d_h 批量跑必须 per-dh tag（后续修 run_e0 或脚本）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R5 = REPO / "experiments/r5_world_calibration"
LOG = REPO / "experiments/night3_long/night3.log"


def r5_table():
    rows = {}
    for p in sorted(R5.glob("plateau_k*_ln_post_s*.json")):
        d = json.load(open(p))
        rows[(d["k"], d["d_h"], d["seed"])] = d["plateau_mse"]
    print("== R5 门矩阵（plateau_mse，ln_post，s0）==")
    print("k |   dh64 |  dh256 | dh1024 | 256→1024")
    for k in range(5):
        a, b, c = (rows.get((k, dh, 0)) for dh in (64, 256, 1024))
        imp = f"{(1 - c / b) * 100:+.1f}%" if (b and c) else "-"
        fmt = lambda v: f"{v:.5f}" if v is not None else "   -   "
        print(f"{k} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {imp}")
    print("\n== s1 佐证 ==")
    for k in range(5):
        a, b = rows.get((k, 256, 1)), rows.get((k, 1024, 1))
        if a and b:
            print(f"k={k}: 256={a:.5f} → 1024={b:.5f}  {(1 - b / a) * 100:+.1f}%")
    print("\n== C 上探 2048 ==")
    for k in (0, 4):
        v = rows.get((k, 2048, 0))
        if v is not None:
            print(f"k={k}: 2048={v:.5f}  对照(256={rows.get((k,256,0)):.5f} "
                  f"1024={rows.get((k,1024,0)):.5f})")


def e0_analysis():
    text = open(LOG, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"\n>>> D (\w+) dh=(\d+) s=(\d+)", text)
    runs = {}
    for i in range(1, len(blocks), 4):
        sched, dh, s = blocks[i], int(blocks[i + 1]), int(blocks[i + 2])
        cycles = [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
                  for m in re.finditer(
                      r"cycle (\d+)/20 steps=(\d+) pred_mse=([\d.]+)", blocks[i + 3])]
        if cycles:
            runs[(sched, dh, s)] = cycles
    print("\n== D · A3 重测（steps_to_target / wake 总步数；阈值 ≤0.6）==")
    for dh in (64, 256, 1024):
        for s in (0, 1):
            w = runs.get(("wake_only", dh, s))
            if not w:
                print(f"dh={dh} s={s}: wake 缺失")
                continue
            wake_total, wake_mse = w[-1][1], w[-1][2]
            line = f"dh={dh:4d} s={s}: wake 终值 {wake_mse:.4f} @{wake_total}步"
            for sched in ("dream_first", "alternate"):
                r = runs.get((sched, dh, s))
                if not r:
                    line += f" | {sched}: 缺"
                    continue
                hit = next((c for c in r if c[2] <= wake_mse), None)
                if hit:
                    ratio = hit[1] / wake_total
                    line += (f" | {sched[:5]}: {ratio:.2f} "
                             f"{'✅' if ratio <= 0.6 else '❌'} (终值 {r[-1][2]:.4f})")
                else:
                    line += f" | {sched[:5]}: 未达标 (终值 {r[-1][2]:.4f})"
            print(line)


def long_curve():
    p = REPO / "experiments/night3_long/long_k4_dh1024_ln_post_s0.jsonl"
    rows = [json.loads(l) for l in open(p)]
    print(f"\n== E 长训（k=4/dh1024/ln_post；{rows[-1]['cycle']} cycles，"
          f"{rows[-1]['env_steps']} env steps）==")
    by_cycle = {r["cycle"]: r for r in rows}
    for c in (1, 10, 50, 100, 250, 500, 1000, 2000, 4000, 6000, 8000, 9000):
        r = by_cycle.get(c) or min(rows, key=lambda x: abs(x["cycle"] - c))
        print(f"cyc{r['cycle']:5d}: mse={r['mse']:.5f} a={r['alpha']:.3f} "
              f"hN={r['h_norm']:.3f} sat={r['sat']:.3f} di={r['di']:.4f}")


if __name__ == "__main__":
    r5_table()
    e0_analysis()
    long_curve()
