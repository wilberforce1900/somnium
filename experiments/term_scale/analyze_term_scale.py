#!/usr/bin/env python
"""ITER9 收割：读 experiments/term_scale/ 的 final.json，按 ITER9-PLAN.md §2
预注册判据打印门表格（只读、不判——判决记录在 ITER9-PLAN §5-R 与 registry）。

用法：python experiments/term_scale/analyze_term_scale.py [dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHORS = {0: 0.5424, 1: 0.5064}      # P0-D4 acc0@600 逐位锚（D4 权重均衡版）
GATE = 0.85
MID = 0.70


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    recs = []
    for f in sorted(d.glob("term_scale_dh*_lam*_s*.final.json")):
        recs.append(json.loads(f.read_text()))
    if not recs:
        print("no final.json found in", d)
        return
    print(f"{'arm':<16}{'acc@600':>9}{'acc@1200':>10}{'acc_end':>9}"
          f"{'Δ(1200→end)':>12}{'mae_end':>9}  split(+/−)  n")
    for r in sorted(recs, key=lambda x: (x["d_h"], x["lam_rew"], x["seed"])):
        arm = f"dh{r['d_h']}_lam{r['lam_rew']:g}"
        sp = (f"{r['split_acc_pos']:.2f}/{r['split_acc_neg']:.2f}"
              if r["split_acc_pos"] is not None
              and r["split_acc_neg"] is not None else "n/a")
        def f2(v):
            return f"{v:.4f}" if isinstance(v, float) else "None"
        print(f"{arm:<16}{f2(r['acc0_at600']):>9}{f2(r['acc0_at1200']):>10}"
              f"{f2(r['acc0_end']):>9}{f2(r['delta_1200_end']):>12}"
              f"{f2(r['mae_end']):>9}  {sp:>10}  {r['n_end']}")

    print("\n--- 预注册门（ITER9-PLAN §2）---")
    for r in recs:
        if r["d_h"] != 64 or r["lam_rew"] != 1.0:
            continue
        dev = (r["acc0_at600"] - ANCHORS[r["seed"]]
               if r["acc0_at600"] is not None else None)
        flag = ("ALARM >0.02" if dev is not None and abs(dev) > 0.02 else "ok")
        print(f"锚点门 T64 s{r['seed']}: acc0@600={r['acc0_at600']} "
              f"vs {ANCHORS[r['seed']]} (dev={dev:+.4f}) {flag}"
              if dev is not None else f"锚点门 T64 s{r['seed']}: None")
    for r in recs:
        v = r["acc0_end"]
        if v is None:
            verdict = "None"
        elif v >= GATE:
            verdict = "✓ 过门"
        elif v >= MID:
            verdict = "◐ 升级臂"
        else:
            verdict = "✗ 不可建"
        tag = ("容量" if r["lam_rew"] == 1.0 else "加权")
        print(f"{tag}门 dh{r['d_h']}_lam{r['lam_rew']:g}_s{r['seed']}: "
              f"acc_end={v} → {verdict}")
    alldead = all((r["acc0_end"] or 0) < MID for r in recs) and len(recs) >= 10
    print(f"全灭分支（五臂全<0.70）: {'生效（H1/H2 双否决）' if alldead else '未触发'}")


if __name__ == "__main__":
    main()
