#!/usr/bin/env python
"""跨 seed 汇总 + 预注册阈值判定。P1-3 工具，正式跑后使用。

用法：
    .venv/bin/python experiments/summarize.py            # 全部实验
    .venv/bin/python experiments/summarize.py --exp E2   # 只看 E2

读 results/registry.md（只统计 备注=formal/* 的行）与 experiments/*/details_*.jsonl。
预注册判定（ROADMAP §1.7，2026-10-08 锁定）：
    E0/A3：某梦日程 steps_to_target ≤ 0.6 × wake_only 总步数（target=同 seed
           wake_only 终值 MSE）且两个 seed 方向一致 → 正信号
    E1/A2：learned 的 acc 均值 ≥ fixed_yang + 2pp（两任务、两 seed 方向一致）
    E2/A1：k64（或最优 K）succ ≥ k0 + 5pp；64 优于 32 与 128 → 数字保留
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "results" / "registry.md"


def parse_registry():
    rows = []
    for ln in REGISTRY.read_text(encoding="utf-8").split("\n"):
        if not ln.startswith("| 2026"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 9 or "formal" not in cells[8]:
            continue
        rows.append({
            "exp": cells[1], "config": cells[2], "seed": int(cells[3]),
            "steps": int(cells[4]), "mse": float(cells[5]), "aux": cells[6],
        })
    return rows


def load_details(exp_dir):
    """details 文件名: details_<...>_<tag>_<seed>.jsonl → {config_key: [runs]}"""
    out = defaultdict(list)
    for f in sorted(exp_dir.glob("details_*_formal_*.jsonl")):
        m = re.match(r"details_(.*)_(\d+)\.jsonl", f.name)
        if not m:
            continue
        recs = [json.loads(x) for x in f.read_text().split("\n") if x.strip()]
        out[m.group(1)].append({"seed": int(m.group(2)), "cycles": recs})
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def aux_field(aux, key):
    m = re.search(rf"{key}=([-\d.]+)", aux)
    return float(m.group(1)) if m else float("nan")


def summarize_e0(rows):
    det = load_details(REPO / "experiments/e0_dream_first")
    print("\n== E0 · 先梦后醒 (A3) ==")
    by_cfg = defaultdict(list)
    for r in rows:
        if r["exp"] == "E0":
            by_cfg[r["config"]].append(r)
    for cfg, rs in sorted(by_cfg.items()):
        mses = [r["mse"] for r in rs]
        steps = [r["steps"] for r in rs]
        print(f"  {cfg:22s} mse={mean(mses):.4f}  steps={mean(steps):.0f}  (n={len(rs)})")
    # steps_to_target：target = 同 seed wake_only 终值 MSE
    wake = {}
    for key, runs in det.items():
        if key.startswith("wake_only"):
            for run in runs:
                wake[run["seed"]] = run["cycles"][-1]["pred_mse"]
    if not wake:
        print("  [尚无 wake_only formal details，跳过判定]")
        return
    total_wake = max(r["steps"] for r in rows if r["exp"] == "E0" and "wake_only" in r["config"]) \
        if any(r["exp"] == "E0" and "wake_only" in r["config"] for r in rows) else None
    for key, runs in sorted(det.items()):
        sched = key.split("_")[0]
        ratios = []
        for run in runs:
            tgt = wake.get(run["seed"])
            if tgt is None:
                continue
            hit = next((c["env_steps"] for c in run["cycles"] if c["pred_mse"] <= tgt), None)
            if hit is not None and total_wake:
                ratios.append(hit / total_wake)
        if ratios and sched != "wake_only":
            ok = all(x <= 0.6 for x in ratios)
            print(f"  steps_to_target/total_wake  {sched}: "
                  f"{['%.2f' % x for x in ratios]} → {'✅ A3 正信号' if ok else '❌ 未达 0.6 阈值'}")


def summarize_e1(rows):
    print("\n== E1 · 阴阳门控 (A2) ==")
    by_cfg = defaultdict(list)
    for r in rows:
        if r["exp"] == "E1":
            by_cfg[r["config"]].append(r)
    accs = {}
    for cfg, rs in sorted(by_cfg.items()):
        acc = mean([aux_field(r["aux"], "acc") for r in rs])
        accs[cfg] = acc
        al = mean([aux_field(r["aux"], "α") for r in rs])
        print(f"  {cfg:24s} acc={acc:.3f} α={al:.2f} (n={len(rs)})")
    for task in ("grid", "maze"):
        lrn = accs.get(f"{task}/learned+r0")
        fy = accs.get(f"{task}/fixed_yang+r0")
        if lrn is not None and fy is not None:
            ok = lrn - fy >= 0.02
            print(f"  {task}: learned−fixed_yang = {lrn-fy:+.3f} → "
                  f"{'✅' if ok else '❌'} +2pp 阈值")


def summarize_e2(rows):
    print("\n== E2 · 六十四卦码本 (A1) ==")
    by_cfg = defaultdict(list)
    for r in rows:
        if r["exp"] == "E2":
            by_cfg[r["config"]].append(r)
    succs = {}
    for cfg, rs in sorted(by_cfg.items()):
        succ = mean([aux_field(r["aux"], "succ") for r in rs])
        succs[cfg] = succ
        print(f"  {cfg:34s} succ={succ:.3f} (n={len(rs)})")
    base = succs.get("grid_long/k0/free+g0.3")
    for k in (64, 32, 128):
        g = succs.get(f"grid_long/k{k}/graph+g0.3")
        if base is not None and g is not None:
            print(f"  k{k}/graph − k0/free = {g-base:+.3f} → "
                  f"{'✅' if g-base >= 0.05 else '❌'} +5pp 阈值")
    g64, g32, g128 = (succs.get(f"grid_long/k{k}/graph+g0.3") for k in (64, 32, 128))
    if all(v is not None for v in (g64, g32, g128)):
        verdict = "六十四数字保留为架构主张" if g64 > g32 and g64 > g128 else "数字降级为命名（机制保留）"
        print(f"  64 vs 32/128: {g64:.3f}/{g32:.3f}/{g128:.3f} → {verdict}")


def summarize_e3(rows):
    """E3a/A4：读 details_e3_*_formal_*.jsonl，保持率聚合与组件贡献。"""
    import numpy as np  # noqa: F401
    det_dir = REPO / "experiments/e3_dream_ablation"
    data = defaultdict(dict)  # variant -> seed -> record
    for f in sorted(det_dir.glob("details_e3_*_formal_*.jsonl")):
        m = re.match(r"details_e3_(\w+)_formal_(\d+)\.jsonl", f.name)
        if not m:
            continue
        rec = json.loads(f.read_text().strip().split("\n")[0])
        data[m.group(1)][int(m.group(2))] = rec
    if not data:
        print("\n== E3 · 梦功能分解 (A4) ==\n  [尚无 formal details]")
        return
    print("\n== E3 · 梦功能分解 (A4) ==")
    means = {}
    for var in ("full", "none", "no_imagination", "no_consolidation",
                "no_reverse_learning", "no_rehearsal"):
        if var not in data:
            continue
        rets = [r["retention_mean"] for r in data[var].values()]
        robs = [r["robust_ratio"][0] for r in data[var].values()]
        means[var] = mean(rets)
        print(f"  dream_{var:18s} retention={mean(rets):+.3f} (n={len(rets)}) "
              f"robust≈{mean(robs):.2f}")
    if "full" in means and "none" in means:
        d = means["full"] - means["none"]
        ok = d >= 0.10
        print(f"  full − none = {d:+.3f} → {'✅ A4 正信号' if ok else '❌ 未达 +10pp'}")
    for var, mv in means.items():
        if var.startswith("no_") and "full" in means:
            d = means["full"] - mv
            if d >= 0.05:
                print(f"  组件贡献：关 {var.replace('no_', '')} 后保持率降 {d:+.3f} ≥5pp → 有独立贡献")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["E0", "E1", "E2", "E3"], default=None)
    args = ap.parse_args()
    rows = parse_registry()
    if args.exp in (None, "E0", "E1", "E2") and not rows:
        print("registry 中暂无 formal 行。")
        return
    if rows:
        print(f"formal 行数：{len(rows)}")
    if args.exp in (None, "E0"):
        summarize_e0(rows)
    if args.exp in (None, "E1"):
        summarize_e1(rows)
    if args.exp in (None, "E2"):
        summarize_e2(rows)
    if args.exp in (None, "E3"):
        summarize_e3(rows)


if __name__ == "__main__":
    main()
