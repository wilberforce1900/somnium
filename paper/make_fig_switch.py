#!/usr/bin/env python
"""fig_switch：非平稳维护度悬崖（ITER12-fix 实测 + ITER12a 平凡域对照）。
数据全部来自 experiments/env_e_v3/ 的 final.json（无合成数字）。
x=dwell（周期，log）；y=切换相位末十维护度 acc_sw_end（in-regime 终局符号准确率）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]  # 仓库根（文件相对定位）
E = R / "experiments/env_e_v3"
FIG = R / "paper/figs"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def load(name):
    return json.loads((E / name).read_text())


# --- 修正矩阵（dwell 以周期计；λ=1/(dwell*180)）---
valid = {
    ("wake", 750): "switch_wake_lam7.407e-06_s{}.final.json",
    ("wake", 150): "switch_wake_lam3.704e-05_s{}.final.json",
    ("wake", 50): "switch_wake_lam1.111e-04_s{}.final.json",
    ("dream", 750): "switch_dream_lam7.407e-06_s{}.final.json",
    ("dream", 150): "switch_dream_lam3.704e-05_s{}.final.json",
    ("dream", 50): "switch_dream_lam1.111e-04_s{}.final.json",
}
# --- 首矩阵（单位错误版：dwell 以步计 → 周期 = steps/180；留档作悬崖底）---
triv = {
    ("wake", 750 / 180): "switch_wake_dwell750_s{}.final.json",
    ("wake", 150 / 180): "switch_wake_dwell150_s{}.final.json",
    ("wake", 50 / 180): "switch_wake_dwell50_s{}.final.json",
    ("dream", 750 / 180): "switch_dream_dwell750_s{}.final.json",
    ("dream", 150 / 180): "switch_dream_dwell150_s{}.final.json",
    ("dream", 50 / 180): "switch_dream_dwell50_s{}.final.json",
}

fig, ax = plt.subplots(figsize=(4.6, 3.0))

# τ 带（ITER11 实测）
ax.axvspan(25, 70, color="tab:orange", alpha=0.13, lw=0)
ax.axvspan(90, 215, color="tab:orange", alpha=0.07, lw=0)
ax.axhline(0.5, color="crimson", ls="--", lw=0.9)
ax.annotate("chance", (0.22, 0.5), fontsize=6.5, color="crimson",
            va="bottom")

styles = {"wake": ("tab:blue", "o", "wake-only"),
          "dream": ("tab:green", "s", "dream-first")}
for sched, (color, marker, label) in styles.items():
    xs, ys = [], []
    for (sch, dwell), pat in valid.items():
        if sch != sched:
            continue
        vals = [load(pat.format(s))["acc_sw_end"] for s in (0, 1)]
        ax.scatter([dwell] * len(vals), vals, marker=marker, s=24,
                   color=color, alpha=0.85, zorder=4,
                   label=label if dwell == 750 else None)
        xs.append(dwell)
        ys.append(sum(vals) / len(vals))
    ax.plot(xs, ys, color=color, lw=1.1, alpha=0.6, zorder=3)

# 平凡域（单位错误版）灰点
for (sch, dwell), pat in triv.items():
    vals = [load(pat.format(s))["acc_sw_end"] for s in (0, 1)]
    ax.scatter([dwell] * len(vals), vals, marker="x", s=20,
               color="dimgray", alpha=0.75, zorder=4,
               label="units-error run (retained)" if (sch, dwell) == ("wake", 50 / 180) else None)

ax.set_xscale("log")
ax.set_xlim(0.18, 1600)
ax.set_ylim(0.34, 1.02)
ax.set_xticks([0.3, 3, 30, 300])
ax.set_xticklabels(["0.3", "3", "30", "300"])
ax.set_xlabel("regime dwell (cycles; continuous switching)")
ax.set_ylabel("in-regime accuracy (acc$_{sw}$, end)")
ax.annotate("dwell $\\approx$ recovery $\\tau$", (32, 0.88), fontsize=6.5,
            color="tab:orange", ha="left")
ax.legend(fontsize=6.8, frameon=False, loc="lower right")
fig.savefig(FIG / "fig_switch.pdf")
plt.close(fig)
print("figures written:", list(p.name for p in FIG.glob("*.pdf")))
