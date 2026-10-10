#!/usr/bin/env python
"""生成论文两图（数据全部来自实测 run，无合成数字）。
Fig1 规模/剂量阶梯：梦增益比值随参数量衰减，剂量放大不救。
Fig2 两夜自驱：梦长模型的生长曲线与归一残差轨迹。
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/Users/willsmacbookpro/Documents/dsh/mission_yin_yang")
FIG = R / "paper/figs"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 200, "savefig.bbox": "tight",
})

# ---------------- Fig 1: 规模与剂量阶梯 ----------------
params = [16e3, 210e3, 3.2e6]          # d_h 64/256/1024
e0 = [0.39, 0.39, 0.42]                 # E0 三 seed @16K
r1_mean = [0.40, 0.835, 0.705]          # R1 两 seed 均值
r1_pts = [(16e3, 0.41), (16e3, 0.39),
          (210e3, 0.88), (210e3, 0.79),
          (3.2e6, 0.88), (3.2e6, 0.53)]
r1b_pts = [(210e3, 0.88), (210e3, 0.79)]  # 剂量×4：与恒剂量逐位同值

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axhline(0.6, color="crimson", ls="--", lw=1.0, label="pre-registered threshold (0.6)")
ax.scatter([16e3] * 3, e0, marker="D", s=22, facecolor="none", edgecolor="tab:green",
           label="E0 (dream-first, 3 seeds)", zorder=3)
ax.scatter(*zip(*r1_pts), marker="o", s=22, color="tab:blue", alpha=0.55,
           label="R1 (fixed dose 200)", zorder=3)
ax.plot(params, r1_mean, color="tab:blue", lw=1.2, alpha=0.8)
ax.scatter(*zip(*r1b_pts), marker="s", s=26, color="tab:orange",
           label=r"R1b (dose $\propto d_h$: 800)", zorder=4)
ax.scatter([3.2e6], [1.0], marker="x", s=55, color="tab:orange", zorder=4)
ax.annotate("dose 3200:\nnever reached", (3.2e6, 1.0), xytext=(2.35e6, 0.90),
            fontsize=7.5, color="tab:orange")
ax.set_xscale("log")
ax.set_xticks(params)
ax.set_xticklabels(["16K\n($d_h$64)", "210K\n($d_h$256)", "3.2M\n($d_h$1024)"])
ax.set_ylim(0.2, 1.12)
ax.set_xlabel("parameters")
ax.set_ylabel(r"steps-to-target / wake-only steps")
ax.legend(fontsize=6.8, frameon=False, loc="lower left")
fig.savefig(FIG / "fig_scale.pdf")
plt.close(fig)

# ---------------- Fig 2: 两夜自驱轨迹 ----------------
def load_night(path_glob):
    rows = []
    for f in sorted(Path(path_glob).glob("round_[0-9]*.json")):
        try:
            d = json.loads(f.read_text())
            p = d["probes"]
            if p.get("res_norm") is None:
                continue
            rows.append((d["round"], d["d_h"], p["res_norm"]))
        except Exception:
            pass
    return rows

n1 = load_night(R / "experiments/overnight_autodriver/out_night1_local/seeds") \
    if (R / "experiments/overnight_autodriver/out_night1_local").exists() else load_night("/tmp/nonexist")
if not n1:
    # 夜一种子在仓库内（dabf471 入库），定位之
    n1 = load_night(R / "experiments/overnight_autodriver/out/seeds")
n2 = load_night("/tmp/somnium_n2/seeds")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=True,
                             gridspec_kw={"hspace": 0.12})
for rows, color, label in ((n1, "gray", "night 1 (no guards)"),
                           (n2, "tab:blue", "night 2 (guards)")):
    r = [x[0] for x in rows]
    a1.step(r, [x[1] for x in rows], where="post", color=color, lw=1.3, label=label)
    a2.semilogy(r, [max(x[2], 1e-2) for x in rows], color=color, lw=0.8, alpha=0.9)
a1.set_ylabel(r"$d_h$")
a1.legend(fontsize=7, frameon=False, loc="upper left")
a2.set_xlabel("autonomous round")
a2.set_ylabel(r"res$_{norm}$")
a2.axhline(0.6, color="crimson", ls="--", lw=0.8)
a2.axhline(1.2, color="tab:red", ls=":", lw=0.8)
a2.annotate("night1: unclipped divergence\nafter growth to $d_h{=}1024$",
            (760, 2.3), fontsize=6.5, color="dimgray", ha="right")
fig.savefig(FIG / "fig_overnight.pdf")
plt.close(fig)
print("figures written:", list(p.name for p in FIG.glob("*.pdf")))
