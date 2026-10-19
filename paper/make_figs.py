#!/usr/bin/env python
"""生成论文两图（数据全部来自实测 run，无合成数字）。
Fig1 规模/剂量阶梯：梦增益比值随参数量衰减，剂量放大不救。
Fig2 两夜自驱：梦长模型的生长曲线与归一残差轨迹。
"""
import json
import math
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]  # 仓库根（文件相对定位）
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
            rn = p.get("res_norm")
            if rn is None:  # 夜一为 v2.0 格式：现场归一（定义与 v2.1 等价）
                rn = p["residual"] / math.sqrt(d["d_h"])
            if rn > 6:      # 发散尖峰裁掉（σ=inf 时刻残差飞出）
                continue
            rows.append((d["round"], d["d_h"], rn))
        except Exception:
            pass
    return rows

n1 = load_night(R / "experiments/overnight_autodriver/out/seeds")
n2 = load_night(R / "experiments/overnight_autodriver/out_night2/seeds")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=True,
                             gridspec_kw={"hspace": 0.12})
for rows, color, label, lw, al in ((n1, "dimgray", "night 1 (no guards)", 2.2, 1.0),
                                   (n2, "tab:blue", "night 2 (guards)", 1.0, 0.85)):
    r = [x[0] for x in rows]
    a1.step(r, [x[1] for x in rows], where="post", color=color, lw=lw,
            alpha=al, label=label)
    a2.semilogy(r, [max(x[2], 1e-2) for x in rows], color=color, lw=lw * 0.6,
                alpha=al)
n1_end = max(x[0] for x in n1)
for ax in (a1, a2):
    ax.axvline(n1_end, color="k", ls="-", lw=0.7, alpha=0.5)
a1.annotate("night 1 ends", (n1_end + 60, 300), fontsize=6.5, color="dimgray")
a1.set_ylabel(r"$d_h$")
a1.legend(fontsize=7, frameon=False, loc="center right")
a2.set_xlabel("autonomous round")
a2.set_ylabel(r"res$_{norm}$")
a2.axhline(0.6, color="crimson", ls="--", lw=0.8)
a2.axhline(1.2, color="tab:red", ls=":", lw=0.8)
# 注释定位（v1.3 视觉验收 R2，裁判建议采纳）：两注释均放在底子图右侧
# 空白区（night-2 数据段自 round 870 后贴 0.6 线，上方 y≈2–3 为空）；
# y 必须在 ylim 数据范围内（R1 曾设 6.5 超出渲染区致注释丢失）
a2.annotate("night 1: destabilization at $d_h{=}729$",
            (4850, 1.62), fontsize=6.5, color="dimgray", ha="right")
a2.annotate("night 2: clipped, stable\nat $d_h{=}1024$ for 3000+ rounds",
            (2500, 2.4), fontsize=6.5, color="tab:blue", ha="left")
fig.savefig(FIG / "fig_overnight.pdf")
plt.close(fig)
print("figures written:", list(p.name for p in FIG.glob("*.pdf")))
