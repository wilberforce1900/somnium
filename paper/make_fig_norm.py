#!/usr/bin/env python
"""fig_norm（论文 v1.2）：治愈 + 宽度阶梯。

左：120cyc plateau MSE @k0/dh1024 四归一化臂（对数轴）——none/ln_pre ≈1.03
病态地板 vs ln_post 0.166 / rms 0.197；64 档 none 参考 0.134。
右：宽度阶梯 none vs ln_post（64/256/1024 plateau MSE，对数轴）——悬崖被压平。
数据源：experiments/r5_world_calibration/plateau_*_s0.json（seed 0）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R5 = HERE.parent / "experiments" / "r5_world_calibration"


def load(name):
    return json.load(open(R5 / name))["plateau_mse"]


def main():
    none_1024 = load("plateau_k0_dh1024_h32_lr0.001_linear_s0.json")
    ln_post_1024 = load("plateau_k0_dh1024_h32_lr0.001_linear_ln_post_s0.json")
    rms_1024 = load("plateau_k0_dh1024_h32_lr0.001_linear_rms_post_s0.json")
    ln_pre_1024 = load("plateau_k0_dh1024_h32_lr0.001_linear_ln_pre_s0.json")
    none_64 = load("plateau_k0_dh64_s0.json")
    none_256 = load("plateau_k0_dh256_s0.json")
    ln_64 = load("plateau_k0_dh64_h32_lr0.001_linear_ln_post_s0.json")
    ln_256 = load("plateau_k0_dh256_h32_lr0.001_linear_ln_post_s0.json")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    ax = axes[0]
    arms = ["none", "ln_pre\n(input-side)", "rms_post", "ln_post"]
    vals = [none_1024, ln_pre_1024, rms_1024, ln_post_1024]
    colors = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
    bars = ax.bar(arms, vals, color=colors, width=0.62)
    ax.axhline(none_64, ls="--", lw=1.2, color="#444",
               label=f"$d_h$=64 reference ({none_64:.3f})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.3f}",
                ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.08, 2.4)
    ax.set_ylabel("plateau latent MSE (120 cycles)")
    ax.set_title("(a) $d_h{=}1024$: normalization arms", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ws = [64, 256, 1024]
    ax.plot(ws, [none_64, none_256, none_1024], "o--", color="#b2182b",
            label="none (v0)")
    ax.plot(ws, [ln_64, ln_256, ln_post_1024], "s-", color="#2166ac",
            label="ln_post")
    # 标签防叠印（R4：x=64 簇终修）——红 64 点正下（下方空旷）；蓝 64 方块左上
    # （与 1024 处 0.166 对称）；256/1024 维持已通过的布局
    red_pos = {64: ((0, -14), "center"), 256: ((-16, 8), "right"),
               1024: ((-16, 8), "right")}
    blue_pos = {64: ((0, 10), "center"), 256: ((16, -5), "left"),
                1024: ((-15, 6), "right")}
    for x, y in zip(ws, [none_64, none_256, none_1024]):
        (dx, dy), ha = red_pos[x]
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8, color="#b2182b")
    for x, y in zip(ws, [ln_64, ln_256, ln_post_1024]):
        (dx, dy), ha = blue_pos[x]
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8, color="#2166ac")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_yscale("log")
    ax.set_ylim(0.06, 2.2)
    ax.set_xlim(48, 1400)
    ax.set_xlabel("latent width $d_h$")
    ax.set_title("(b) width ladder, $k{=}0$ seed 0", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    out = HERE / "figs" / "fig_norm.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
