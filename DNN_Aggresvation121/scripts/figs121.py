# -*- coding: utf-8 -*-
"""121 号：目录内中文图。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = os.path.join(ROOT, "outputs", "analysis"); FG = os.path.join(ROOT, "figures")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e6e5e0"
R = pd.read_csv(os.path.join(A, "regrade.csv"))
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, (g, lab) in zip(axs[:2], [("pjm_load", "PJM 实际负荷"), ("rts", "RTS 线路 C35")]):
    d = R[(R.组 == g) & (R.τ == 0.7)].sort_values("新增关键", ascending=False)
    x = np.arange(len(d))
    ax.bar(x, d.新增关键, .6, color=ORANGE, label="公开后新增的关键字段")
    ax.bar(x, -d.解除关键, .6, color="#9fc3ea", label="公开后解除的关键字段")
    ax.set_xticks(x); ax.set_xticklabels(d.已公开字段.str.slice(0, 18), rotation=70, ha="right", fontsize=6.5)
    ax.axhline(0, color="#52514e", lw=.8); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{'AB'[axs.tolist().index(ax)]}. {lab}（τ=0.7）：先公开哪个字段 → 其余字段关键性的变化", loc="left", fontsize=9.5)
S = pd.read_csv(os.path.join(A, "regrade_summary.csv")); S = S[S.τ == 0.7]
x = np.arange(len(S)); w = .2
for k, (col, lab, c) in enumerate([("估计关键召回", "估计值直接判定：关键召回", "#9fc3ea"), ("新增关键召回", "估计值直接判定：新增关键召回", "#cfcec8"),
                                   ("余量005关键召回", "保守余量 δ=0.05：关键召回", BLUE), ("余量005新增召回", "保守余量 δ=0.05：新增关键召回", "#0d366b")]):
    axs[2].bar(x + (k - 1.5) * w, S[col], w * .92, color=c, label=lab)
axs[2].set_xticks(x); axs[2].set_xticklabels(["PJM 实际负荷", "RTS 线路 C35"]); axs[2].set_ylim(0, 1.45); axs[2].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
axs[2].grid(axis="y", color=GRID); axs[2].set_axisbelow(True); axs[2].set_title("C. 通用模型跟踪再定级（τ=0.7）：高阈值需要保守余量", loc="left", fontsize=9.5)
fig.suptitle("图1  公开基底变化后的再定级（只汇报，不进论文）：真值为规模 ≤4 的三攻击器重训", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig1_regrade.png"), bbox_inches="tight"); plt.close(fig)
print("ok")
