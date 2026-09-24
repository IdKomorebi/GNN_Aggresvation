# -*- coding: utf-8 -*-
"""118 号 步骤 3：目录内的中文图。"""
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
GRID, BLUE, ORANGE = "#e6e5e0", "#2a78d6", "#eb6834"
V = pd.read_csv(os.path.join(A, "variants_by_target.csv"))
order = ["lin", "polyS", "head3", "rand1", "masknone1", "maskbern1", "phionly1", "nofix1", "D1", "E"]
lab = dict(zip(V.变体, V.名称)); m = V.groupby("变体")[["V误差", "M2误差", "认证前1", "M2排序Spearman"]].mean().reindex(order)
per = V.groupby(["变体", "数据"])[["V误差", "M2误差", "认证前1", "M2排序Spearman"]].mean()
fig, axs = plt.subplots(1, 4, figsize=(17, 5.2), sharey=True)
y = np.arange(len(order))[::-1]
cols = ["#cfcec8" if v in ("lin", "polyS", "head3") else ("#8a8984" if v not in ("D1", "E") else BLUE) for v in order]
for ax, (c, t, lim) in zip(axs, [("V误差", "A. V 的平均绝对误差", (0, .23)), ("M2误差", "B. M^(2) 的平均绝对误差", (0, .16)),
                                 ("认证前1", "C. 只认证首选背景的下界比", (.6, 1)), ("M2排序Spearman", "D. M^(2) 排序 Spearman", (.6, 1))]):
    ax.barh(y, m[c], .62, color=cols)
    for yy, v in zip(y, order):
        ax.scatter(per.loc[v][c].values, [yy] * len(per.loc[v]), s=9, color="#0b0b0b", zorder=3, alpha=.7)
        ax.text(m.loc[v, c], yy + .33, f"{m.loc[v, c]:.3f}", fontsize=7.5, color="#52514e")
    ax.set_xlim(*lim); ax.set_title(t, loc="left", fontsize=10); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
axs[0].set_yticks(y); axs[0].set_yticklabels([lab[v] for v in order])
fig.suptitle("图1  估计器对比与消融（10 个目标平均；黑点为各数据集平均；灰浅 = 无子集读出或无主干，灰深 = 单种子消融，蓝 = 本文）",
             x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig1_variants.png"), bbox_inches="tight"); plt.close(fig)
F = pd.read_csv(os.path.join(A, "finetune.csv")); F["名"] = F.数据 + "\n" + F.目标.str.slice(0, 14)
fig, ax = plt.subplots(figsize=(14, 4.4)); x = np.arange(len(F)); w = .27
ax.bar(x - w, F.微调V误差, w * .92, color=ORANGE, label="热启动微调（整网 ≤100 步，val 早停）")
ax.bar(x, F.单种子读出V误差, w * .92, color="#9fc3ea", label="本文·单种子闭式读出")
ax.bar(x + w, F.新主口径V误差, w * .92, color=BLUE, label="本文·三种子（新主口径）")
ax.set_xticks(x); ax.set_xticklabels(F.名, fontsize=7.5); ax.set_ylabel("V 的平均绝对误差（同一批 150 个集合）"); ax.legend(frameon=False)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.set_title(f"图2  热启动微调并不比闭式读出更准（微调约 {F.微调耗时s.mean():.2f} 秒/集合）", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FG, "fig2_finetune.png"), bbox_inches="tight"); plt.close(fig)
print("ok")
