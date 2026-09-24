# -*- coding: utf-8 -*-
"""120 号：目录内中文图。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); A = os.path.join(ROOT, "outputs", "analysis"); FG = os.path.join(ROOT, "figures")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e6e5e0"
S = pd.read_csv(os.path.join(A, "summary120.csv"), index_col=0)["值"]; P = pd.read_csv(os.path.join(A, "field_profile120.csv"))
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
d = P.head(25).iloc[::-1]; y = np.arange(len(d))
for yy, (_, r) in zip(y, d.iterrows()):
    axs[0].plot([r.M0, r.M2], [yy, yy], color="#cfcec8", lw=2)
axs[0].scatter(d.M0, y, s=26, facecolor="white", edgecolor=ORANGE, lw=1.3, label="M^(0) 单字段"); axs[0].scatter(d.M2, y, s=26, color=BLUE, label="M^(2) 精确")
axs[0].scatter(d.M2估计, y, s=14, marker="x", color="#0b0b0b", label="M^(2) 估计（单种子）")
axs[0].set_yticks(y); axs[0].set_yticklabels(d.字段, fontsize=7); axs[0].legend(frameon=False, fontsize=8); axs[0].set_xlim(0, 1)
axs[0].set_title("A. 线路 C35：M^(2) 前 25 个字段（共 87 个）", loc="left", fontsize=10); axs[0].grid(axis="x", color=GRID)
lab = ["单字段定级", "扫描—认证\n(δ=0)", "扫描—认证\n(δ=0.05)"]
for k, (tau, off) in enumerate([(0.5, -.2), (0.7, .2)]):
    crit = S[f"τ{tau}_单字段即危险"] + S[f"τ{tau}_组合危险字段"]
    vals = [S[f"τ{tau}_单字段即危险"] / crit, S[f"τ{tau}_δ0.0_认证召回"], S[f"τ{tau}_δ0.05_认证召回"]]
    axs[1].bar(np.arange(3) + off, vals, .38, color=["#cfcec8", BLUE][k], label=f"τ={tau}")
    for xi, v in zip(np.arange(3) + off, vals):
        axs[1].text(xi, v + .015, f"{v:.2f}", ha="center", fontsize=8)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(lab); axs[1].set_ylim(0, 1.12); axs[1].legend(frameon=False); axs[1].grid(axis="y", color=GRID)
axs[1].set_title(f"B. 关键字段召回（精确率均为 1；认证只重训约 {int(S['τ0.5_δ0.0_认证重训集合数'])} 个集合）", loc="left", fontsize=10)
vals = [S.全枚举串行重训小时, S.扫描小时, S.认证重训小时_τ05]
axs[2].bar(range(3), vals, .55, color=[ORANGE, BLUE, "#0d366b"]); axs[2].set_yscale("log")
for xi, v in enumerate(vals):
    axs[2].text(xi, v * 1.15, f"{v:.1f} 小时", ha="center", fontsize=9)
axs[2].set_xticks(range(3)); axs[2].set_xticklabels(["逐个串行重训\n全部 109,823 个集合", "扫描（单种子估计）", "认证（约 520 个集合）"])
axs[2].set_title("C. 成本（串行重训单集合约 15 秒）", loc="left", fontsize=10); axs[2].grid(axis="y", color=GRID)
fig.suptitle("图1  大字段空间（p=87，只汇报不进论文）：RTS-GMLC 全部节点电价 + 阻塞标志 + 预测，目标线路 C35 逐时潮流", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig1_large_p.png"), bbox_inches="tight"); plt.close(fig); print("ok")
