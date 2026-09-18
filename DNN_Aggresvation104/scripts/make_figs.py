# -*- coding: utf-8 -*-
"""104 号图：M 保真散点、critical 检测、认证下界。"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#2b2b2a"
T = pd.read_csv(A / "M_table_official.csv"); S = pd.read_csv(A / "M_table_summary.csv")
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for k, K in enumerate([0, 1, 2]):
    for ds, color in [("pjm", BLUE), ("caiso", ORANGE)]:
        d = T[(T.数据集 == ds) & (T.K == K)]
        ax[k].scatter(d.exact_M, d.est_M_集成, s=10, alpha=0.5, color=color, edgecolor="none", label=ds.upper())
    ax[k].plot([0, 1], [0, 1], color="#8a8984", lw=1, ls="--"); ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1)
    mae = [S[(S.数据集 == ds) & (S.K == K)].MAE_集成.values[0] for ds in ["pjm", "caiso"]]
    ax[k].set_title(f"K={K}：MAE PJM {mae[0]:.3f} / CAISO {mae[1]:.3f}", loc="left", fontsize=10.5)
    ax[k].set_xlabel("精确 M（正式攻击器族真值）")
ax[0].set_ylabel("估计 M（φ+x,x² 三种子集成）"); ax[0].legend(frameon=False, markerscale=2)
fig.suptitle("图1  M 保真（41 字段 × 12 目标 × 2 数据集 = 2,952 条）", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig1_M_fidelity.png", bbox_inches="tight"); plt.close(fig); print("-> fig1")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
x = np.arange(3); w = 0.38
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = S[S.数据集 == ds].sort_values("K")
    for k, (col, ttl, ylim) in enumerate([("recall_τ0.7", "A. τ=0.7 critical 字段召回", (0, 1.15)),
                                          ("precision_τ0.7", "B. τ=0.7 critical 精确率", (0, 1.15)),
                                          ("认证下界比_集成", "C. 认证下界 / 精确 M", (0, 1.15))]):
        v = d[col].values
        ax[k].bar(x + (d_i - .5) * w, v, w * .9, color=color, label=ds.upper() if k == 0 else None)
        for xi, vv in zip(x + (d_i - .5) * w, v): ax[k].text(xi, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
        ax[k].set_xticks(x); ax[k].set_xticklabels(["K=0", "K=1", "K=2"]); ax[k].set_ylim(*ylim); ax[k].grid(axis="x", visible=False)
        ax[k].set_title(ttl, loc="left", fontsize=10.5)
ax[0].legend(frameon=False)
fig.suptitle("图2  用估计 M 做安全判定：精确率 0.95–1.00，召回随 K 下降（PJM K=2 为 0.75）", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig2_critical_detection.png", bbox_inches="tight"); plt.close(fig); print("-> fig2")

# 图3：K 升级曲线（正式真值）
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    piv = T[T.数据集 == ds].pivot_table(index=["目标", "字段"], columns="K", values="exact_M")
    ax[0].plot([0, 1, 2], [piv[k].mean() for k in [0, 1, 2]], marker="o", lw=2.2, color=color, label=f"{ds.upper()} 平均")
    ax[0].fill_between([0, 1, 2], [piv[k].quantile(.25) for k in [0, 1, 2]], [piv[k].quantile(.75) for k in [0, 1, 2]], color=color, alpha=0.15)
    up = [(piv[k] > piv[0] + 0.02).mean() for k in [1, 2]]
    ax[1].bar(np.arange(2) + (d_i - .5) * 0.38, up, 0.34, color=color, label=ds.upper())
    for xi, vv in zip(np.arange(2) + (d_i - .5) * 0.38, up): ax[1].text(xi, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=9)
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["K=0 单字段", "K=1", "K=2"]); ax[0].set_ylabel("精确 M（阴影为四分位区间）")
ax[0].legend(frameon=False); ax[0].set_title("A. 推断升级曲线（正式真值）", loc="left", fontsize=10.5)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["K=1 相对单字段", "K=2 相对单字段"]); ax[1].set_ylim(0, 1.05)
ax[1].legend(frameon=False); ax[1].grid(axis="x", visible=False); ax[1].set_title("B. 增益 >0.02 的 (字段,目标) 占比", loc="left", fontsize=10.5)
fig.suptitle("图3  背景预算的实际意义：多数字段在拿到 1–2 个辅助字段后风险明显上升", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig3_escalation_official.png", bbox_inches="tight"); plt.close(fig); print("-> fig3")
