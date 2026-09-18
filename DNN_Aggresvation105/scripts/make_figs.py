# -*- coding: utf-8 -*-
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
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#a3a29c", "#2b2b2a"
R = pd.read_csv(A / "105_critical_detection.csv"); R2 = pd.read_csv(A / "105_combination_only.csv"); P = pd.read_csv(A / "105_pr_curves.csv")
order = ["全量模型置换重要性", "全量模型 SAGE", "Pearson |r|", "Spearman |ρ|", "NMI", "单字段 M^(0)（精确）", "估计 M^(1)", "估计 M^(2)", "精确 M^(2)（参考上界）"]
lab = {o: o.replace("（精确）", "\n(精确)").replace("（参考上界）", "\n(参考上界)").replace("全量模型 ", "全量模型\n").replace("全量模型置换重要性", "全量模型\n置换重要性") for o in order}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
x = np.arange(len(order)); w = 0.38
for k, (df, ttl) in enumerate([(R[R.tau == 0.7], "A. 全部字段×目标"), (R2[R2.tau == 0.7], "B. 只看“组合才危险”的条目")]):
    for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
        v = df[df.数据集 == ds].set_index("指标").reindex(order).PR_AUC.values
        ax[k].bar(x + (d_i - .5) * w, v, w * .9, color=color, label=ds.upper() if k == 0 else None)
        for xi, vv in zip(x + (d_i - .5) * w, v): ax[k].text(xi, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=7.5)
    ax[k].set_xticks(x); ax[k].set_xticklabels([lab[o] for o in order], fontsize=7.5); ax[k].set_ylim(0, 1.05)
    ax[k].grid(axis="x", visible=False); ax[k].set_title(ttl, loc="left", fontsize=10.5)
ax[0].set_ylabel("τ-critical 字段检测 PR-AUC（τ=0.7, K=2）"); ax[0].legend(frameon=False)
fig.suptitle("图1  RQ-B1：模型依赖型指标（SAGE/置换重要性）明显最差；关联指标次之；单字段精确 R² 是很强的基线，M 的增益在 CAISO 明显、PJM 边际", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig1_baseline_pr.png", bbox_inches="tight"); plt.close(fig); print("-> fig1")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
show = ["全量模型 SAGE", "Pearson |r|", "NMI", "单字段 M^(0)（精确）", "估计 M^(2)"]
cols = {"全量模型 SAGE": GRAY, "Pearson |r|": "#9085e9", "NMI": "#1baf7a", "单字段 M^(0)（精确）": BLUE, "估计 M^(2)": ORANGE}
for k, ds in enumerate(["pjm", "caiso"]):
    for nm in show:
        d = P[(P.数据集 == ds) & (P.指标 == nm)].sort_values("recall")
        ax[k].plot(d.recall, d.precision, lw=2, color=cols[nm], label=nm)
    ax[k].set_xlabel("召回"); ax[k].set_title(ds.upper(), loc="left", fontsize=10.5); ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1.02)
ax[0].set_ylabel("精确率"); ax[1].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("图2  τ=0.7、K=2 的 PR 曲线（全部条目）", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig2_pr_curves.png", bbox_inches="tight"); plt.close(fig); print("-> fig2")

fig, ax = plt.subplots(figsize=(11, 4.2))
d = R.pivot_table(index=["数据集", "tau"], columns="指标", values="PR_AUC")
xs = np.arange(len(d)); w = 0.1
for j, nm in enumerate(["Pearson |r|", "NMI", "全量模型 SAGE", "单字段 M^(0)（精确）", "估计 M^(2)"]):
    ax.bar(xs + (j - 2) * w, d[nm].values, w * .9, label=nm)
ax.set_xticks(xs); ax.set_xticklabels([f"{a.upper()}\nτ={b}" for a, b in d.index], fontsize=8.5)
ax.set_ylabel("PR-AUC"); ax.legend(frameon=False, fontsize=8.5, ncol=3); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.1)
ax.set_title("图3  阈值敏感性：τ 越高（越严格的泄露判定），M 相对传统指标的优势越明显", loc="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(F / "fig3_tau_sensitivity.png", bbox_inches="tight"); plt.close(fig); print("-> fig3")
