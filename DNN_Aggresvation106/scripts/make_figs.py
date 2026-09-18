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
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#2b2b2a"
S = pd.read_csv(A / "release_audit_summary.csv"); G = pd.read_csv(A / "grading_table.csv")
order = ["单字段放行", "相关性放行", "直接估计放行", "逐阶预算放行", "预算放行_估计", "预算放行"]
lab = {"单字段放行": "单字段能力\n≤τ 放行", "相关性放行": "相关性\n≤0.5 放行", "直接估计放行": "通用模型直接\n估计 V(A)≤τ",
       "逐阶预算放行": "逐阶预算\nU^(2)≤τ", "预算放行_估计": "估计 M 预算\nΣM̂≤τ", "预算放行": "精确 M 预算\nΣM≤τ"}
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
x = np.arange(len(order)); w = 0.38
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = S[S.数据集 == ds].set_index("审查规则").reindex(order)
    ax[0].bar(x + (d_i - .5) * w, d.危险放行率, w * .9, color=color, label=ds.upper())
    for xi, v in zip(x + (d_i - .5) * w, d.危险放行率): ax[0].text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8)
    ax[1].bar(x + (d_i - .5) * w, d.误拒率, w * .9, color=color)
    for xi, v in zip(x + (d_i - .5) * w, d.误拒率): ax[1].text(xi, v + 0.012, f"{v:.0%}", ha="center", fontsize=8)
for k, t in enumerate(["A. 危险放行率（不安全集合被放行的比例）", "B. 误拒率（安全集合被拒的比例）"]):
    ax[k].set_xticks(x); ax[k].set_xticklabels([lab[o] for o in order], fontsize=8); ax[k].grid(axis="x", visible=False)
    ax[k].set_title(t, loc="left", fontsize=10.5)
ax[0].set_ylim(0, 0.65); ax[1].set_ylim(0, 1.05); ax[0].legend(frameon=False)
fig.suptitle("图1  发布审查（随机 2,000 个规模 3 的待发布集合，τ=0.7）：单字段规则放行 19%/56% 的危险集合；M 预算规则零危险放行", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig1_release_audit.png", bbox_inches="tight"); plt.close(fig); print("-> fig1")

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
for k, (ds, tgt) in enumerate([("pjm", "total_gen"), ("caiso", "actual_load__mw__ca_iso_tac")]):
    d = G[(G.数据集 == ds)].sort_values("M2", ascending=False).head(14)[::-1]
    y = np.arange(len(d))
    ax[k].barh(y, d.M0, 0.5, color=GRAY, label="单字段 M^(0)")
    ax[k].barh(y, d.M2 - d.M0, 0.5, left=d.M0, color=ORANGE, label="K=2 背景增强")
    ax[k].scatter(d.估计M2, y, s=28, color=BLUE, zorder=3, label="估计 M^(2)")
    ax[k].axvline(0.7, color="#8a8984", ls="--", lw=1)
    nm = [f[:26] + ("*" if c and not s else "") for f, c, s in zip(d.字段, d.critical, d.单字段critical)]
    ax[k].set_yticks(y); ax[k].set_yticklabels(nm, fontsize=7.5); ax[k].grid(axis="y", visible=False); ax[k].set_xlim(0, 1.05)
    ax[k].set_title(f"{ds.upper()}：{tgt[:34]}", loc="left", fontsize=10.5)
ax[0].set_xlabel("推断风险（R² 口径），虚线为 τ=0.7"); ax[1].set_xlabel("推断风险（R² 口径）"); ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("图2  字段风险分级表（M^(2) 前 14 名）：星号 = 单字段看起来安全、但在 ≤2 个背景字段下会越过阈值", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "fig2_grading_table.png", bbox_inches="tight"); plt.close(fig); print("-> fig2")
