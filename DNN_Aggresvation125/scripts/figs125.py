# -*- coding: utf-8 -*-
"""125 号：目录内中文图。图 1 候选估计器比较（相对旧口径 E）；图 2 下游结果新旧对比。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
FG = os.path.join(ROOT, "figures"); os.makedirs(FG, exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
COL = {"E_old": "#eb6834", "C1": "#9fc3ea", "C2": "#3987e5", "C3": "#0d366b"}
NAME = {"E_old": "旧口径 E（只预测目标×3）", "C1": "C1 重建×3+截断", "C2": "C2 重建⊕随机×1+截断（3种子均值）", "C3": "C3 重建⊕随机×3+截断（选定）"}
M = pd.read_csv(os.path.join(ROOT, "outputs", "select", "candidates_mean.csv"), index_col=0)
met = [("V误差_3", "V误差 ≤3", -1), ("V误差_4", "V误差 规模4", -1), ("M2误差", "M^(2)误差", -1), ("Γ2误差", "Γ^(2)误差", -1),
       ("M2排序", "M^(2)排序", 1), ("认证前1", "认证前1", 1), ("认证前3", "认证前3", 1), ("背景召回前3", "背景召回前3", 1),
       ("召回_τ0.5_δ0.0", "关键召回\nτ=0.5", 1), ("召回_τ0.7_δ0.0", "关键召回\nτ=0.7", 1)]
fig, ax = plt.subplots(figsize=(16, 4.8)); w = .2
for k, c in enumerate(["E_old", "C1", "C2", "C3"]):
    vals = [M.loc[c, m] / M.loc["E_old", m] for m, _, _ in met]
    ax.bar(np.arange(len(met)) + (k - 1.5) * w, vals, w, color=COL[c], label=NAME[c], edgecolor="white")
    for j, (m, _, _) in enumerate(met):
        ax.text(j + (k - 1.5) * w, vals[j] + .005, f"{M.loc[c, m]:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
ax.axhline(1, color="#52514e", lw=.8); ax.set_xticks(range(len(met)))
ax.set_xticklabels([f"{l}\n({'越低越好' if s < 0 else '越高越好'})" for _, l, s in met], fontsize=7.5); ax.set_ylim(.55, 1.3)
ax.set_ylabel("相对旧口径 E"); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left"); ax.grid(color="#e6e5e0", axis="y"); ax.set_axisbelow(True)
ax.set_title("C3 在 10 项主要指标中 9 项不劣于其余候选（唯一例外：τ=0.5 关键召回，C2 均值 0.931 > C3 0.910），且无一项劣于旧口径 E", loc="left", fontsize=9.5)
fig.suptitle("图1  候选估计器比较（选择规则事先写入 base.yaml；柱顶为绝对值）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig1_candidates.png"), bbox_inches="tight"); plt.close(fig)

# 图 2：下游新旧对比
O = {"旧": os.path.join(REPO, "DNN_Aggresvation117", "outputs", "analysis"), "新": os.path.join(ROOT, "outputs", "d117")}
P = {"旧": os.path.join(REPO, "DNN_Aggresvation124", "outputs"), "新": os.path.join(ROOT, "outputs", "d124")}
C = {"旧": "#eb6834", "新": "#0d366b"}
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axs[0]; lab = ["τ=0.5\nδ=0", "τ=0.5\nδ=0.05", "τ=0.7\nδ=0", "τ=0.7\nδ=0.05"]
for k, t in enumerate(["旧", "新"]):
    r = pd.read_csv(os.path.join(O[t], "decision_rules.csv")); r = r[r.规则 == "扫描—认证标记"]
    v = [r[(r.τ == tau) & (r.δ == dl)].召回.mean() for tau in (0.5, 0.7) for dl in (0.0, 0.05)]
    ax.bar(np.arange(4) + (k - .5) * .38, v, .38, color=C[t], label=f"{t}估计器", edgecolor="white")
    for j, x in enumerate(v):
        ax.text(j + (k - .5) * .38, x + .005, f"{x:.3f}", ha="center", fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(lab); ax.set_ylim(.6, 1.0); ax.legend(frameon=False, fontsize=8)
ax.set_title("A. 扫描—认证的关键字段召回（精确率均为 1.00）", loc="left", fontsize=9.5)
ax = axs[1]
for k, t in enumerate(["旧", "新"]):
    wdf = pd.read_csv(os.path.join(O[t], "withholding_strategies.csv")); v = []
    for s in ["估计MUS最小命中集", "自适应M̂2贪心"]:
        for tau in (0.5, 0.7):
            e = wdf[(wdf.策略 == s) & (wdf.τ == tau) & (wdf.δ == 0.05)]; v.append(100 * e.残余真危险组合.sum() / e.危险组合数.sum())
    ax.bar(np.arange(4) + (k - .5) * .38, v, .38, color=C[t], edgecolor="white")
    for j, x in enumerate(v):
        ax.text(j + (k - .5) * .38, x + .08, f"{x:.1f}%", ha="center", fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(["命中集\nτ=0.5", "命中集\nτ=0.7", "估计M贪心\nτ=0.5", "估计M贪心\nτ=0.7"])
ax.set_ylabel("残留的真实危险组合（%）"); ax.set_title("B. 只用估计值选扣留字段（δ=0.05）：残留危险组合", loc="left", fontsize=9.5)
ax = axs[2]; names = ["Γ^(2)误差", "背景召回前1", "背景召回前3", "背景召回前5", "认证前3恢复Γ"]
for k, t in enumerate(["旧", "新"]):
    S = pd.read_csv(os.path.join(P[t], "profile_summary.csv")); Pp = pd.read_csv(os.path.join(P[t], "profile.csv"))
    v = [S.G2误差.mean(), S.背景召回前1.mean(), S.背景召回前3.mean(), S.背景召回前5.mean(), Pp.G2认证前3.sum() / Pp.G2[Pp.G2 > 1e-9].sum()]
    ax.bar(np.arange(5) + (k - .5) * .38, v, .38, color=C[t], edgecolor="white")
    for j, x in enumerate(v):
        ax.text(j + (k - .5) * .38, x + .01, f"{x:.3f}", ha="center", fontsize=7)
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=8); ax.set_title("C. 字段画像：Γ 误差与危险背景召回", loc="left", fontsize=9.5)
for a_ in axs:
    a_.grid(color="#e6e5e0", axis="y"); a_.set_axisbelow(True)
fig.suptitle("图2  下游用新估计器重算（117、124 号派生脚本，已用旧估计器逐数复现原结果）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig2_downstream.png"), bbox_inches="tight"); plt.close(fig)
print("图已生成")
