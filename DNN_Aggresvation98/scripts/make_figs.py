# -*- coding: utf-8 -*-
"""98 号图表。全部读 outputs/analysis 下的结果文件，不重新计算。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#2b2b2a"
NAME = {"lin": "线性闭式", "lin2": "线性+平方", "poly2": "二次多项式", "direct": "共享头oracle(69/75号)",
        "L0": "L0 冻结φ+闭式读出(91号)", "L0x": "L0 + x,x²", "L1": "L1 元训练φ(97号r93)", "L1x": "L1 + x,x²",
        "L1ens": "L1 三种子集成 + x,x²", "L0ensx": "L0 三种子集成 + x,x²", "Lmixx": "L0⊕L1 + x,x²",
        "rffx": "随机傅里叶基(不学习)", "mono1": "单调约束 λ=1", "mono10": "单调约束 λ=10",
        "truth_seed1": "真值另一种子(噪声底)"}


def save(fig, name):
    fig.savefig(F / name, bbox_inches="tight"); plt.close(fig); print("->", name)


# 图1 估计器阶梯 ---------------------------------------------------------------
B = pd.read_csv(A / "B_fidelity_rand.csv", index_col=0)
order = ["lin", "direct", "rffx", "L1", "L0", "L1x", "L0x", "Lmixx", "L1ens", "L0ensx"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
y = np.arange(len(order))
for k, (col, title, color) in enumerate([("MAE_all", "集合价值 MAE（越小越好）", BLUE),
                                         ("danger@0.7", "τ=0.7 危险漏判率（真值>0.7 而估计≤0.7）", ORANGE),
                                         ("pairsyn_spearman", "字段对协同排序 Spearman（越大越好）", AQUA)]):
    v = B.loc[order, col].values
    ax[k].barh(y, v, height=0.62, color=color)
    for yi, vi in zip(y, v): ax[k].text(vi, yi, f" {vi:.3f}", va="center", fontsize=8.5, color=INK)
    ref = B.loc["truth_seed1", col]
    ax[k].axvline(ref, color=GRAY, ls="--", lw=1.2); ax[k].set_title(title, fontsize=10.5, loc="left")
    ax[k].text(ref, len(order) - 0.35, " 真值种子噪声底", color="#52514e", fontsize=8)
    ax[k].grid(axis="y", visible=False)
ax[0].set_yticks(y); ax[0].set_yticklabels([NAME[m] for m in order])
fig.suptitle("图1  44 维未见集合上的通用价值模型阶梯（4090 个集合 × 12 目标，真值=并行专用重训两种子平均）", x=0.01, ha="left", fontsize=12)
save(fig, "fig1_value_model_ladder.png")

# 图2 精确博弈：Shapley / 分级 / 交互 的排序保真 -------------------------------
rows = []
for g in ["gameA", "gameB"]:
    C = pd.read_csv(A / f"C_{g}_fidelity.csv", index_col=0); C["game"] = g; rows.append(C)
C = pd.concat(rows)
mods = ["SAGE_fullmodel_s0", "direct", "poly2", "L0", "L1x", "L1ens", "truth_seed0"]
lab = {"SAGE_fullmodel_s0": "全量模型SAGE\n(模型依赖型)", "direct": "共享头oracle", "poly2": "二次多项式",
       "L0": "L0", "L1x": "L1 + x,x²", "L1ens": "L1 集成", "truth_seed0": "真值另一种子\n(噪声底)"}
fig, ax = plt.subplots(figsize=(12, 4.4))
w = 0.26; x = np.arange(len(mods))
for k, (col, nm, color) in enumerate([("phi_tau", "字段 Shapley 值", BLUE), ("M1_tau", "背景最大边际 M¹（分级量）", AQUA),
                                      ("faith2_tau", "二阶交互 Faith-Shap", ORANGE)]):
    v = [C.loc[m, col].mean() for m in mods]
    ax.bar(x + (k - 1) * w, v, width=w - 0.03, color=color, label=nm)
    for xi, vi in zip(x + (k - 1) * w, v): ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in mods]); ax.set_ylim(0, 1.08)
ax.set_ylabel("与真值的 Kendall τ（逐目标平均，两博弈平均）"); ax.legend(frameon=False, loc="upper left", ncol=3)
ax.grid(axis="x", visible=False)
ax.set_title("图2  精确枚举博弈上：Shapley 排得较准，二阶交互明显更难；全量模型 SAGE 的交互与总体能力几乎无关", loc="left", fontsize=11.5)
save(fig, "fig2_exact_game_rank_fidelity.png")

# 图3 语义：同一目标下三种字段量 ----------------------------------------------
S = pd.read_csv(A / "C_gameA_semantics.csv")
d = S[S.conf == "metered_load_mw"].copy()
short = lambda f: (f.replace("gen_fuel_", "").replace("forecast_load_mw_latest_available", "负荷预测(最新)")
                   .replace("forecast_load_mw_day_ahead", "负荷预测(日前)").replace("wind_mw", "风电出力")
                   .replace("wind_pct", "风电占比").replace("da_as_", "").replace("_reserve", "")
                   .replace("_interchange_mw", "_ic"))
d["name"] = d.field.map(short); d = d.sort_values("M1")
fig, ax = plt.subplots(figsize=(9.5, 5.6)); yy = np.arange(len(d))
for col, nm, color, mk in [("single", "单字段 R²  v({i})", GRAY, "o"), ("shapley", "Shapley 值 φ_i", BLUE, "s"),
                           ("M1", "背景最大边际 M¹ = max_j v(i,j)−v(j)", ORANGE, "D")]:
    ax.scatter(d[col], yy, s=46, color=color, marker=mk, label=nm, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
for yi, (_, r) in zip(yy, d.iterrows()):
    ax.plot([min(r.single, r.shapley, r.M1), max(r.single, r.shapley, r.M1)], [yi, yi], color="#d6d5cf", lw=1, zorder=1)
ax.set_yticks(yy); ax.set_yticklabels(d.name); ax.set_xlim(-0.02, 1.05); ax.grid(axis="y", visible=False)
ax.set_xlabel("推断能力（测试 R² 口径）"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
ax.text(0.50, 1.0, "• 风电出力/占比：单独只能推 0.06 / 0.15，合起来 0.83；\n   Shapley 只有 0.035 / 0.050——多数联盟里已有负荷预测，边际≈0\n"
        "• 两个近重复负荷预测：各自可推 0.98，Shapley 各只分到 0.22，\n   全集删除边际≈0（删掉一个还有另一个）",
        fontsize=8.8, color="#2b2b2a", va="bottom", bbox=dict(boxstyle="round,pad=0.5", fc="#f3f2ee", ec="#d6d5cf"))
ax.set_title("图3  目标 metered_load_mw（gameA 真值）：Shapley 是“平均分摊”，不是“单独泄露风险”", loc="left", fontsize=11.5)
save(fig, "fig3_shapley_vs_risk_semantics.png")

# 图4 预算曲线 ----------------------------------------------------------------
D = pd.read_csv(A / "D_budget_curve.csv")
bench = json.load(open(A / "E_bench_cost.json"))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
for k, g in enumerate(["gameB", "gameA"]):
    d = D[D.game == g]
    for meth, nm, color in [("kernel_retrain", "直接重训 + 成对 KernelSHAP", BLUE), ("perm_retrain", "直接重训 + 置换采样", AQUA)]:
        q = d[d.method == meth]; ax[k].plot(q.budget, q.phi_MAE, color=color, lw=2, marker="o", ms=5, label=nm)
    for m, nm, ls in [("exact_direct", "共享头oracle 精确枚举", ":"), ("exact_L0", "L0 精确枚举", "--"), ("exact_L1ens", "L1集成 精确枚举", "-.")]:
        v = d[d.method == m].phi_MAE.values[0]
        ax[k].axhline(v, color=ORANGE, ls=ls, lw=1.4)
        ax[k].text(1150 if m != "exact_L1ens" else 12, v, f" {nm} {v:.4f}", fontsize=8, va="bottom" if m != "exact_L1ens" else "top",
                   ha="right" if m != "exact_L1ens" else "left", color=INK)
    p = 12 if g == "gameB" else 14
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel("需要的专用重训次数（不同联盟数）")
    ax[k].set_title(f"{g}（p={p}，精确枚举需 {2**p} 次查询）", loc="left", fontsize=10.5)
ax[0].set_ylabel("字段 Shapley 值 MAE（对精确真值）"); ax[1].legend(frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle(f"图4  达到通用模型同等 Shapley 精度所需的重训次数：约 300–600 次（并行重训 {bench['batched_B1024_s_per_set']*1e3:.0f} ms/集合 ≈ 半分钟）",
             x=0.01, ha="left", fontsize=12)
save(fig, "fig4_budget_curve.png")

# 图5 误差界 ------------------------------------------------------------------
d = D[(D.game == "gameA") & D.method.str.startswith("exact_")].copy(); d["m"] = d.method.str[6:]
d = d[d.m.isin(["lin", "direct", "poly2", "L0", "L0x", "L1x", "L1ens", "rffx"])].drop_duplicates("m")
fig, ax = plt.subplots(figsize=(11, 4.3)); x = np.arange(len(d)); w = 0.27
for k, (col, nm, color) in enumerate([("bound_2eps_mean", "GPT 方案 8.3：2ε 界（ε=最大价值误差）", GRAY),
                                      ("bound_margerr_max", "Shapley 加权边际误差界 max_i E|Δ_i e(S)|", BLUE),
                                      ("actual_max", "实际最大 Shapley 误差", ORANGE)]):
    ax.bar(x + (k - 1) * w, d[col], width=w - 0.03, color=color, label=nm)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in d.m], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("逐目标平均（对数轴）"); ax.set_ylim(0.008, 5); ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3); ax.grid(axis="x", visible=False)
ax.set_title("图5  gameA：2ε 界比实际最大误差松 9–60 倍；边际误差界只松 1–2 倍（价值误差在相邻联盟间高度相关）", loc="left", fontsize=11.5)
save(fig, "fig5_error_bounds.png")

# 图6 标签口径 + 单调约束 -------------------------------------------------------
L = pd.read_csv(A / "A_label_protocol.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
x = np.arange(len(L)); w = 0.38
ax[0].bar(x - w / 2, L.seed_noise, width=w - 0.03, color=GRAY, label="种子噪声 |s0−s1|/2")
ax[0].bar(x + w / 2, L.leak69_minus_clean, width=w - 0.03, color=ORANGE, label="测试集选轮次偏差（69号口径 − val口径）")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"规模 {b}" for b in L.index]); ax[0].set_ylim(0, 0.01)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left"); ax[0].grid(axis="x", visible=False)
ax[0].set_title("A. 历史真值标签的选择偏差只有 +0.002~0.003", loc="left", fontsize=10.5)
Cb = pd.read_csv(A / "C_gameB_fidelity.csv", index_col=0); Ca = pd.read_csv(A / "C_gameA_fidelity.csv", index_col=0)
mm = ["mono0", "mono1", "mono10"]; x = np.arange(3); w = 0.27
vals = [(B.loc[mm, "MAE_all"].values, "44维价值 MAE", BLUE),
        ((Ca.loc[mm, "phi_MAE"].values + Cb.loc[mm, "phi_MAE"].values) / 2 * 5, "Shapley MAE ×5（两博弈平均）", AQUA),
        (B.loc[mm, "mono_viol(<-0.02)"].values * 5, "单调违例率 ×5", ORANGE)]
for k, (v, nm, color) in enumerate(vals):
    ax[1].bar(x + (k - 1) * w, v, width=w - 0.03, color=color, label=nm)
ax[1].axhline(B.loc["truth_seed1", "mono_viol(<-0.02)"] * 5, color=GRAY, ls="--", lw=1.2)
ax[1].text(2.45, B.loc["truth_seed1", "mono_viol(<-0.02)"] * 5, "真值自身违例率×5", fontsize=8, ha="right", va="bottom")
ax[1].set_xticks(x); ax[1].set_xticklabels(["λ=0（=75号，逐位相同）", "λ=1", "λ=10"])
ax[1].legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.62)); ax[1].grid(axis="x", visible=False)
ax[1].set_title("B. 单调正则（GPT 8.2）：估计器违例本就低于真值；λ=10 明显变差", loc="left", fontsize=10.5)
fig.suptitle("图6  标签口径与单调约束", x=0.01, ha="left", fontsize=12)
save(fig, "fig6_labels_and_monotonicity.png")
