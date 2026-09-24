# -*- coding: utf-8 -*-
"""113 号三张图：口径对照 / 字段风险画像 / 增益矩阵。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A, F = os.path.join(ROOT, "outputs", "analysis"), os.path.join(ROOT, "figures")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
S = pd.read_csv(os.path.join(A, "113_summary.csv")); P = pd.read_csv(os.path.join(A, "113_field_profile.csv"))
S["短名"] = S["数据集"] + "\n" + S["目标"].str.replace(r"（.*", "", regex=True).str.replace(r"^\S+ ", "", regex=True)

# ---------------- 图1：两种口径对照
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
tg = S["短名"].drop_duplicates().tolist(); x = np.arange(len(tg)); w = .36
for k, (col, ttl, fmt) in enumerate([
        ("τ0.7_单看安全组合危险", "A. 单看安全、至多 2 个背景字段即越过 τ=0.7 的字段数", "{:.0f}"),
        ("背景升级均值", "B. 背景带来的平均升级 M^(2)−M^(0)", "{:.3f}"),
        ("τ0.7_最少扣留", "C. 打破全部 ≤3 字段危险组合需扣留的最少字段数", "{:.0f}")]):
    for j, (var, colr, lab) in enumerate([("A", "#9fc3ea", "口径 A：全部候选"), ("B", BLUE, "口径 B：剔除次日才披露的字段")]):
        d = S[S.口径 == var].set_index("短名").reindex(tg)
        bars = axs[k].bar(x + (j - .5) * w, d[col], w * .92, color=colr, label=lab)
        for xi, v, n in zip(x + (j - .5) * w, d[col], d["候选数"]):
            txt = fmt.format(v) + (f"/{n}" if k != 1 else "")
            axs[k].text(xi, v * 1.02 + (0.002 if k == 1 else 0.3), txt, ha="center", fontsize=7.5, color=INK2)
    axs[k].set_xticks(x); axs[k].set_xticklabels(tg, fontsize=8.5)
    axs[k].set_title(ttl, loc="left", fontsize=9.8, color=INK); axs[k].grid(axis="y", color=GRID); axs[k].set_axisbelow(True)
axs[1].legend(frameon=False, fontsize=8.5, loc="upper left")
for ax in axs: ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
fig.suptitle("图1  两种候选口径下，组合推断风险都普遍存在；口径 A 中大量风险来自次日才披露的实际运行数据（电量平衡关系）",
             x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(F, "fig1_variant_comparison.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- 图2：字段风险画像（口径 B）
cases = [("PJM", "metered_load_mw", "PJM 实际计量负荷（口径 B，17 个候选）"),
         ("CAISO", "actual_load__mw__ca_iso_tac", "CAISO 实际负荷（口径 B，25 个候选，取 M^(2) 前 17）")]
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, (ds, t, ttl) in zip(axs, cases):
    d = P[(P.数据集 == ds) & (P.目标 == t) & (P.口径 == "B")].nlargest(17, "M2").sort_values("M2").reset_index(drop=True)
    ys = np.arange(len(d))
    for k, r in d.iterrows():
        ax.plot([r.M0, r.M2], [k, k], color="#b9b8b2", lw=2, zorder=1)
    ax.scatter(d.r2, ys, marker="D", s=40, color=AQUA, zorder=3, label="r²：单字段线性相关", edgecolor="#fcfcfb", lw=1.2)
    ax.scatter(d.M0, ys, s=58, facecolor="#fcfcfb", edgecolor=ORANGE, lw=2, zorder=4, label="M(0)：单字段推断能力")
    ax.scatter(d.M2, ys, s=58, color=BLUE, zorder=5, label="M^(2)：至多 2 个背景下的最坏增益", edgecolor="#fcfcfb", lw=1.2)
    for k, r in d.iterrows():
        if r.M2 - r.M0 >= 0.10:
            ax.text(r.M2 + .012, k, f"+{r.M2 - r.M0:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(d.中文名, fontsize=8)
    ax.set_xlim(0, 1.05); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_title(ttl, loc="left", fontsize=10.2, color=INK); ax.set_xlabel("推断能力（测试集 R²）")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("图2  字段风险画像（按披露时点的现实口径）：右侧数字为背景带来的升级", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(os.path.join(F, "fig2_field_profile.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- 图3：增益矩阵（PJM 实际负荷，口径 B）
G = pd.read_csv(os.path.join(A, "113_gain_matrix_caiso_actual_load__mw__ca_iso_tac_B.csv"), index_col=0)
g = G.values.copy(); n = len(g)
fig, ax = plt.subplots(figsize=(9.8, 7.4))
disp = np.clip(g, 0, None)
im = ax.imshow(disp, cmap=SEQ, vmin=0, vmax=max(0.6, np.nanmax(disp)), aspect="auto")
for r in range(n):
    for c in range(n):
        v = g[r, c]
        if r == c:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, edgecolor=INK2, lw=1.2))
        if v >= 0.10 or r == c:
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.4 else INK)
ax.set_xticks(range(n)); ax.set_xticklabels(G.columns, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(G.index, fontsize=8)
ax.set_xlabel("攻击者已掌握的背景字段 j"); ax.set_ylabel("拟公开字段 i")
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label("增益 V({i,j})−V({j})；对角线为单字段 V({i})", color=INK2)
ax.set_title("图3  CAISO 实际负荷（口径 B，M^(2) 前 12 个字段）：同一字段在不同背景下的推断增益\n只标注 ≥0.10 的格子；对角框为单字段能力",
             loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_gain_matrix_caiso_load.png"), bbox_inches="tight"); plt.close(fig)
print("3 张图已写入")
