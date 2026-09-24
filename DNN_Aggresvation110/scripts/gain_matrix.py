# -*- coding: utf-8 -*-
"""110 号：背景依赖的边际推断增益矩阵 + 字段风险画像，并出两张图。
增益矩阵 G[i, j] = V({i, j}) − V({j})：攻击者已掌握字段 j 时，再公开字段 i 带来的推断增益；
第一列 j=∅ 即单字段能力 M^(0)。V 为单调闭包（val 选子集，test 报告）。"""
import itertools, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
O, A, FG = (os.path.join(ROOT, d) for d in ("outputs", "outputs/analysis", "figures"))
T = pd.read_pickle(os.path.join(O, "v_table.pkl")); df = pd.read_csv(os.path.join(O, "case30_market.csv"))
F = [c for c in df.columns if not c.startswith("Y_")]; Y = [c for c in df.columns if c.startswith("Y_")]
p = len(F); idx = {s: r for r, s in enumerate(T.S)}


def closed(y):
    vt = np.clip(T[f"test|{y}"].values, 0, 1); vv = T[f"val|{y}"].values; V = {(): 0.0}
    for r, s in enumerate(T.S):
        bs, bv = s, vv[r]
        for k in range(1, len(s)):
            for sub in itertools.combinations(s, k):
                if vv[idx[sub]] > bv: bs, bv = sub, vv[idx[sub]]
        V[s] = vt[idx[bs]]
    return V


prof, G = [], {}
for y in Y:
    V = closed(y)
    g = np.full((p, p + 1), np.nan)
    for i in range(p):
        g[i, 0] = V[(i,)]
        for j in range(p):
            if j != i: g[i, j + 1] = V[tuple(sorted((i, j)))] - V[(j,)]
    G[y] = g
    pd.DataFrame(g, index=F, columns=["∅"] + F).to_csv(os.path.join(A, f"110_gain_matrix_{y}.csv"))
    for i in range(p):
        oth = [j for j in range(p) if j != i]; m = []
        for K in [0, 1, 2, 3, p - 1]:
            m.append(max(V[tuple(sorted(Tt + (i,)))] - V[Tt] for k in range(K + 1) for Tt in itertools.combinations(oth, k)))
        prof.append(dict(目标=y, 字段=F[i], 相关系数=abs(np.corrcoef(df[F[i]], df[y])[0, 1]),
                         M0=m[0], M1=m[1], M2=m[2], M3=m[3], MCI=m[4], 背景升级_M2减M0=m[2] - m[0]))
P = pd.DataFrame(prof); P.to_csv(os.path.join(A, "110_field_profile.csv"), index=False)

# ------------------------------------------------------------------ 图
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
short = {f: f.replace("节点电价_", "电价").replace("总出力预测", "预测").replace("断面阻塞_", "阻塞").replace("系统", "") for f in F}
ylab = {"Y_机组出力_bus26": "目标 A：bus26 机组实际出力", "Y_机组出力_bus21": "目标 B：bus21 机组实际出力"}

# 图1：增益矩阵
fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.6))
for ax, y in zip(axs, Y):
    g = np.clip(G[y], 0, None)
    order = np.argsort(-np.nanmax(g, 1))
    gm = g[order]
    im = ax.imshow(np.where(np.isnan(gm), np.nan, gm), cmap=SEQ, vmin=0, vmax=0.8, aspect="auto")
    for r in range(p):
        for c in range(p + 1):
            v = gm[r, c]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#efeeea", lw=0)); continue
            if v >= 0.15 or c == 0:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                        color="#ffffff" if v > 0.45 else INK)
    ax.axvline(0.5, color=INK2, lw=1.2)
    ax.set_xticks(range(p + 1)); ax.set_xticklabels(["无背景\n(M⁰)".replace("⁰", "0")] + [short[f] for f in F], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(p)); ax.set_yticklabels([short[F[i]] for i in order], fontsize=8)
    ax.set_xlabel("攻击者已掌握的背景字段 j"); ax.set_ylabel("拟公开字段 i")
    ax.set_title(ylab[y], loc="left", fontsize=10.5, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=axs, fraction=.018, pad=.01); cb.set_label("边际推断增益  V({i,j}) − V({j})", color=INK2)
fig.suptitle("图1  IEEE 30 节点市场算例：同一个公开字段，在不同背景下带来的推断增益差别很大\n"
             "每格 = 攻击者已掌握 j 时再公开 i 的增益；第一列为无背景（单字段）。只标注 ≥0.15 的格子", x=.01, ha="left", fontsize=11.5)
fig.savefig(os.path.join(FG, "fig1_gain_matrix.png"), bbox_inches="tight"); plt.close(fig)

# 图2：字段风险画像（|r|、M0、M2）
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.4), sharex=True)
for ax, y in zip(axs, Y):
    d = P[P.目标 == y].sort_values("M2").reset_index(drop=True)
    ys = np.arange(len(d))
    for k, r in d.iterrows():
        ax.plot([r.M0, r.M2], [k, k], color="#b9b8b2", lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(d.相关系数 ** 2, ys, marker="D", s=46, color=AQUA, zorder=3, label="r²：单字段线性相关（现行口径）",
               edgecolor="#fcfcfb", linewidth=1.5)
    ax.scatter(d.M0, ys, s=64, facecolor="#fcfcfb", edgecolor=ORANGE, linewidth=2, zorder=4, label="M⁰ 单字段推断能力".replace("⁰", "(0)"))
    ax.scatter(d.M2, ys, s=64, color=BLUE, zorder=5, label="M^(2) 至多 2 个背景字段下的最坏增益", edgecolor="#fcfcfb", linewidth=1.5)
    for k, r in d.iterrows():
        if r.背景升级_M2减M0 >= 0.15:
            ax.text(r.M2 + .015, k, f"+{r.背景升级_M2减M0:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([short[f] for f in d.字段], fontsize=8.5)
    ax.set_xlim(0, 1); ax.grid(axis="x", color=GRID, lw=.7); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.set_title(ylab[y], loc="left", fontsize=10.5, color=INK); ax.set_xlabel("推断能力（测试集 R²）")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("图2  字段风险画像：线性相关 r² 与单字段能力 M^(0) 都会低估组合风险（右侧数字为背景带来的升级 M^(2)−M^(0)）",
             x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(os.path.join(FG, "fig2_field_profile.png"), bbox_inches="tight"); plt.close(fig)
print(P.round(3).to_string(index=False))
