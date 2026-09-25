# -*- coding: utf-8 -*-
"""124 号：目录内中文图。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); A = os.path.join(ROOT, "outputs"); FG = os.path.join(ROOT, "figures")
os.makedirs(FG, exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
COL = {"RTS-GMLC": "#2a78d6", "NEM": "#eb6834", "PJM-load": "#1baf7a", "PJM-gen/ic": "#008300", "CAISO-load": "#4a3aa7"}
R = pd.read_csv(os.path.join(A, "profile.csv")); Dd = pd.read_csv(os.path.join(A, "context_gain_distribution.csv"))
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axs[0]
for ds, g in R.groupby("数据"):
    ax.scatter(g.M0, g.G2, s=16, alpha=.75, color=COL[ds], label=ds, edgecolor="white", lw=.4)
ax.axvline(0.3, color="#8a8984", lw=.7, ls=":"); ax.axhline(0.1, color="#8a8984", lw=.7, ls=":")
n = ((R.M0 < 0.3) & (R.G2 > 0.1)).sum(); ax.text(0.02, 0.47, f"左上象限（M^(0)<0.3、Γ^(2)>0.1）：{n}/{len(R)} 个", fontsize=8, color="#52514e")
ax.set_xlabel("单字段风险 M^(0)"); ax.set_ylabel("背景放大量 Γ^(2) = M^(2) − M^(0)"); ax.set_xlim(0, 1); ax.set_ylim(0, .5); ax.legend(frameon=False, fontsize=8)
ax.set_title("A. 固有风险—背景放大图（每点一个字段×目标）：左上 = 单字段不显眼、背景下被放大", loc="left", fontsize=9.5)
ax = axs[1]
pick = R[R.数据.isin(Dd.数据.unique())].sort_values("G2", ascending=False).drop_duplicates(["数据", "目标"]).head(7)
for k, (_, r) in enumerate(pick.iterrows()):
    d = Dd[(Dd.数据 == r.数据) & (Dd.目标 == r.目标) & (Dd.字段 == r.字段)].Δ.values
    ax.scatter(d, np.full(len(d), k) + np.random.RandomState(k).uniform(-.25, .25, len(d)), s=3, alpha=.35, color=COL[r.数据])
    ax.plot([r.M0], [k], "o", ms=7, mfc="white", mec="#eb6834", mew=1.5); ax.plot([d.mean()], [k], "|", ms=14, color="#0b0b0b", mew=2)
    ax.plot([d.max()], [k], "o", ms=7, color="#2a78d6")
ax.set_yticks(range(len(pick))); ax.set_yticklabels([f"{r.数据}｜{str(r.目标)[:10]}\n{str(r.字段)[:26]}" for _, r in pick.iterrows()], fontsize=7)
ax.set_xlabel("边际增益 Δ_i(T)（T 取遍规模 ≤2 的全部背景）"); ax.invert_yaxis()
ax.set_title("B. 同一字段在不同背景下的边际增益：空心=单字段、竖线=平均、实心=最大（M^(2)）", loc="left", fontsize=9.5)
for a_ in axs:
    a_.grid(color="#e6e5e0"); a_.set_axisbelow(True)
fig.suptitle("图1  背景依赖的字段风险：M 是边际增益分布的上包络，平均值会把背景放大稀释掉", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig1_profile_map.png"), bbox_inches="tight"); plt.close(fig)
S = pd.read_csv(os.path.join(A, "profile_summary.csv"))
fig, ax = plt.subplots(figsize=(11, 4)); x = np.arange(len(S)); w = .26
for k, (c, lab, col) in enumerate([("背景召回前1", "估计器首选背景 = 真实最危险背景", "#9fc3ea"), ("背景召回前3", "真实最危险背景在估计器前 3", "#2a78d6"),
                                   ("背景召回前5", "在估计器前 5", "#0d366b")]):
    ax.bar(x + (k - 1) * w, S[c], w * .92, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels((S.数据 + "\n" + S.目标.str.slice(0, 12)).values, fontsize=7); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8)
ax.set_title("图2  估计器找回最危险背景的能力（精确到同一个背景；另：认证前 3 个背景恢复 Γ^(2) 总量的 92.8%）", loc="left", fontsize=10)
ax.grid(axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(FG, "fig2_context_recall.png"), bbox_inches="tight"); plt.close(fig); print("ok")
