# -*- coding: utf-8 -*-
"""119 号：目录内中文图。"""
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
BLUE, ORANGE, AQUA, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e0"
# 图1 种子/划分一致性
C = pd.read_csv(os.path.join(A, "robust_consistency.csv")); C["组"] = C.数据 + "｜" + C.版本
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
for ax, (col, t) in zip(axs, [("M2排序Spearman", "A. 字段 M^(2) 排序 Spearman"), ("关键集合Jaccard_τ05", "B. τ=0.5 关键字段集合 Jaccard"),
                              ("关键集合Jaccard_τ07", "C. τ=0.7 关键字段集合 Jaccard")]):
    g = C.groupby("组", sort=False)[col]; y = np.arange(g.ngroups)[::-1]
    for yy, (k, v) in zip(y, g):
        ax.scatter(v.values, [yy] * len(v), s=30, color=BLUE, zorder=3); ax.plot([v.min(), v.max()], [yy, yy], color="#9fc3ea", lw=2)
    ax.set_yticks(y); ax.set_yticklabels(list(g.groups.keys())); ax.set_xlim(.5, 1.02); ax.set_title(t, loc="left", fontsize=10)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
fig.suptitle("图1  与主结果（攻击器种子 0、随机划分 42）的一致性：每个点为一个目标", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig1_seed_split.png"), bbox_inches="tight"); plt.close(fig)
# 图2 NEM K=3 与各数据集 K 饱和
f = os.path.join(A, "nem_k3.csv")
if os.path.exists(f):
    N = pd.read_csv(f); n4 = N[N.口径.str.startswith("规模 ≤4")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(n4))
    for k, K in enumerate(range(4)):
        axs[0].bar(x + (k - 1.5) * .2, n4[f"M{K}均值"], .19, color=["#cfcec8", "#9fc3ea", BLUE, "#0d366b"][k], label=f"M^({K})")
    axs[0].set_xticks(x); axs[0].set_xticklabels(n4.目标); axs[0].legend(frameon=False); axs[0].set_title("A. NEM 三台机组的 K 递进（规模 ≤4 正式口径）", loc="left", fontsize=10)
    axs[1].bar(x - .18, n4["M1/M2"], .34, color="#9fc3ea", label="M^(1)/M^(2)"); axs[1].bar(x + .18, n4["M2/M3"], .34, color=BLUE, label="M^(2)/M^(3)")
    for xi, v in zip(x + .18, n4["M2/M3"]):
        axs[1].text(xi, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    axs[1].set_ylim(.5, 1.08); axs[1].set_xticks(x); axs[1].set_xticklabels(n4.目标); axs[1].legend(frameon=False, loc="lower right")
    axs[1].set_title("B. 饱和比：第二个背景仍有增益，第三个背景几乎没有", loc="left", fontsize=10)
    for ax in axs:
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.suptitle("图2  NEM 的 K=3 证据（112 号只到规模 3，本号补做规模 4 的 27,405 个集合）", x=.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig2_nem_k3.png"), bbox_inches="tight"); plt.close(fig)
# 图3 τ 扫描
T = pd.read_csv(os.path.join(A, "tau_sweep.csv")); T["组合比"] = T.组合危险字段 / T.字段数; T["单字段比"] = T.单字段即危险 / T.字段数
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2)); col = {"RTS-GMLC": BLUE, "NEM": ORANGE, "PJM-load": AQUA, "PJM-gen/ic": "#008300", "CAISO-load": "#4a3aa7"}
for ds, g in T.groupby("数据"):
    m = g.groupby("τ")[["组合比", "单字段比", "仍暴露比例"]].mean()
    axs[0].plot(m.index, m.组合比, color=col[ds], lw=1.8, label=f"{ds} 组合危险"); axs[0].plot(m.index, m.单字段比, color=col[ds], lw=1, ls="--")
    axs[1].plot(m.index, m.仍暴露比例, color=col[ds], lw=1.8, label=ds)
axs[0].set_xlabel("阈值 τ"); axs[0].set_ylabel("占候选字段比例（实线：单看安全、组合危险；虚线：单字段即危险）"); axs[0].legend(frameon=False, fontsize=7.5)
axs[1].set_xlabel("阈值 τ"); axs[1].set_ylabel("单字段定级后仍暴露的危险组合比例"); axs[1].legend(frameon=False, fontsize=7.5)
for ax in axs:
    ax.grid(color=GRID); ax.set_axisbelow(True)
fig.suptitle("图3  阈值 τ 从 0.3 到 0.9：组合危险字段在绝大多数阈值上多于单字段危险字段", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig3_tau_sweep.png"), bbox_inches="tight"); plt.close(fig)
print("ok")
