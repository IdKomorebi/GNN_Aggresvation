# -*- coding: utf-8 -*-
"""103 号图：基线阶梯、消融、攻击器族构成、规模分层误差。"""
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
BLUE, ORANGE, AQUA, YEL, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c", "#2b2b2a"
B = pd.read_csv(A / "103_B.csv"); S = pd.read_csv(A / "103_A.csv"); C = pd.read_csv(A / "103_C.csv")
main = ["raw ridge", "raw+x² ridge", "随机特征 ridge", "共享输出头（direct）", "φ ridge", "φ+x,x²（单种子）", "φ+x,x²（三种子集成）"]
lab = {"raw ridge": "raw\nridge", "raw+x² ridge": "raw+x²\nridge", "随机特征 ridge": "随机特征\nridge", "共享输出头（direct）": "共享输出头\n(direct)",
       "φ ridge": "φ\nridge", "φ+x,x²（单种子）": "φ+x,x²\n(单种子)", "φ+x,x²（三种子集成）": "φ+x,x²\n(三种子集成)"}
fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.5))
x = np.arange(len(main)); w = 0.38
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = B[B.数据集 == ds].set_index("估计器").reindex(main)
    for k, (col, ttl) in enumerate([("V_MAE", "A. V 的 MAE"), ("危险漏判", "B. 危险漏判率（真值>0.7 判为≤0.7）"),
                                    ("M_MAE_K2", "C. M^(2) 的 MAE"), ("见证召回_K2", "D. 见证召回 K=2")]):
        ax[k].bar(x + (d_i - .5) * w, d[col].values, w * .9, color=color, label=ds.upper() if k == 0 else None)
        for xi, v in zip(x + (d_i - .5) * w, d[col].values):
            ax[k].text(xi, v, f" {v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color=INK)
        ax[k].set_xticks(x); ax[k].set_xticklabels([lab[m] for m in main], fontsize=7.5); ax[k].grid(axis="x", visible=False)
        ax[k].set_title(ttl, loc="left", fontsize=10.5)
ax[0].set_ylim(0, 0.16); ax[1].set_ylim(0, 0.52); ax[2].set_ylim(0, 0.15); ax[3].set_ylim(0, 1.15); ax[0].legend(frameon=False, fontsize=9)
fig.suptitle("图1  RQ-A2（正式三攻击器真值，k3 全部集合）：共享输出头明显更差；学出的 φ 是关键；x,x² 旁路在强真值下增益消失", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "fig1_baselines.png", bbox_inches="tight"); plt.close(fig); print("-> fig1")

abl = ["掩码消融：无掩码", "掩码消融：Bernoulli", "φ+x,x²（单种子）", "元训练 φ+x,x²", "single-target φ+x,x²（102 号）"]
al = {"掩码消融：无掩码": "无掩码", "掩码消融：Bernoulli": "Bernoulli\n掩码", "φ+x,x²（单种子）": "uniform 掩码\n(主口径)",
      "元训练 φ+x,x²": "元训练 φ", "single-target φ+x,x²（102 号）": "single-target"}
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
x = np.arange(len(abl))
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = B[B.数据集 == ds].set_index("估计器").reindex(abl)
    for k, (col, ttl) in enumerate([("M_MAE_K1", "A. M^(1) MAE"), ("M_MAE_K2", "B. M^(2) MAE"), ("见证召回_K2", "C. 见证召回 K=2")]):
        v = d[col].values.astype(float)
        ax[k].bar(x + (d_i - .5) * w, np.nan_to_num(v), w * .9, color=color, label=ds.upper() if k == 0 else None)
        for xi, vv in zip(x + (d_i - .5) * w, v):
            if not np.isnan(vv): ax[k].text(xi, vv, f" {vv:.3f}", ha="center", va="bottom", fontsize=7, rotation=90, color=INK)
        ax[k].set_xticks(x); ax[k].set_xticklabels([al[m] for m in abl], fontsize=8); ax[k].grid(axis="x", visible=False)
        ax[k].set_title(ttl, loc="left", fontsize=10.5)
ax[0].set_ylim(0, 0.062); ax[1].set_ylim(0, 0.075); ax[2].set_ylim(0, 1.15); ax[0].legend(frameon=False, fontsize=9)
fig.suptitle("图2  RQ-A3 消融：掩码分布几乎不影响（闭式读出自适应）；元训练 φ 在 PJM 的 M 与见证上最好；single-target 最差", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig2_ablation.png", bbox_inches="tight"); plt.close(fig); print("-> fig2")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for d_i, ds in enumerate(["pjm", "caiso"]):
    d = S[S.数据集 == ds]; bot = np.zeros(3)
    for col, color, nm in [("单目标DNN被选中", BLUE, "单目标 DNN"), ("多目标DNN被选中", AQUA, "多目标 DNN"), ("树模型被选中", ORANGE, "梯度提升树")]:
        ax[0].bar(np.arange(3) + (d_i - .5) * 0.38, d[col].values, 0.34, bottom=bot, color=color, alpha=1 if d_i == 0 else 0.6,
                  label=nm if d_i == 0 else None); bot += d[col].values
    ax[1].plot(d.规模, d.正式减仅多目标, marker="o", lw=2, color=[BLUE, ORANGE][d_i], label=ds.upper())
ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels(["规模 1", "规模 2", "规模 3"]); ax[0].set_ylim(0, 1.05)
ax[0].legend(frameon=False, fontsize=8.5, ncol=3); ax[0].grid(axis="x", visible=False)
ax[0].set_title("A. 各攻击器被 val 选中的占比（深 PJM / 浅 CAISO）", loc="left", fontsize=10.5)
ax[1].set_xticks([1, 2, 3]); ax[1].set_xlabel("集合规模"); ax[1].set_ylabel("正式真值 − 仅多目标 DNN")
ax[1].legend(frameon=False); ax[1].set_title("B. 单一攻击器低估经验攻击能力的幅度", loc="left", fontsize=10.5)
fig.suptitle("图3  正式攻击器族：规模越大，树模型越常胜出；只用一个 DNN 会低估 0.005–0.027", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig3_attacker_share.png", bbox_inches="tight"); plt.close(fig); print("-> fig3")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for k, ds in enumerate(["pjm", "caiso"]):
    d = C[C.数据集 == ds]
    for nm, color in [("raw+x² ridge", GRAY), ("共享输出头（direct）", YEL), ("φ+x,x²（单种子）", BLUE), ("φ+x,x²（三种子集成）", ORANGE)]:
        q = d[d.估计器 == nm]
        ax[k].plot(q.规模, q.V_MAE, marker="o", lw=2, color=color, label=nm)
    ax[k].set_xticks([1, 2, 3]); ax[k].set_xlabel("集合规模"); ax[k].set_title(ds.upper(), loc="left", fontsize=10.5)
ax[0].set_ylabel("V 的 MAE"); ax[1].legend(frameon=False, fontsize=8.5)
fig.suptitle("图4  RQ-A1：误差随集合规模增长，但学出的 φ 把增长压住（规模 3 上 0.028–0.033 对 0.069–0.092）", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig4_size_layer.png", bbox_inches="tight"); plt.close(fig); print("-> fig4")
