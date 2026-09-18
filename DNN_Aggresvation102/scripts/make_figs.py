# -*- coding: utf-8 -*-
"""102 号图：single vs multi 的保真与成本对照，以及真值攻击器族诊断。"""
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
B = pd.read_csv(A / "102_B.csv"); C = pd.read_csv(A / "102_C.csv"); Ad = pd.read_csv(A / "102_A.csv")
order = ["无 φ：x,x² ridge", "single-target φ+x,x²", "multi-target φ+x,x²", "multi-target 三种子集成+x,x²"]
short = {"无 φ：x,x² ridge": "无 φ\n(x,x² ridge)", "single-target φ+x,x²": "single-target\nφ+x,x²",
         "multi-target φ+x,x²": "multi-target\nφ+x,x²", "multi-target 三种子集成+x,x²": "multi-target\n三种子集成"}
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
x = np.arange(len(order)); w = 0.38
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = B[B.数据集 == ds].set_index("估计器").reindex(order)
    for k, (col, ttl) in enumerate([("V_MAE", "A. 集合价值 V 的 MAE"), ("M_MAE_K1", "B. M^(1) 的 MAE"),
                                    ("M_MAE_K2", "C. M^(2) 的 MAE"), ("τ0.7critical_recall_K2", "D. τ=0.7、K=2 的 critical 字段召回")]):
        ax[k].bar(x + (d_i - 0.5) * w, d[col].values, w * 0.9, color=color, label=ds.upper() if k == 0 else None)
        for xi, v in zip(x + (d_i - 0.5) * w, d[col].values):
            ax[k].text(xi, v, f" {v:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK, rotation=90)
        ax[k].set_title(ttl, loc="left", fontsize=10.5); ax[k].set_xticks(x); ax[k].set_xticklabels([short[o] for o in order], fontsize=8)
        ax[k].grid(axis="x", visible=False)
for k in range(3): ax[k].set_ylim(0, max(B[["V_MAE", "M_MAE_K1", "M_MAE_K2"]].max()) * 1.35)
ax[3].set_ylim(0, 1.18); ax[0].legend(frameon=False, fontsize=9)
fig.suptitle("图1  102 号（P0-2）：single-target 并不优于 multi-target；去掉学出的 φ 后明显变差（k3 全部 11,521 个集合 × 12 目标）", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "fig1_single_vs_multi.png", bbox_inches="tight"); plt.close(fig); print("-> fig1")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
d = C.dropna(subset=["折算每集合12目标ms"])
lbl = {"single-target（每目标一次查询）": "single-target\n(每目标一次查询)", "multi-target（一次查询给 12 目标）": "multi-target\n(一次查询 12 目标)"}
x = np.arange(2); w = 0.38
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    v = [d[(d.数据集 == ds) & (d.估计器 == k)]["折算每集合12目标ms"].values[0] for k in lbl]
    ax[0].bar(x + (d_i - 0.5) * w, v, w * 0.9, color=color, label=ds.upper())
    for xi, vv in zip(x + (d_i - 0.5) * w, v): ax[0].text(xi, vv + 1, f"{vv:.1f}", ha="center", fontsize=8.5)
    tr = C[(C.数据集 == ds)]["训练秒"].dropna().values
    ax[1].bar(d_i, tr[0] if len(tr) else 0, 0.5, color=color)
    ax[1].text(d_i, (tr[0] if len(tr) else 0) + 8, f"{tr[0]:.0f}s", ha="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(list(lbl.values()), fontsize=9); ax[0].set_ylabel("每集合 12 目标的查询耗时（ms）")
ax[0].legend(frameon=False); ax[0].grid(axis="x", visible=False); ax[0].set_title("A. 查询成本：single-target 高一个量级", loc="left", fontsize=10.5)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["PJM", "CAISO"]); ax[1].set_ylabel("12 个目标的预学习总秒数")
ax[1].grid(axis="x", visible=False); ax[1].set_title("B. 预学习成本：multi-target 只需 27s / 40s（单模型）", loc="left", fontsize=10.5)
fig.suptitle("图2  single-target 的代价：查询 12 倍、训练 12 倍，而精度并不更好", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig2_single_cost.png", bbox_inches="tight"); plt.close(fig); print("-> fig2")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = Ad[Ad.数据集 == ds]
    ax[0].plot(d.规模, d.单目标DNN, color=color, marker="o", lw=2, label=f"{ds.upper()} 单目标 DNN")
    ax[0].plot(d.规模, d.多目标DNN, color=color, marker="s", ls="--", lw=1.6, label=f"{ds.upper()} 多目标 DNN")
    ax[0].plot(d.规模, d.树模型, color=color, marker="^", ls=":", lw=1.6, label=f"{ds.upper()} 梯度提升树")
    ax[1].plot(d.规模, d.单目标减多目标均值, color=color, marker="o", lw=2, label=ds.upper())
ax[0].set_xticks([1, 2, 3]); ax[0].set_xlabel("字段集合规模"); ax[0].set_ylabel("测试 R² 均值"); ax[0].legend(frameon=False, fontsize=8, ncol=2)
ax[0].set_title("A. 三类攻击器的平均推断能力", loc="left", fontsize=10.5)
ax[1].axhline(0, color="#8a8984", lw=0.8); ax[1].set_xticks([1, 2, 3]); ax[1].set_xlabel("字段集合规模"); ax[1].set_ylabel("单目标 − 多目标 DNN")
ax[1].legend(frameon=False); ax[1].set_title("B. 单目标 DNN 的增益很小但一致（≤0.009）", loc="left", fontsize=10.5)
fig.suptitle("图3  真值攻击器族诊断：单目标 DNN 略强于多目标 DNN，但规模 3 上树模型最强 ⟹ 攻击器族三者都要", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(F / "fig3_attacker_family.png", bbox_inches="tight"); plt.close(fig); print("-> fig3")
