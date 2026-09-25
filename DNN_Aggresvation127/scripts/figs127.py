# -*- coding: utf-8 -*-
"""127 号：中文图。图 1 耗时—精度（全部外部方法）；图 2 本文与 TabPFN 分数据组对比。"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
FG = os.path.join(ROOT, "figures"); os.makedirs(FG, exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
sys.path.insert(0, os.path.join(REPO, "paper", "claude", "V5.2_en", "scripts"))
import importlib; mt = importlib.import_module("make_tables"); T = mt._baseline_times()
M = pd.read_csv(os.path.join(ROOT, "outputs", "analysis", "summary.csv"), index_col=0)
NM = {"lin": "逐集合线性回归", "poly": "逐集合二次回归", "dropout": "均值代替（Dropout）", "lazyvi": "LazyVI", "ws": "热启动+早停",
      "surrogate": "掩码代理模型直接输出", "tabpfn": "TabPFN v2", "ours": "本文", "retrain": "串行重训（真值）"}
STY = {"lin": ("#8a8984", "v"), "poly": ("#8a8984", "^"), "dropout": ("#8a8984", "X"), "ws": ("#8a8984", "D"), "surrogate": ("#8a8984", "P"),
       "tabpfn": ("#eb6834", "o"), "ours": ("#2a78d6", "o"), "retrain": ("#0b0b0b", "s")}
tm = {k: np.sqrt(T[k][0] * T[k][1]) for k in T}
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, c, lab, lim in [(axs[0], "M2误差", "M^(2) 误差", (-0.005, 0.26)), (axs[1], "认证前1", "认证前 1 比", (0.25, 1.04))]:
    for k in STY:
        y = (0.0 if c == "M2误差" else 1.0) if k == "retrain" else M.loc[k, c]; cl, mk = STY[k]
        ax.errorbar(tm[k], y, xerr=[[tm[k] - T[k][0]], [T[k][1] - tm[k]]], fmt=mk, ms=8, color=cl, elinewidth=1, label=NM[k])
    ax.set_xscale("log"); ax.set_ylim(*lim); ax.set_xlabel("每集合耗时（秒，对数轴；横线为五组范围）"); ax.set_ylabel(lab); ax.grid(color="#e6e5e0")
axs[0].legend(frameon=False, fontsize=8, loc="upper right"); axs[0].set_title("A. M^(2) 误差对耗时", loc="left"); axs[1].set_title("B. 认证前 1 比对耗时", loc="left")
ax = axs[2]; ks = sorted(["dropout", "lin", "poly", "surrogate", "ws", "lazyvi", "ours", "tabpfn"], key=lambda k: -M.loc[k, "样本V误差"])
ax.barh(range(len(ks)), [M.loc[k, "样本V误差"] for k in ks], color=[STY.get(k, ("#8a8984",))[0] if k in ("ours", "tabpfn") else "#b4b3ad" for k in ks])
for i, k in enumerate(ks):
    ax.text(M.loc[k, "样本V误差"] + .005, i, f"{M.loc[k, '样本V误差']:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(ks))); ax.set_yticklabels([NM[k] for k in ks]); ax.set_xlabel("V 误差（每组同一批 200 个集合）"); ax.set_title("C. 同一批 200 个集合上的 V 误差（含 LazyVI）", loc="left")
fig.suptitle("图1  与外部免重训方法的对比：只有 TabPFN 与本文精度相当，但每集合慢 13–110 倍；其余方法 M 误差为本文的 3–16 倍", x=.01, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig1_baselines.png"), bbox_inches="tight"); plt.close(fig)
R = pd.read_csv(os.path.join(ROOT, "outputs", "analysis", "full_by_target.csv")); r = R[R.方法.isin(["ours", "tabpfn"])]
DS = ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen/ic", "CAISO-load"]
fig, axs = plt.subplots(1, 4, figsize=(19, 4.3))
for ax, c, lab in zip(axs, ["V误差", "M2误差", "Γ2误差", "关键召回07"], ["V 误差", "M^(2) 误差", "Γ^(2) 误差", "关键召回 τ=0.7（δ=0）"]):
    P = r.pivot_table(index="数据", columns="方法", values=c).reindex(DS)
    ax.bar(np.arange(5) - .2, P["ours"], .4, color="#2a78d6", label="本文"); ax.bar(np.arange(5) + .2, P["tabpfn"], .4, color="#eb6834", label="TabPFN v2")
    ax.set_xticks(range(5)); ax.set_xticklabels(DS, fontsize=7.5, rotation=15); ax.set_title(lab, loc="left"); ax.grid(color="#e6e5e0", axis="y"); ax.set_axisbelow(True)
axs[0].legend(frameon=False)
fig.suptitle("图2  本文与 TabPFN 分数据组：TabPFN 的优势集中在 PJM（训练行最少，本文在此系统低估 0.03–0.04）；NEM 与 Γ 上本文更准", x=.01, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .91]); fig.savefig(os.path.join(FG, "fig2_ours_vs_tabpfn.png"), bbox_inches="tight"); plt.close(fig)
print("图已生成")
