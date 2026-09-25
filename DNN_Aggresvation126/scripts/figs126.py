# -*- coding: utf-8 -*-
"""126 号：耗时—精度对照（中文图）。图 1：每集合耗时（原实现 / 快速实现）；图 2：耗时对 M^(2) 误差（对数横轴）。"""
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
FG = os.path.join(ROOT, "figures"); os.makedirs(FG, exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
NAME = {"head1": "共享输出头×1", "head3": "共享输出头×3", "lin": "线性读出（无主干）", "poly": "[x,x²] 读出（无主干）", "polyS": "集合内二阶字典",
        "rand1": "随机主干读出×1", "D1": "只预测目标×1", "E": "只预测目标×3（旧 E）", "Ecat": "只预测目标 3 种子特征拼接（1 次读出）",
        "C1": "重建×3+截断", "C2": "重建⊕随机×1+截断", "C3": "重建⊕随机×3+截断（本文）", "ft": "热启动微调×1"}
ORDER = list(NAME)
A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "outputs", "time_[A-Z]*.csv"))]).groupby("方法").每集合ms.mean()
F = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "outputs", "time_fast_*.csv"))]).groupby("方法").每集合ms.mean()
fig, ax = plt.subplots(figsize=(13, 5.2)); y = np.arange(len(ORDER))[::-1]
ax.barh(y + .2, [A[m] for m in ORDER], .4, color="#b4b3ad", label="原实现")
ax.barh(y - .2, [F[m] for m in ORDER], .4, color=["#0d366b" if m == "C3" else "#2a78d6" for m in ORDER], label="等价快速实现（结果逐位相同）")
for yy, m in zip(y, ORDER):
    ax.text(F[m] * 1.1, yy - .2, f"{F[m]:.1f} ms", va="center", fontsize=7.5); ax.text(A[m] * 1.1, yy + .2, f"{A[m]:.1f}", va="center", fontsize=7, color="#6f6e69")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in ORDER]); ax.set_xlabel("每集合耗时（毫秒，对数轴；同一空闲 GPU、每组同一批 200 个集合、批 8、五组平均）")
seq = np.mean([np.load(os.path.join(REPO, "DNN_Aggresvation118", "outputs", "est", t, "seq.npz"))["t"].sum(1).mean() for t in ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen_ic", "CAISO-load"]])
ax.axvline(seq * 1000, color="#eb6834", ls="--", lw=1); ax.text(seq * 1000 * .95, y[len(y) // 2], f"串行重训\n≈{seq:.0f} s/集合", color="#eb6834", ha="right", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.grid(color="#e6e5e0", axis="x"); ax.set_axisbelow(True)
fig.suptitle("图1  各方法每集合耗时：读出耗时随特征维度增长；×3 = 三次独立读出再平均；快速实现（全体 Gram 一次、各折做减法、λ 批量求解）约快 2.8 倍",
             x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig1_time_per_method.png"), bbox_inches="tight"); plt.close(fig)

# 图 2：耗时 vs M 误差（十个目标平均）
V = pd.read_csv(os.path.join(REPO, "DNN_Aggresvation118", "outputs", "analysis", "variants_by_target.csv")).groupby("变体")[["M2误差", "认证前1"]].mean()
S = pd.read_csv(os.path.join(REPO, "DNN_Aggresvation125", "outputs", "select", "candidates_by_target.csv"))
S = S.groupby(["候选", "实现"])[["M2误差", "认证前1"]].mean().groupby("候选").mean()
R = pd.read_csv(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "analysis", "variants_by_target.csv")).groupby("变体")[["M2误差", "认证前1"]].mean()
pts = {"head3": V.loc["head3"], "lin": V.loc["lin"], "poly": V.loc["poly"], "polyS": V.loc["polyS"], "rand1": V.loc["rand1"], "D1": V.loc["D1"],
       "E": S.loc["E_old"], "C1": S.loc["C1"], "C2": S.loc["C2"], "C3": S.loc["C3"]}
OFFB = {"rand1": (6, -12), "D1": (6, 4), "E": (6, 6), "C1": (-70, 10), "C2": (-95, -14), "C3": (6, -4), "rand_cy": (6, -12)}
OFFC = {"rand1": (6, 4), "D1": (6, -12), "E": (6, 6), "C1": (-60, 10), "C2": (8, -16), "C3": (6, -4), "rand_cy": (6, 6)}
fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))
def draw(ax, col, zoom, OFF=OFFB):
    for m, r in pts.items():
        if zoom and m in ("head3", "lin", "poly", "polyS"):
            continue
        c = "#0d366b" if m == "C3" else ("#b4b3ad" if m in ("head3", "lin", "poly", "polyS") else "#2a78d6")
        ax.scatter(F[m], r[col], s=70 if m == "C3" else 40, color=c, zorder=3, edgecolor="white")
        if zoom or m in ("head3", "lin", "poly", "polyS"):
            ax.annotate(NAME[m], (F[m], r[col]), xytext=OFF.get(m, (5, 3)) if zoom else (5, 3), textcoords="offset points", fontsize=7.5)
    rc = R.loc["random_cy"]; ax.scatter(F["rand1"], rc[col], s=40, color="#1baf7a", zorder=3, edgecolor="white")
    if zoom:
        ax.annotate("随机主干读出×1+截断", (F["rand1"], rc[col]), xytext=OFF["rand_cy"], textcoords="offset points", fontsize=7.5, color="#1baf7a")
    ax.set_xscale("log"); ax.set_xlabel("每集合耗时（毫秒，快速实现，对数轴）"); ax.grid(color="#e6e5e0"); ax.set_axisbelow(True)
draw(axs[0], "M2误差", False); axs[0].set_ylim(0, .16); axs[0].set_ylabel("M^(2) 误差（越低越好）")
axs[0].add_patch(plt.Rectangle((5, .012), 70, .013, fill=False, ec="#eb6834", lw=1)); axs[0].text(5, .027, "右图放大", color="#eb6834", fontsize=8)
axs[0].set_title("A. 全部方法（蓝点见 B 放大）", loc="left", fontsize=9.5)
draw(axs[1], "M2误差", True); axs[1].set_xlim(5, 90); axs[1].set_ylim(.013, .024); axs[1].set_ylabel("M^(2) 误差")
axs[1].set_title("B. 放大：带主干的读出", loc="left", fontsize=9.5)
draw(axs[2], "认证前1", True, OFFC); axs[2].set_xlim(5, 90); axs[2].set_ylim(.9, .97); axs[2].set_ylabel("认证前 1 比（越高越好）")
axs[2].set_title("C. 认证前 1 比（带主干的读出）", loc="left", fontsize=9.5)
fig.suptitle("图2  耗时—精度：共享输出头最快但 M 误差约为本文的 4 倍；单种子拼接（C2）耗时 1/3、平均精度接近本文但随种子波动；本文（C3）在精度上最好",
             x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig2_time_vs_accuracy.png"), bbox_inches="tight"); plt.close(fig)
print("图已生成")
