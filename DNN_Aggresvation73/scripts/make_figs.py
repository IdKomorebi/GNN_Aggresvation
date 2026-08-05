# -*- coding: utf-8 -*-
"""DNN73 图：训练方案消融——有/无随机失活的通用模型对重训真值的保真度。"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]; plt.rcParams["axes.unicode_minus"] = False
D = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation73"
BLUE="#2a78d6"; GREEN="#008300"; ORANGE="#eb6834"; MAG="#e87ba4"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3e0"
COL = {"none": ORANGE, "bern50": GREEN, "ours": BLUE}
LAB = {"none": "无随机失活（置零口径）", "bern50": "伯努利(0.5)失活", "ours": "两段式尺寸均匀（本文）"}
MK = {"none": "s", "bern50": "^", "ours": "o"}
ORDER = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-44"]
def style(ax):
    ax.set_facecolor("white")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)

ov = pd.read_csv(f"{D}/outputs/metrics_overall.csv")
bs = pd.read_csv(f"{D}/outputs/metrics_by_size.csv")
merged = pd.read_csv(f"{D}/outputs/merged_estimates.csv")
ARCH = "gnn" if (ov.arch == "gnn").any() else "mlp"
ARCHNAME = {"gnn": "图 oracle", "mlp": "MLP oracle"}

# ============ 图1：(a) MAE vs K  (b) MAE vs 集合规模 (K=0) ============
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=300)
ax = axes[0]; style(ax)
Ks = [0, 10, 50, 200]; xp = np.arange(len(Ks))
for sc in ["none", "bern50", "ours"]:
    s = ov[(ov.arch == ARCH) & (ov.scheme == sc)].set_index("K").reindex(Ks)
    ax.plot(xp, s.mae, color=COL[sc], marker=MK[sc], ms=6, lw=2.2 if sc == "ours" else 1.8,
            zorder=5 if sc == "ours" else 4, label=LAB[sc])
    for xi, v in zip(xp, s.mae):
        if not np.isnan(v):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5, color=COL[sc])
ax.set_xticks(xp); ax.set_xticklabels([str(k) for k in Ks])
ax.set_xlabel("查询时微调步数 $K$", color=INK, fontsize=10)
ax.set_ylabel("对重训真值的 MAE", color=INK, fontsize=10)
ax.set_title(f"(a) 训练方案对保真度的影响（{ARCHNAME[ARCH]}）", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5)

ax = axes[1]; style(ax)
sub = bs[(bs.arch == ARCH) & (bs.K == 0)]
bands = [b for b in ORDER if b in sub.band.values]
xb = np.arange(len(bands))
for sc in ["none", "bern50", "ours"]:
    s = sub[sub.scheme == sc].set_index("band").reindex(bands)
    ax.plot(xb, s.mae, color=COL[sc], marker=MK[sc], ms=6, lw=2.2 if sc == "ours" else 1.8,
            zorder=5 if sc == "ours" else 4, label=LAB[sc])
ax.set_xticks(xb); ax.set_xticklabels(bands)
ax.set_xlabel("字段集合规模 $|S|$", color=INK, fontsize=10)
ax.set_ylabel("对重训真值的 MAE（$K{=}0$）", color=INK, fontsize=10)
ax.set_title("(b) 无失活模型在小集合上崩溃", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{D}/figures/dropout_ablation_zh.png", facecolor="white", bbox_inches="tight")
plt.close(fig); print("fig1 done")

# ============ 图2：K=0 校准散点（三方案并排）============
m0 = merged[(merged.arch == ARCH) & (merged.K == 0)]
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), dpi=300)
for ax, sc in zip(axes, ["none", "bern50", "ours"]):
    style(ax)
    s = m0[m0.scheme == sc]
    ax.scatter(s.truth, s.est, s=9, color=COL[sc], alpha=0.28, linewidths=0, zorder=3)
    ax.plot([0, 1], [0, 1], color=INK2, ls=(0, (4, 3)), lw=1.1, zorder=4)
    mae = (s.est - s.truth).abs().mean()
    from scipy.stats import spearmanr
    sp = spearmanr(s.est, s.truth).correlation
    ax.text(0.04, 0.95, f"MAE={mae:.3f}\nSpearman={sp:.3f}", transform=ax.transAxes,
            va="top", fontsize=9.5, color=INK)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("重训真值 $v_c(S)$", color=INK, fontsize=10)
    if sc == "none": ax.set_ylabel("通用模型估计 $\\hat v_c(S)$", color=INK, fontsize=10)
    ax.set_title(LAB[sc], fontsize=10.5, color=COL[sc])
fig.suptitle(f"不微调（$K{{=}}0$）时三种训练方案的校准（{ARCHNAME[ARCH]}）", fontsize=11, color=INK, y=1.02)
fig.tight_layout(); fig.savefig(f"{D}/figures/dropout_calibration_zh.png", facecolor="white", bbox_inches="tight")
plt.close(fig); print("fig2 done")
