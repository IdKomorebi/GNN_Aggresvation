# -*- coding: utf-8 -*-
"""DNN70 图：低阶代理保真度 / 校准散点 / 残差分层 / Apriori 层级性。"""
import numpy as np, pandas as pd, json, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]; plt.rcParams["axes.unicode_minus"] = False
OUT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation70"
BLUE="#2a78d6"; GREEN="#008300"; ORANGE="#eb6834"; MAG="#e87ba4"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3e0"
def style(ax):
    ax.set_facecolor("white")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)

df = pd.read_csv(f"{OUT}/outputs/surrogate_predictions.csv")
cal = pd.read_csv(f"{OUT}/outputs/isotonic_calibration.csv")
res = pd.read_csv(f"{OUT}/outputs/residual_by_size.csv")
hier = pd.read_csv(f"{OUT}/outputs/hierarchy_test.csv")

# ---------- 图1：(a) 各阶代理保真度  (b) 最强三元组校准散点 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=300)
ax = axes[0]; style(ax)
sub = cal[cal.subset == "size>=5"]
x = np.arange(3); w = 0.36
order = ["最强单字段","最强字段对","最强三元组"]
sm = sub.set_index("model").loc[order]
ax.bar(x - w/2, sm.raw_mae, w, color=MAG, zorder=3, label="原始 MAE（未校准）")
ax.bar(x + w/2, sm.cal_mae, w, color=BLUE, zorder=3, label="等距校准后 MAE")
for xi,(r,c) in enumerate(zip(sm.raw_mae, sm.cal_mae)):
    ax.text(xi-w/2, r+0.006, f"{r:.3f}", ha="center", fontsize=8, color=INK)
    ax.text(xi+w/2, c+0.006, f"{c:.3f}", ha="center", fontsize=8, color=INK)
ax.axhline(0.012, color=INK2, ls=(0,(4,3)), lw=1); ax.text(2.3, 0.018, "3σ 噪声底", fontsize=8, color=INK2, ha="right")
ax.set_xticks(x); ax.set_xticklabels([f"{o}\nSpr={s:.3f}" for o,s in zip(order, sm.cal_spearman)], fontsize=8.5)
ax.set_ylabel("对宽尺寸真值 MAE", color=INK, fontsize=10); ax.set_ylim(0,0.28)
ax.set_title("(a) 低阶代理保真度（|S|≥5 留出）", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# (b) 校准散点：最强三元组
ax = axes[1]; style(ax)
d5 = df[df["size"]>=5]
preds=[]; truths=[]
for held in d5.sid.unique():
    tr=d5[d5.sid!=held]; te=d5[d5.sid==held]
    iso=IsotonicRegression(out_of_bounds="clip"); iso.fit(tr["m3"].values, tr["truth"].values)
    preds.extend(iso.predict(te["m3"].values)); truths.extend(te["truth"].values)
preds=np.array(preds); truths=np.array(truths)
ax.scatter(truths, preds, s=14, color=BLUE, alpha=0.35, linewidths=0, zorder=3)
ax.plot([0,1],[0,1], color=INK2, ls=(0,(4,3)), lw=1.1)
ax.text(0.04,0.93, f"最强三元组 + 等距校准\nMAE={np.abs(preds-truths).mean():.3f}\nSpearman={spearmanr(preds,truths).correlation:.3f}",
        transform=ax.transAxes, va="top", fontsize=9.5, color=INK)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel("重训真值 $v_c(S)$", color=INK, fontsize=10)
ax.set_ylabel("低阶代理预测（校准后）", color=INK, fontsize=10)
ax.set_title("(b) 低阶代理复原任意集合敏感度", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/surrogate_fidelity_zh.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
print("fig1 done")

# ---------- 图2：(a) 残差按规模  (b) Apriori 层级性直方图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=300)
ax = axes[0]; style(ax)
xb = np.arange(len(res));
ax.bar(xb, res.resid_m2_mean, 0.5, color=BLUE, zorder=3, label="残差=真值−最强字段对")
ax.plot(xb, res.resid_m3_mean, color=ORANGE, marker="o", lw=2, zorder=4, label="残差（用到三元组后）")
ax.axhline(0.012, color=INK2, ls=(0,(4,3)), lw=1); ax.text(len(res)-0.5, 0.02, "3σ 噪声底", fontsize=8, color=INK2, ha="right")
ax.set_xticks(xb); ax.set_xticklabels(res.band); ax.set_ylim(0,0.2)
ax.set_xlabel("集合规模 $|S|$", color=INK, fontsize=10)
ax.set_ylabel("平均残差（未被低阶捕获的泄露）", color=INK, fontsize=9.5)
ax.set_title("(a) 高阶/累积残差随规模的变化", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

ax = axes[1]; style(ax)
ax.hist(hier.best_child_syn2, bins=24, color=BLUE, alpha=0.75, zorder=3)
ax.axvline(0.10, color=ORANGE, lw=1.6, zorder=4); ax.text(0.108, ax.get_ylim()[1]*0.9, "显著阈值 0.10", color=ORANGE, fontsize=8.5)
frac = (hier.best_child_syn2>0.10).mean()
ax.text(0.30, ax.get_ylim()[1]*0.75, f"{frac:.0%} 的显著三阶协同\n含显著二阶子对\n（Apriori 剪枝合法性）", fontsize=9, color=INK)
ax.set_xlabel("强三元组的最强二元子对 syn2", color=INK, fontsize=10)
ax.set_ylabel("频数", color=INK, fontsize=10)
ax.set_title("(b) 层级性：强三阶多含强二阶子结构", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/hierarchy_residual_zh.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
print("fig2 done")
