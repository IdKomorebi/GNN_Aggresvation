# -*- coding: utf-8 -*-
"""DNN72 图：防护管线三策略曲线（含误差棒+重训证书）；跨 τ 稳定性与鲁棒核心。"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]; plt.rcParams["axes.unicode_minus"] = False
D = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation72"
BLUE="#2a78d6"; GREEN="#008300"; ORANGE="#eb6834"; MAG="#e87ba4"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3e0"
def style(ax):
    ax.set_facecolor("white")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)

curve = pd.read_csv(f"{D}/outputs/direct_greedy_curve.csv")
agg = pd.read_csv(f"{D}/outputs/seed_robustness_agg.csv")

# ============ 图1：防护管线三策略 ============
fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=300)
style(ax)
# 低阶贪心（3 种子误差棒）
ax.errorbar(agg.k, agg.worst_mean, yerr=agg.worst_std, color=BLUE, lw=2.2, marker="o",
            ms=5, capsize=3, zorder=5, label="低阶贪心（3 oracle 种子 ±σ）")
# 纯 oracle 贪心
d = curve[curve.strategy == "direct"]
ax.plot(d.k, d.worst_k200, color=ORANGE, lw=1.9, ls="--", marker=".", ms=4, zorder=4,
        label="纯 oracle 贪心（短视瓶颈优化）")
# 混合
h = curve[curve.strategy == "hybrid"]
ax.plot(h.k, h.worst_k200, color=GREEN, lw=2.4, marker="s", ms=5.5, zorder=6,
        label="混合：低阶骨架 24 + oracle 精修")
ax.annotate(f"k=27 达标 0.474", xy=(27, 0.474), xytext=(28.2, 0.56),
            fontsize=8.8, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
# 重训证书点
try:
    cert = pd.read_csv(f"{D}/outputs/retrain_certificates.csv")
    c0 = cert[cert.seed == 0]
    for _, r in c0.iterrows():
        ax.scatter(r.k, r.worst_retrain, marker="*", s=170, color=MAG, zorder=8,
                   edgecolors="white", linewidths=0.8,
                   label="重训证书（真值）" if _ == c0.index[0] else None)
except FileNotFoundError:
    pass
ax.axhline(0.5, color=INK2, lw=1.2, ls=(0, (4, 3)))
ax.text(0.7, 0.512, "目标 τ=0.5", fontsize=8.5, color=INK2)
ax.set_xlabel("防护预算 $|P|$（删除字段数）", color=INK, fontsize=10)
ax.set_ylabel("共享补集最坏机密泄露（oracle $K{=}200$）", color=INK, fontsize=9.5)
ax.set_title("防护管线对比：低阶定骨架 + oracle 修边界", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_ylim(0.35, 1.02); ax.set_xlim(-0.8, 31.5)
fig.tight_layout(); fig.savefig(f"{D}/figures/protection_pipeline_zh.png", facecolor="white", bbox_inches="tight")
plt.close(fig); print("fig1 done")

# ============ 图2：跨 τ 稳定性 + 鲁棒核心 ============
taus = pd.read_csv(f"{D}/outputs/tau_stability.csv")
core = pd.read_csv(f"{D}/outputs/tau_robust_core.csv")
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=300)
ax = axes[0]; style(ax)
for m, col in [(6, BLUE), (12, GREEN), (18, ORANGE)]:
    sub = taus[taus.prefix == m]
    labels = [f"{r.tau1}\nvs {r.tau2}" for r in sub.itertuples()]
    ax.plot(range(len(sub)), sub.jaccard, color=col, marker="o", ms=5, lw=1.8, label=f"前 {m} 步")
ax.set_xticks(range(len(sub))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.4, 1.05); ax.set_ylabel("防护序列前缀 Jaccard 重合度", color=INK, fontsize=9.5)
ax.set_xlabel("τ 组合", color=INK, fontsize=10)
ax.set_title("(a) 防护选择对风险阈值 τ 的稳定性", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

ax = axes[1]; style(ax)
core = core.sort_values("single_leak")
colors = [MAG if v < 0.3 else BLUE for v in core.single_leak]
ax.barh(range(len(core)), core.single_leak, color=colors, zorder=3)
ax.set_yticks(range(len(core)))
ax.set_yticklabels(core.field, fontsize=6.8)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(0.51, 0.2, "τ=0.5", fontsize=8, color=INK2)
ax.set_xlabel("单字段泄露 max$_c$ $v_c(\\{i\\})$", color=INK, fontsize=9.5)
ax.set_title("(b) τ-鲁棒核心防护字段（16 个）", fontsize=10.5, color=INK)
ax.text(0.52, 2.2, "粉色=单字段“安全”但因协同\n必须防护（单字段评分盲区）", fontsize=8, color=MAG)
fig.tight_layout(); fig.savefig(f"{D}/figures/tau_stability_zh.png", facecolor="white", bbox_inches="tight")
plt.close(fig); print("fig2 done")
