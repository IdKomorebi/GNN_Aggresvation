# -*- coding: utf-8 -*-
"""DNN71 图：防护集覆盖曲线/近最优+次模；oracle 验证的真实最坏泄露。"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]; plt.rcParams["axes.unicode_minus"] = False
D = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation71"
BLUE="#2a78d6"; GREEN="#008300"; ORANGE="#eb6834"; MAG="#e87ba4"; VIO="#4a3aa7"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3e0"
def style(ax):
    ax.set_facecolor("white")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)

curves = pd.read_csv(f"{D}/outputs/protection_curves.csv")
gvo = pd.read_csv(f"{D}/outputs/greedy_vs_optimal.csv")
sub = pd.read_csv(f"{D}/outputs/submodularity_check.csv")
val = pd.read_csv(f"{D}/outputs/protection_validation.csv")

# ========== 图1：(a) 覆盖曲线 vs 基线  (b) 次模递减收益 ==========
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), dpi=300)
ax = axes[0]; style(ax)
c = curves[curves.tau == 0.5]
ax.plot(c.k, c.greedy_cov, color=BLUE, lw=2.2, marker="o", ms=3, zorder=5, label="贪心（协同感知）")
ax.plot(c.k, c.degree_cov, color=GREEN, lw=1.8, ls="--", zorder=4, label="加权度基线")
ax.plot(c.k, c.single_cov, color=ORANGE, lw=1.8, ls="-.", zorder=4, label="单字段泄露基线")
ax.plot(c.k, c.random_cov, color=INK2, lw=1.6, ls=":", zorder=3, label="随机（20 次均值）")
ax.axhline(1.0, color=GRID, lw=1)
kstar = int(c[c.greedy_cov >= 0.999].k.min())
ax.axvline(kstar, color=MAG, lw=1.2, ls=(0,(3,2))); ax.text(kstar+0.6, 0.72, f"$k^*$={kstar}\n(覆盖全部\n危险项)", color=MAG, fontsize=8.5)
ax.set_xlabel("防护集大小 $|P|$（删除字段数）", color=INK, fontsize=10)
ax.set_ylabel("已中和危险泄露权重占比", color=INK, fontsize=10)
ax.set_title("(a) 防护覆盖曲线（τ=0.5，贪心 vs 基线）", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.3, loc="lower right", bbox_to_anchor=(0.99,0.02)); ax.set_ylim(0,1.05)

ax = axes[1]; style(ax)
band = pd.cut(sub.Psize, [-1,4,9,14,19], labels=["0-4","5-9","10-14","15-19"])
mg = sub.groupby(band, observed=True).marginal_gain.agg(["mean","std"])
xb = np.arange(len(mg))
ax.bar(xb, mg["mean"], 0.55, yerr=mg["std"], color=BLUE, zorder=3, error_kw=dict(ecolor=INK2, capsize=3, lw=1))
ax.set_xticks(xb); ax.set_xticklabels(mg.index)
ax.set_xlabel("已选防护集大小 $|P|$", color=INK, fontsize=10)
ax.set_ylabel("新增一个字段的边际覆盖收益", color=INK, fontsize=9.5)
ax.set_title("(b) 次模性：边际收益递减", fontsize=10.5, color=INK)
# 内嵌近最优表述
txt = "贪心 vs 暴力最优覆盖：\n" + "  ".join(f"k={int(r.k)}:{r.ratio:.3f}" for r in gvo.itertuples())
ax.text(0.5, -0.34, txt + "\n（均 = 1.000，远超 (1−1/e)=0.632 保证）", transform=ax.transAxes,
        ha="center", fontsize=8.2, color=INK2)
fig.subplots_adjust(bottom=0.28)
fig.savefig(f"{D}/figures/protection_greedy_zh.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
print("fig1 done")

# ========== 图2：oracle 验证的真实最坏泄露 ==========
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), dpi=300)
labelmap = {"greedy":"贪心（协同感知）","degree":"加权度基线","single_leak":"单字段泄露基线","random":"随机"}
colmap = {"greedy":BLUE,"degree":GREEN,"single_leak":ORANGE,"random":INK2}
stylemap = {"greedy":"-","degree":"--","single_leak":"-.","random":":"}
for metric, ax, ttl in [("worst_conf_r2", axes[0], "(a) 最坏机密字段真实泄露"),
                        ("mean_conf_r2", axes[1], "(b) 平均机密字段真实泄露")]:
    style(ax)
    for strat in ["greedy","degree","single_leak","random"]:
        s = val[val.strategy == strat]
        ax.plot(s.k, s[metric], color=colmap[strat], ls=stylemap[strat],
                lw=2.2 if strat=="greedy" else 1.7, marker="o" if strat=="greedy" else None,
                ms=3, zorder=5 if strat=="greedy" else 3, label=labelmap[strat])
    ax.axhline(0.5, color=MAG, lw=1.2, ls=(0,(3,2))); ax.text(0.5, 0.52, "低阶阈值 τ=0.5", color=MAG, fontsize=8.5)
    ax.set_xlabel("防护集大小 $|P|$", color=INK, fontsize=10)
    ax.set_ylabel("oracle 估计泄露 $R^2$（max/mean over conf）", color=INK, fontsize=9)
    ax.set_title(ttl, fontsize=10.5, color=INK); ax.set_ylim(0.3,1.02)
    if metric=="worst_conf_r2":
        ax.annotate("覆盖全部危险对后\n真实泄露仍 0.62>τ\n（=70号累积残差）", xy=(24,0.616), xytext=(15.5,0.66),
                    fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
fig.tight_layout(); fig.savefig(f"{D}/figures/protection_validation_zh.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
print("fig2 done")
