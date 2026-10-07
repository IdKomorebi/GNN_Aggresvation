# -*- coding: utf-8 -*-
"""作图：图1 E1 代价对比；图2 E2/E2b 残余与验证量；图3 E3 GRPO 对比与训练曲线。"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); OUT = os.path.join(HERE, "outputs"); FIG = os.path.join(HERE, "figures")
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; font_manager.fontManager.addfont(fp)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"; COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NAME = {"all_raise": "全部上调", "greedy_M": "M 贪心", "greedy_M_prune": "M 贪心+回退", "greedy_cover": "覆盖贪心", "greedy_cover_prune": "覆盖贪心+回退", "optimal": "精确最优",
        "grpo_raw": "GRPO", "grpo_prune": "GRPO+回退", "grpo_x16": "GRPO 采样16次", "noisy_greedy_x16": "扰动贪心 采样16次"}


def style(ax, title):
    ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=11.5, color=INK); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, labelsize=9, length=0)


def fig1():
    d = pd.read_csv(os.path.join(OUT, "e1_instances.csv")); d = d[d.lock == 0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.7, 1]))
    g = d[d.方法.isin(["all_raise", "greedy_cover_prune", "optimal"])].groupby(["组", "τ", "方法"]).代价.mean().unstack()
    x = np.arange(len(g)); w = 0.26
    for k, m in enumerate(["all_raise", "greedy_cover_prune", "optimal"]):
        ax[0].bar(x + (k - 1) * w, g[m], w - 0.03, color=COL[k], label=NAME[m])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{a}\nτ={b}" for a, b in g.index], fontsize=8.5)
    style(ax[0], "(a) 平均上调代价（级次 × 代价权重），每组 160 个随机实例"); ax[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    r = d.groupby("方法").与最优之比.mean().drop("all_raise").sort_values(ascending=False)
    ax[1].barh([NAME[m] for m in r.index], r.values - 1, color=COL[0], height=0.6)
    for y, v in enumerate(r.values):
        ax[1].text(v - 1 + 0.002, y, f"+{100*(v-1):.1f}%", va="center", fontsize=9, color=INK2)
    style(ax[1], "(b) 相对精确最优多花的代价"); ax[1].grid(axis="x", color=GRID, lw=0.8); ax[1].grid(axis="y", visible=False)
    ax[1].set_xlim(0, (r.values.max() - 1) * 1.25); ax[1].set_xticklabels([f"{100*t:.0f}%" for t in ax[1].get_xticks()])
    fig.suptitle("图1  E1：上调最少的一致等级 vs 全部上调（“全部上调”平均为最优的 3.8 倍，未画入右图）", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(FIG, "图1_E1_代价对比.png"), dpi=160, facecolor=SURF); plt.close(fig)


def fig2():
    d = pd.read_csv(os.path.join(OUT, "e2_instances.csv")); d = d[d.求解 == "optimal"]; b = pd.read_csv(os.path.join(OUT, "e2b_instances.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4), facecolor=SURF)
    for j, (col, ttl) in enumerate((("有残余", "(a) 按真值仍有越线组合未消除的实例比例"), ("与真值最优之比", "(b) 代价与真值最优之比"))):
        g = d.groupby(["δ", "口径"])[col].mean().unstack(); x = np.arange(len(g)); w = 0.36
        for k, m in enumerate(["只用估计值", "扫描—认证"]):
            ax[j].bar(x + (k - 0.5) * w, g[m], w - 0.04, color=COL[k], label=m)
            for xi, v in zip(x, g[m]):
                ax[j].text(xi + (k - 0.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
        ax[j].set_xticks(x); ax[j].set_xticklabels([f"余量 δ={v}" for v in g.index]); style(ax[j], ttl); ax[j].legend(frameon=False, fontsize=9)
    ax[1].axhline(1, color=INK2, lw=0.8, ls="--")
    g = b.groupby(["组", "先验"]).验证集合比例.mean().unstack()[["不用估计值", "估计值+0", "估计值+0.05"]]; x = np.arange(len(g)); w = 0.26
    for k, m in enumerate(g.columns):
        ax[2].bar(x + (k - 1) * w, g[m], w - 0.03, color=[COL[3], COL[0], COL[2]][k], label={"不用估计值": "不用估计值", "估计值+0": "估计值先验 δ=0", "估计值+0.05": "估计值先验 δ=0.05"}[m])
    ax[2].set_xticks(x); ax[2].set_xticklabels(g.index, fontsize=8.5); style(ax[2], "(c) 验证闭环（零残余、达到真值最优）需重训的集合比例"); ax[2].set_ylim(0, 1.18); ax[2].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center")
    fig.suptitle("图2  E2/E2b：用估计值代替真值——一次性做法有残余；闭环验证可保证零残余，但要重训相当比例的集合", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(FIG, "图2_E2_残余与验证量.png"), dpi=160, facecolor=SURF); plt.close(fig)


def fig3():
    d = pd.read_csv(os.path.join(OUT, "e3_instances.csv")); d["方法族"] = d.方法.str.replace(r"_s[01]", "", regex=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.5, 1]))
    order = ["greedy_M_prune", "greedy_cover_prune", "grpo_raw", "grpo_prune", "noisy_greedy_x16", "grpo_x16"]
    g = d[d.方法族.isin(order)].groupby(["方法族", "训练时见过"]).与最优之比.mean().unstack().reindex(order) - 1; x = np.arange(len(order)); w = 0.36
    for k, (c, lab) in enumerate(((True, "训练时见过的组"), (False, "没见过的组（NEM、CAISO）"))):
        ax[0].bar(x + (k - 0.5) * w, g[c], w - 0.04, color=COL[k], label=lab)
        for xi, v in zip(x, g[c]):
            ax[0].text(xi + (k - 0.5) * w, v + 0.0008, f"{100*v:.1f}%", ha="center", fontsize=8.5, color=INK2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([NAME[o] for o in order], fontsize=9); style(ax[0], "(a) 留出实例上相对精确最优多花的代价（两个种子平均）"); ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_yticklabels([f"{100*t:.0f}%" for t in ax[0].get_yticks()])
    for k, s in enumerate((0, 1)):
        f = os.path.join(OUT, "grpo", f"trainlog_seed{s}.json")
        if os.path.exists(f):
            L = json.load(open(f)); it = [r["iter"] for r in L["log"]]
            ax[1].plot(it, [r["val_prune"] for r in L["log"]], color=COL[k], lw=2, label=f"GRPO+回退 种子{s}"); ax[1].plot(it, [r["val_raw"] for r in L["log"]], color=COL[k], lw=1.2, ls=":", label=f"GRPO 种子{s}")
            ax[1].axhline(L["val_greedy"], color=INK2, lw=1, ls="--")
    ax[1].text(ax[1].get_xlim()[1], L["val_greedy"], " 覆盖贪心+回退", va="bottom", ha="right", fontsize=8.5, color=INK2)
    ax[1].set_ylim(1.0, 1.12); style(ax[1], "(b) 训练过程：验证实例上与最优之比"); ax[1].legend(frameon=False, fontsize=8.5); ax[1].set_xlabel("迭代", fontsize=9, color=INK2)
    fig.suptitle("图3  E3：改造后的 GRPO 与贪心的对比", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(FIG, "图3_E3_GRPO对比.png"), dpi=160, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    if os.path.exists(os.path.join(OUT, "e3_instances.csv")):
        fig3()
    print("ok")
