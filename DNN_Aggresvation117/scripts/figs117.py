# -*- coding: utf-8 -*-
"""117 号 步骤 4：目录内的中文图（论文英文图见 paper/claude/V4_en）。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = os.path.join(ROOT, "outputs", "analysis"); FG = os.path.join(ROOT, "figures"); O = os.path.join(ROOT, "outputs")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
INK, GRID, BLUE, ORANGE, AQUA = "#0b0b0b", "#e6e5e0", "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
NM = {"pearson": "Pearson r²", "spearman": "Spearman ρ²", "mi": "互信息", "dcor": "距离相关", "M0": "单字段 V({i})", "loco": "LOCO",
      "perm": "置换重要性", "sage": "SAGE", "graph": "推断图传播", "M2_est": "M^(2) 估计（通用模型）", "M2_cert": "M^(2) 扫描—认证", "M2": "M^(2) 精确"}

# 图1 机理小例子
T = pd.read_csv(os.path.join(O, "toy_scores.csv")); cols = ["pearson", "mi", "M0", "loco", "perm", "sage", "graph", "M2"]
M = T[cols].values; Mn = np.clip(M, 0, None) / np.clip(M, 0, None).max(0, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.imshow(Mn, cmap=SEQ, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = 0.0 if abs(M[i, j]) < .005 else M[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if Mn[i, j] > .6 else INK)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([NM[c] for c in cols]); ax.xaxis.tick_top()
lab = {"A": "A（协同）", "B": "B（协同）", "C1": "C1（冗余）", "C2": "C2（冗余）", "D": "D（间接）", "N": "N（噪声）"}
ax.set_yticks(range(len(T))); ax.set_yticklabels([lab[f] for f in T.字段])
ax.set_title("图1  机理小例子：Y = a·A·B + b·P + 噪声。相关与单字段看不见协同（A、B），LOCO/置换/SAGE 看不见冗余与间接（C、D），M^(2) 都能看见\n"
             "（格内为原始分数，颜色按列归一化；真值：A+B 0.54，C1 0.39，A+B+C1 0.94，A+B+D 0.73）", fontsize=9.5, loc="left", pad=28)
fig.tight_layout(); fig.savefig(os.path.join(FG, "fig1_toy_mechanism.png"), bbox_inches="tight"); plt.close(fig)

# 图2 决策与扣留
rules = pd.read_csv(os.path.join(A, "decision_rules.csv")); adapt = pd.read_csv(os.path.join(A, "withholding_strategies.csv"))
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
items = [("单字段规则 V({i})>τ", None, ORANGE), ("扫描标记（估计）", 0.05, "#9fc3ea"), ("扫描—认证标记", 0.0, BLUE), ("扫描—认证标记", 0.05, "#0d366b")]
x = np.arange(2); w = .2
for k, (r, dl, c) in enumerate(items):
    d = rules[(rules.规则 == r) & (rules.δ.isna() if dl is None else rules.δ == dl)]; m = d.groupby("τ").召回.mean().reindex([.5, .7])
    axs[0].bar(x + (k - 1.5) * w, m.values, w * .92, color=c, label=r + ("" if dl is None else f"（δ={dl}）"))
    for xi, v in zip(x + (k - 1.5) * w, m.values):
        axs[0].text(xi, v + .015, f"{v:.2f}", ha="center", fontsize=8)
axs[0].set_xticks(x); axs[0].set_xticklabels(["τ=0.5", "τ=0.7"]); axs[0].set_ylim(0, 1.1); axs[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2)
axs[0].set_ylabel("关键字段召回率（精确率均≈1）"); axs[0].set_title("A. 发现关键字段：单字段规则只召回 13–17%", loc="left", fontsize=10)
strat = [("单字段定级（扣留单字段越阈者）", None, ORANGE), ("估计MUS最小命中集", .05, AQUA), ("自适应M̂2贪心", .05, BLUE)]
for k, (lab_, dl, c) in enumerate(strat):
    d = adapt[(adapt.策略 == lab_) & (adapt.δ.isna() if dl is None else adapt.δ == dl)]
    g = d.groupby("τ").apply(lambda e: pd.Series(dict(残余=e.残余真危险组合.sum() / e.危险组合数.sum(), 扣留=e.扣留数.mean(), 最优=e.最优扣留数.mean())))
    axs[1].bar(x + (k - 1) * .26, g.残余.values, .24, color=c, label=lab_.replace("M̂2", " M^(2)（估计值）"))
    for xi, (_, r) in zip(x + (k - 1) * .26, g.iterrows()):
        axs[1].text(xi, r.残余 + .015, f"{r.残余:.1%}\n扣{r.扣留:.1f}/最优{r.最优:.1f}", ha="center", fontsize=7.5)
axs[1].set_xticks(x); axs[1].set_xticklabels(["τ=0.5", "τ=0.7"]); axs[1].set_ylim(0, 1.15); axs[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3)
axs[1].set_ylabel("扣留后仍未打破的真实危险组合比例"); axs[1].set_title("B. 处置：不用真值选择的两种策略打破 96% 以上的危险组合", loc="left", fontsize=10)
for ax in axs:
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.suptitle("图2  决策层面的对比（10 个目标平均；选择只用估计值，真值只用于评价）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig2_decision_disposal.png"), bbox_inches="tight"); plt.close(fig)

# 图3 排序与静态扣留
det = pd.read_csv(os.path.join(A, "detection.csv")); prot = pd.read_csv(os.path.join(A, "protection.csv"))
order = list(NM); fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
rp = det.groupby(["τ", "方法"]).R精确率.mean().unstack(0).reindex(order); xx = np.arange(len(order))
axs[0].bar(xx - .2, rp[.5], .38, color="#cfcec8", label="τ=0.5"); axs[0].bar(xx + .2, rp[.7], .38, color=BLUE, label="τ=0.7")
axs[0].set_ylim(.5, 1.02); axs[0].set_xticks(xx); axs[0].set_xticklabels([NM[m] for m in order], rotation=35, ha="right", fontsize=8)
axs[0].legend(frameon=False); axs[0].set_title("A. 排序：R-precision（按真实关键数给阈值，对基线最宽松）；τ=0.5 时几乎全是关键字段，指标饱和", loc="left", fontsize=9.5)
ex = prot.groupby(["τ", "方法"]).超出最优.mean().unstack(0).reindex(order)
axs[1].bar(xx - .2, ex[.5], .38, color="#cfcec8", label="τ=0.5"); axs[1].bar(xx + .2, ex[.7], .38, color=BLUE, label="τ=0.7")
axs[1].set_xticks(xx); axs[1].set_xticklabels([NM[m] for m in order], rotation=35, ha="right", fontsize=8); axs[1].legend(frameon=False)
axs[1].set_ylabel("按分数依次扣留，比最优多扣的字段数"); axs[1].set_title("B. 静态扣留：LOCO/置换/SAGE 最差（冗余被稀释）", loc="left", fontsize=10)
for ax in axs:
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.suptitle("图3  排序层面的对比（10 个目标平均）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(os.path.join(FG, "fig3_ranking.png"), bbox_inches="tight"); plt.close(fig)
print("ok")
