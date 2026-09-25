# -*- coding: utf-8 -*-
"""122 号：目录内中文图。"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); A = os.path.join(ROOT, "outputs", "analysis")
FG = os.path.join(ROOT, "figures"); os.makedirs(FG, exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 170, "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
DS = ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen/ic", "CAISO-load"]
COL = {"RTS-GMLC": "#2a78d6", "NEM": "#eb6834", "PJM-load": "#1baf7a", "PJM-gen/ic": "#008300", "CAISO-load": "#4a3aa7"}
VC = {"random": "#b4b3ad", "uniform": "#eb6834", "small": "#f3a47f", "recon": "#2a78d6", "small_recon": "#9fc3ea",
      "uniform+random": "#d9a38a", "recon+random": "#0d366b", "uniformE": "#a3401a", "reconE": "#0b2a55",
      "random_cy": "#8a8984", "uniform_cy": "#c65a2a", "uniformE_cy": "#7a2e12", "recon_cy": "#3987e5", "reconE_cy": "#1baf7a", "recon+random_cy": "#008300"}
NAME = {"random": "随机初始化", "uniform": "只预测目标\n(现行)", "small": "小集合掩码", "recon": "目标+重建", "small_recon": "小集合+重建",
        "uniform+random": "现行+随机\n特征拼接", "recon+random": "重建+随机\n特征拼接", "uniformE": "现行×3种子\n(论文口径E)", "reconE": "重建×3种子",
        "random_cy": "随机\n+截断", "uniform_cy": "只预测目标\n+截断", "uniformE_cy": "E+截断", "recon_cy": "重建\n+截断",
        "reconE_cy": "重建×3种子\n+截断", "recon+random_cy": "重建+随机拼接\n+截断(推荐,3种子均值)"}
# 推荐配置 recon+random_cy 一律取三个种子（0/1/2）的平均：把种子 1、2 的行并入同一变体名
SEEDMAP = {"recon+random@1_cy": "recon+random_cy", "recon+random@2_cy": "recon+random_cy"}
R = pd.read_csv(os.path.join(A, "variants_by_target.csv")).replace({"变体": SEEDMAP})
Dc = pd.read_csv(os.path.join(A, "error_decomposition.csv")).replace({"变体": SEEDMAP})
VARS = [v for v in NAME if v in set(R.变体)]
BASE7 = ["random", "uniform", "small", "recon", "small_recon", "uniform+random", "recon+random"]


def grid(ax):
    ax.grid(color="#e6e5e0", axis="y"); ax.set_axisbelow(True)


def bars(ax, col, title, src=R, vars_=None, fmt="{:.3f}"):
    vars_ = vars_ or VARS; x = np.arange(len(vars_))
    m = src.groupby("变体")[col].mean().reindex(vars_)
    ax.bar(x, m.values, color=[VC[v] for v in vars_], width=.7, edgecolor="white")
    for k, v in enumerate(vars_):
        g = src[src.变体 == v].groupby("数据")[col].mean()
        for ds, val in g.items():
            ax.plot(k + (DS.index(ds) - 2) * .09, val, "o", ms=3.2, color=COL[ds], mec="white", mew=.4)
        ax.text(k, m[v], fmt.format(m[v]), ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels([NAME[v] for v in vars_], fontsize=7, rotation=0); ax.set_title(title, loc="left", fontsize=9.5); grid(ax)


# 图 1：表征诊断与集合值误差
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
base = ["random", "uniform", "small", "recon", "small_recon"]
bars(axs[0], "有效维度", "A. 表征的有效维度（参与比，越大越不塌缩）", vars_=base, fmt="{:.2f}")
bars(axs[1], "V误差_3", "B. 集合值 V 误差（规模 ≤3，闭包后）", vars_=BASE7)
bars(axs[2], "V误差_4", "C. 集合值 V 误差（规模 4，抽样 ≤1,500）", vars_=BASE7)
for ds in DS:
    axs[0].plot([], [], "o", color=COL[ds], label=ds, ms=4)
axs[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.suptitle("图1  只预测目标的掩码预训练使表征塌缩（有效维度≈1.6，随机初始化≈3.3），集合值精度与随机特征相当；加入被遮列重建后有效维度恢复、V 误差降约 25%",
             x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig1_collapse_and_V.png"), bbox_inches="tight"); plt.close(fig)

# 图 2：秩截断
K = pd.read_csv(os.path.join(A, "rank_truncation.csv")); K["秩"] = K["秩"].astype(str)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharey=False)
ranks = ["0", "2", "4", "8", "32", "全部"]; xr = {r: k for k, r in enumerate(ranks)}
for ax, ds in zip(axs, ["RTS-GMLC", "NEM", "PJM-gen/ic", "CAISO-load"]):
    g = K[K.数据 == ds].groupby(["主干", "秩"]).V误差.mean()
    for bb, lab in [("random", "随机初始化"), ("uniform", "只预测目标"), ("recon", "目标+重建")]:
        rr = [r for r in ranks if (bb, r) in g.index]
        ax.plot([xr[r] for r in rr], [g[(bb, r)] for r in rr], "-o", color=VC[bb], lw=2, ms=6, label=lab, mec="white")
    ax.set_xticks(range(len(ranks))); ax.set_xticklabels(["0\n(无φ)", "2", "4", "8", "32", "256\n(全部)"])
    ax.set_title(ds, loc="left", fontsize=10); ax.set_xlabel("保留的主干特征主成分数 r"); grid(ax)
axs[0].set_ylabel("V 误差（200 个规模 ≤3 集合，未闭包）"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("图2  秩截断：预训练的前几个主方向与目标对齐（r=2 时误差约为随机特征的一半），但读出的精度主要来自高维的其余方向——那里随机 ReLU 特征与只预测目标的主干一样好；重建的收益也在其余方向",
             x=.01, ha="left", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, .9]); fig.savefig(os.path.join(FG, "fig2_rank_truncation.png"), bbox_inches="tight"); plt.close(fig)

# 图 3：集合规模
L = pd.read_csv(os.path.join(A, "large_sets.csv"))
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axs[0]; sz = ["≤3", "4", "6", "10", "16"]
for bb, lab in [("random", "随机初始化"), ("uniform", "只预测目标（现行）"), ("recon", "目标+重建"), ("small_recon", "小集合+重建")]:
    y = [R[R.变体 == bb].V误差_3.mean(), R[R.变体 == bb].V误差_4.mean()] + [L[(L.变体 == bb) & (L.规模 == s)].V误差.mean() for s in (6, 10, 16)]
    ax.plot(range(5), y, "-o", color=VC[bb], lw=2, ms=6, label=lab, mec="white")
ax.axvspan(1.5, 4.4, color="#f1f0eb", zorder=0); ax.text(1.6, ax.get_ylim()[1] * .95 if ax.get_ylim()[1] else .1, "规模 6/10/16：梯度提升树真值（口径不同，只比主干间相对差）", fontsize=7.5, color="#52514e", va="top")
ax.set_xticks(range(5)); ax.set_xticklabels(sz); ax.set_xlabel("集合规模 |S|"); ax.set_ylabel("V 误差（五组平均）"); ax.legend(frameon=False, fontsize=8, loc="upper left"); grid(ax)
ax.set_title("A. 预训练的价值随集合规模增大：小集合上随机特征够用，大集合上随机特征失效", loc="left", fontsize=9.5)
ax = axs[1]; w = .2
for k, bb in enumerate(["random", "uniform", "recon"]):
    v = [L[(L.变体 == bb) & (L.规模 == 16) & (L.数据 == ds)].V误差.mean() for ds in DS]
    ax.bar(np.arange(5) + (k - 1) * w, v, w, color=VC[bb], label=NAME[bb].replace("\n", ""), edgecolor="white")
ax.set_xticks(range(5)); ax.set_xticklabels(DS, fontsize=8); ax.legend(frameon=False, fontsize=8); grid(ax)
ax.set_title("B. 规模 16 的集合：各数据组 V 误差", loc="left", fontsize=9.5)
fig.suptitle("图3  随机特征 ≈ 预训练只在审计用到的小集合上成立：规模 16 时重建主干的误差约为随机特征的 1/3", x=.01, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig3_set_size.png"), bbox_inches="tight"); plt.close(fig)

# 图 4：V 与 M 的分离——M 取最大，误差由边际误差的上尾决定
fig, axs = plt.subplots(1, 4, figsize=(19, 4.4))
V4 = ["random", "uniform", "recon", "uniformE", "reconE", "random_cy", "uniform_cy", "reconE_cy"]
bars(axs[0], "集合误差", "A. 集合值误差 |估计V − 真V|", src=Dc, vars_=V4)
bars(axs[1], "边际误差", "B. 边际误差 |估计Δ − 真Δ|（均值）", src=Dc, vars_=V4)
bars(axs[2], "字段最大高估", "C. 边际误差上尾：每字段 max_T(估计Δ − 真Δ)", src=Dc, vars_=V4)
bars(axs[3], "M2误差", "D. M^(2) 误差", src=R, vars_=V4)
fig.suptitle("图4  M 是对背景取最大，误差由边际误差的上尾决定：未截断时重建主干的尖峰特征让少数读出爆掉、上尾变长；加预测截断后上尾被切掉，"
             "只预测目标的主干不再有任何优势",
             x=.01, ha="left", fontsize=10.5)
for a_ in axs:
    a_.tick_params(axis="x", labelsize=6.3)
for ds in DS:
    axs[3].plot([], [], "o", color=COL[ds], label=ds, ms=4)
axs[3].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(rect=[0, 0, 1, .9]); fig.savefig(os.path.join(FG, "fig4_V_vs_M_tail.png"), bbox_inches="tight"); plt.close(fig)

# 图 5：PJM-gen/ic 的读出塌缩与修复
fig, axs = plt.subplots(1, 2, figsize=(15, 4.4)); V5 = ["uniformE", "reconE", "uniformE_cy", "reconE_cy", "recon+random_cy"]
for ax, col, ttl in [(axs[0], "M2误差", "A. M^(2) 误差（按数据组）"), (axs[1], "认证前1", "B. 认证前 1 下界比（按数据组）")]:
    P = R.pivot_table(index="数据", columns="变体", values=col).reindex(DS)[V5]; w = .8 / len(V5)
    for k, v in enumerate(V5):
        ax.bar(np.arange(5) + (k - len(V5) / 2 + .5) * w, P[v].values, w, color=VC[v], label=NAME[v].replace("\n", ""), edgecolor="white")
    ax.set_xticks(range(5)); ax.set_xticklabels(DS, fontsize=8); grid(ax); ax.set_title(ttl, loc="left", fontsize=9.5)
axs[1].set_ylim(.5, 1.02); axs[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="center left")
axs[0].annotate("单字段 {da_as_total_mw_*} 读出爆掉\n（估计V=0，真值≈0.3）→ 以其为背景的\n价格字段 M 被高估", xy=(2.95, .074), xytext=(0.9, .062), fontsize=7.5,
                color="#52514e", arrowprops=dict(arrowstyle="->", color="#8a8984"))
fig.suptitle("图5  重建主干唯一的失败（PJM 总发电/净交换）来自尖峰特征造成的读出爆掉；把预测截断到训练目标取值范围（不看测试标签）即修复",
             x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig5_collapse_fix.png"), bbox_inches="tight"); plt.close(fig)

# 图 6：最终对比——论文口径 E 与改进方案（相对 E）
Dd = pd.read_csv(os.path.join(A, "decisions.csv")).replace({"估计器": SEEDMAP})
met = [("V误差_3", "V 误差 ≤3", -1), ("V误差_4", "V 误差 规模4", -1), ("M2误差", "M^(2) 误差", -1), ("Γ2误差", "Γ^(2) 误差", -1),
       ("认证前1", "认证前1比", 1), ("背景召回前3", "背景召回前3", 1)]
V6 = ["uniformE", "random_cy", "reconE_cy", "recon+random_cy"]
fig, axs = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]; w = .8 / len(V6); base_ = R[R.变体 == "uniformE"]
for k, v in enumerate(V6):
    g = R[R.变体 == v]; vals = [g[c].mean() / base_[c].mean() for c, _, _ in met]
    ax.bar(np.arange(len(met)) + (k - len(V6) / 2 + .5) * w, vals, w, color=VC[v], label=NAME[v].replace("\n", ""), edgecolor="white")
    for j, (c, _, _) in enumerate(met):
        ax.text(j + (k - len(V6) / 2 + .5) * w, vals[j], f"{g[c].mean():.3f}", ha="center", va="bottom", fontsize=6.2, rotation=90)
ax.axhline(1, color="#52514e", lw=.8); ax.set_xticks(range(len(met))); ax.set_xticklabels([f"{l}\n({'越低越好' if s_ < 0 else '越高越好'})" for _, l, s_ in met], fontsize=8)
ax.set_ylabel("相对论文口径 E"); ax.set_ylim(.55, 1.3); ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left"); grid(ax)
ax.set_title("A. 估计层面（柱顶数字为绝对值）", loc="left", fontsize=9.5)
ax = axs[1]; x = np.arange(4); lab = []
for k, v in enumerate(V6):
    vals = []
    for tau in (0.5, 0.7):
        for dl in (0.0, 0.05):
            vals.append(Dd[(Dd.估计器 == v) & (Dd.规则 == "扫描—认证标记") & (Dd.τ == tau) & (Dd.δ == dl)].召回.mean())
    ax.bar(x + (k - len(V6) / 2 + .5) * w, vals, w, color=VC[v], edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(["τ=0.5\nδ=0", "τ=0.5\nδ=0.05", "τ=0.7\nδ=0", "τ=0.7\nδ=0.05"]); ax.set_ylim(.6, 1.0); grid(ax)
ax.set_title("B. 决策层面：扫描—认证的关键字段召回（精确率均为 1.00）", loc="left", fontsize=9.5)
fig.suptitle("图6  改进方案：重建预训练 + 预测截断在全部估计指标上优于论文口径 E；与随机特征拼接（单主干）在 M、Γ、背景召回上最好",
             x=.01, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig6_final_comparison.png"), bbox_inches="tight"); plt.close(fig)
print("图已生成")
