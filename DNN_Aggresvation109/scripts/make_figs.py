# -*- coding: utf-8 -*-
"""109 号三张图：V/M 随 n_aux；τ-critical 与档位稳定性；审计方低估攻击者的危险放行。"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
BLUE, ORANGE, AQUA, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#cc3d3d", "#a3a29c", "#2b2b2a"
V = pd.read_csv(A / "109_A_V.csv"); M = pd.read_csv(A / "109_B_M.csv")
C = pd.read_csv(A / "109_C_critical.csv"); R = pd.read_csv(A / "109_D_release.csv")
W = pd.read_csv(A / "109_E_winners_curse.csv")
FR = ["10%", "25%", "50%", "100%"]; x = np.arange(4)

# ---------- 图1：V 与 M 随 n_aux ----------
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.3))
for ds, col in [("pjm", BLUE), ("caiso", ORANGE)]:
    d = V[V.数据集 == ds]
    for s, ls, mk in [(1, ":", "^"), (2, "--", "s"), (3, "-", "o")]:
        e = d[d.规模 == s].set_index("比例").reindex(FR)
        ax[0].plot(x, e.V均值, ls, marker=mk, color=col, ms=5, lw=1.6,
                   label=f"{ds.upper()} |S|={s}" if s == 3 else None)
ax[0].set_xticks(x); ax[0].set_xticklabels(FR); ax[0].set_xlabel("攻击者辅助标签量（占 train 比例）")
ax[0].set_ylabel("平均 V（测试 R²）"); ax[0].grid(axis="x", visible=False)
ax[0].legend(frameon=False, fontsize=8, loc="lower right")
ax[0].text(.02, .96, "实线 |S|=3、虚线 |S|=2、点线 |S|=1", transform=ax[0].transAxes, fontsize=7.8, color=GRAY, va="top")
ax[0].set_title("A. V 随样本量单调上升", loc="left", fontsize=10)
for ds, col in [("pjm", BLUE), ("caiso", ORANGE)]:
    d = M[M.数据集 == ds]
    for K, ls, mk in [(0, ":", "^"), (1, "--", "s"), (2, "-", "o")]:
        e = d[d.K == K].set_index("比例").reindex(FR)
        ax[1].plot(x, e.M均值, ls, marker=mk, color=col, ms=5, lw=1.6, label=f"{ds.upper()} K={K}" if K == 2 else None)
ax[1].set_xticks(x); ax[1].set_xticklabels(FR); ax[1].set_xlabel("攻击者辅助标签量")
ax[1].set_ylabel("平均 M"); ax[1].grid(axis="x", visible=False)
ax[1].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].text(.02, .06, "实线 K=2、虚线 K=1、点线 K=0", transform=ax[1].transAxes, fontsize=7.8, color=GRAY, va="bottom")
ax[1].axhline(0, color=GRAY, lw=.5)
ax[1].set_title("B. ★M^(0) 单调，但 M^(1)/M^(2) 小样本下反超 full", loc="left", fontsize=10)
for ds, col in [("pjm", BLUE), ("caiso", ORANGE)]:
    d = M[M.数据集 == ds]
    for K, ls, mk in [(0, ":", "^"), (2, "-", "o")]:
        e = d[d.K == K].set_index("比例").reindex(FR)
        ax[2].plot(x, e.Kendall对full, ls, marker=mk, color=col, ms=5, lw=1.8, label=f"{ds.upper()} K={K}")
ax[2].set_xticks(x); ax[2].set_xticklabels(FR); ax[2].set_xlabel("攻击者辅助标签量")
ax[2].set_ylabel("字段排序 Kendall（对 full）"); ax[2].set_ylim(0, 1.05); ax[2].grid(axis="x", visible=False)
ax[2].legend(frameon=False, fontsize=8, loc="lower right")
ax[2].set_title("C. 排序稳定性：样本量少时排序仍大体保留", loc="left", fontsize=10)
fig.suptitle("图1  攻击者辅助标签量 n_aux 的影响：V 严格单调上升，但 M^(1)/M^(2) 在小样本下反超 full（机制见图4）",
             x=0.005, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(F / "fig1_naux_v_m.png", bbox_inches="tight"); plt.close(fig)

# ---------- 图2：τ-critical ----------
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.4))
for k, ds in enumerate(["pjm", "caiso"]):
    d = C[(C.数据集 == ds) & (C.K == 2)]
    for tau, col, mk in [(0.5, AQUA, "^"), (0.7, BLUE, "o"), (0.9, ORANGE, "s")]:
        e = d[d.tau == tau].set_index("比例").reindex(FR)
        ax[k].plot(x, e.recall, "-", marker=mk, color=col, ms=6, lw=2, label=f"τ={tau} recall")
        ax[k].plot(x, e.precision, "--", marker=mk, color=col, ms=4, lw=1.2, alpha=.65, label=f"τ={tau} precision")
    ax[k].set_xticks(x); ax[k].set_xticklabels(FR); ax[k].set_ylim(0, 1.05)
    ax[k].set_xlabel("审计方假设的攻击者辅助标签量"); ax[k].grid(axis="x", visible=False)
    ax[k].set_title(f"{ds.upper()}（K=2，以 full 攻击者为真值）", loc="left", fontsize=10)
ax[0].set_ylabel("比例"); ax[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
fig.suptitle("图2  用弱攻击者（小 n_aux）做审计会系统性漏判 τ-critical 字段：recall 随假设的攻击者能力下降而下降",
             x=0.005, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(F / "fig2_naux_critical.png", bbox_inches="tight"); plt.close(fig)

# ---------- 图3：危险放行 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.4))
w = .26
for k, ds in enumerate(["pjm", "caiso"]):
    d = R[R.数据集 == ds]
    for j, (tau, col) in enumerate([(0.5, AQUA), (0.7, BLUE), (0.9, ORANGE)]):
        e = d[d.tau == tau].set_index("比例").reindex(FR)
        ax[k].bar(x + (j - 1) * w, e.危险放行率, w * .9, color=col, label=f"τ={tau}")
        for xi, v in zip(x + (j - 1) * w, e.危险放行率):
            if v > 0.004: ax[k].text(xi, v + .006, f"{v:.1%}", ha="center", fontsize=7.2)
    ax[k].set_xticks(x); ax[k].set_xticklabels(FR); ax[k].grid(axis="x", visible=False)
    ax[k].set_xlabel("审计方假设的攻击者辅助标签量")
    ax[k].set_title(f"{ds.upper()}：全部规模 3 集合 × 12 目标", loc="left", fontsize=10)
ax[0].set_ylabel("危险放行率  P(审计放行 | full 攻击者可推断)")
ax[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("图3  审计方低估攻击者能力的代价：按 10% 辅助数据评估并放行时，真实（full）攻击者仍能推断的集合被放行的比例",
             x=0.005, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(F / "fig3_naux_release.png", bbox_inches="tight"); plt.close(fig)
print("3 张图已写入", F)

# ---------- 图4：winner's curse 分解 ----------
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.3))
for ds, col in [("pjm", BLUE), ("caiso", ORANGE)]:
    d = W[(W.数据集 == ds) & (W.比例 != "100%")]
    for K, mk, ls in [(0, "^", ":"), (1, "s", "--"), (2, "o", "-")]:
        e = d[d.K == K].set_index("比例").reindex(FR[:3])
        ax[0].plot(np.arange(3), e.选择增益, ls, marker=mk, color=col, ms=6, lw=1.8,
                   label=f"{ds.upper()} K={K}（{int(W[(W.数据集==ds)&(W.K==K)].候选背景数.iloc[0])} 个候选背景）")
ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels(FR[:3]); ax[0].set_xlabel("攻击者辅助标签量")
ax[0].set_ylabel("选择增益  E[M_frac − Δ_frac(T_full)]"); ax[0].grid(axis="x", visible=False)
ax[0].legend(frameon=False, fontsize=7.2, loc="upper right")
ax[0].set_title("A. 选择增益随候选背景数增长（K=0 恒为 0）", loc="left", fontsize=10)
xx = np.arange(3); ww = .36
for k, ds in enumerate(["pjm", "caiso"]):
    d = W[(W.数据集 == ds) & (W.K == 2) & (W.比例 != "100%")].set_index("比例").reindex(FR[:3])
    ax[1].bar(xx + (k - .5) * ww, d.选择增益, ww * .42, color=RED, alpha=.85 if k == 0 else .5,
              label=f"{ds.upper()} 选择增益")
    ax[1].bar(xx + (k - .5) * ww, -d.真实低估, ww * .42, color=BLUE if k == 0 else ORANGE,
              label=f"{ds.upper()} 真实低估")
    ax[1].plot(xx + (k - .5) * ww, d.总偏差, "o-" if k == 0 else "s--", color=INK, ms=4, lw=1.2,
               label=f"{ds.upper()} 净偏差")
ax[1].axhline(0, color=INK, lw=.8)
ax[1].set_xticks(xx); ax[1].set_xticklabels(FR[:3]); ax[1].set_xlabel("攻击者辅助标签量")
ax[1].set_ylabel("对 M^(2) 的偏差贡献"); ax[1].grid(axis="x", visible=False)
ax[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
ax[1].set_title("B. K=2 分解：选择增益 > 真实低估 => 净正偏", loc="left", fontsize=10)
for ds, col in [("pjm", BLUE), ("caiso", ORANGE)]:
    d = W[(W.数据集 == ds) & (W.K == 2)].set_index("比例").reindex(FR)
    ax[2].plot(x, d.见证一致率, "o-", color=col, ms=6, lw=2, label=f"{ds.upper()} 见证与 full 一致")
    ax[2].plot(x, d.frac见证在full上的边际比, "s--", color=col, ms=5, lw=1.4, alpha=.7,
               label=f"{ds.upper()} 该见证在 full 上的效力")
ax[2].set_xticks(x); ax[2].set_xticklabels(FR); ax[2].set_xlabel("攻击者辅助标签量")
ax[2].set_ylabel("比例"); ax[2].set_ylim(0, 1.05); ax[2].grid(axis="x", visible=False)
ax[2].legend(frameon=False, fontsize=7.2, loc="upper left")
ax[2].set_title("C. 小样本挑出的见证背景大多是噪声", loc="left", fontsize=10)
fig.suptitle("图4  M^(K) 是最大值统计量：小样本下取 max 会系统性挑中正噪声（winner's curse），"
             "候选背景数越多（K 越大）正偏越强 —— 这是 108 号「认证下界 + 区间」口径的第二个独立理由",
             x=0.005, ha="left", fontsize=10.8)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(F / "fig4_winners_curse.png", bbox_inches="tight"); plt.close(fig)
print("图4 已写入")
