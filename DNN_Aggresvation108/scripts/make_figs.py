# -*- coding: utf-8 -*-
"""108 号三张图：A1 区间口径 / A1 区间下的 critical / A3 K 递进口径 + A2 元训练定位。"""
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
B = pd.read_csv(A / "108_A1_bounds.csv"); C = pd.read_csv(A / "108_A1_coverage.csv")
K = pd.read_csv(A / "108_A1_critical_interval.csv"); E = pd.read_csv(A / "108_A3_escalation.csv")
S = pd.read_csv(A / "108_A3_saturation.csv"); MT = pd.read_csv(A / "108_A2_metatrain.csv")

# ---------- 图1：A1 下界抬升 + 上界覆盖 ----------
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.3))
x = np.arange(3); w = .36
for k, (ds, col) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = B[B.数据集 == ds].set_index("K")
    ax[0].plot(x, d.认证下界比_top1, "o--", color=col, ms=6, lw=1.4, label=f"{ds.upper()} top-1")
    ax[0].plot(x, d.认证下界比_top3, "o-", color=col, ms=7, lw=2.2, label=f"{ds.upper()} top-3")
    for xi, a_, b_ in zip(x, d.认证下界比_top1, d.认证下界比_top3):
        if b_ > a_ + 1e-6:
            ax[0].annotate("", (xi, b_), (xi, a_), arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
            ax[0].text(xi + .06, (a_ + b_) / 2, f"+{b_-a_:.3f}", fontsize=8, color=col)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"K={i}" for i in x]); ax[0].set_ylim(.78, 1.02)
ax[0].set_ylabel("认证下界比  ΣL / ΣM"); ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")
ax[0].grid(axis="x", visible=False)
ax[0].set_title("A. top-3 见证认证抬高下界（零额外算力）", loc="left", fontsize=10)
# 见证召回
for k, (ds, col) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    d = B[B.数据集 == ds].set_index("K")
    ax[1].bar(x + (k - .5) * w - .08, d.召回_L1大于M减002, w * .44, color=col, alpha=.45,
              label=f"{ds.upper()} top-1" if k == 0 else f"{ds.upper()} top-1")
    ax[1].bar(x + (k - .5) * w + .08, d.召回_L3大于M减002, w * .44, color=col,
              label=f"{ds.upper()} top-3")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"K={i}" for i in x]); ax[1].set_ylim(0, 1.12)
ax[1].set_ylabel("见证召回  P(L ≥ M − 0.02)"); ax[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
ax[1].grid(axis="x", visible=False)
ax[1].set_title("B. 见证召回：PJM K=2 从 0.35 抬到 0.59", loc="left", fontsize=10)
# 上界覆盖率
for k, (ds, col) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    for al, ls, mk in [(0.05, "--", "o"), (0.01, "-", "s")]:
        d = C[(C.数据集 == ds) & (C.alpha == al)].set_index("K")
        ax[2].plot(x, d.覆盖率_区间含M, ls, marker=mk, color=col, ms=6, lw=1.8,
                   label=f"{ds.upper()} α={al}")
ax[2].axhline(.95, color=AQUA, lw=1, ls=":"); ax[2].text(0.02, .955, "95%", fontsize=8, color=AQUA)
ax[2].axhline(.99, color=GRAY, lw=1, ls=":"); ax[2].text(0.02, .993, "99%", fontsize=8, color=GRAY)
ax[2].set_xticks(x); ax[2].set_xticklabels([f"K={i}" for i in x]); ax[2].set_ylim(.88, 1.005)
ax[2].set_ylabel("区间覆盖率  P(L ≤ M ≤ U)"); ax[2].legend(frameon=False, fontsize=7.5, loc="lower left")
ax[2].grid(axis="x", visible=False)
ax[2].set_title("C. 区间覆盖率：α=0.01 时 96–99%", loc="left", fontsize=10)
fig.suptitle("图1  RQ-A4 改用 [L, U] 区间口径：top-3 见证认证抬高下界，留一字段校准的上界给出 96–99% 覆盖",
             x=0.005, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(F / "fig1_interval_bounds.png", bbox_inches="tight"); plt.close(fig)

# ---------- 图2：区间口径下的 τ-critical ----------
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.4))
taus = [0.5, 0.7, 0.9]
for k, ds in enumerate(["pjm", "caiso"]):
    d = K[(K.数据集 == ds) & (K.alpha == 0.05) & (K.K == 2)].set_index("tau")
    xx = np.arange(len(taus)); ww = .26
    ax[k].bar(xx - ww, d.loc[taus, "点估计recall"], ww * .92, color=GRAY, label="点估计 recall")
    ax[k].bar(xx, d.loc[taus, "区间保守recall"], ww * .92, color=AQUA, label="区间保守 recall")
    ax[k].bar(xx + ww, d.loc[taus, "区间保守precision"], ww * .92, color=ORANGE, label="区间保守 precision")
    for xi, v in zip(xx - ww, d.loc[taus, "点估计recall"]): ax[k].text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(xx, d.loc[taus, "区间保守recall"]): ax[k].text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(xx + ww, d.loc[taus, "区间保守precision"]): ax[k].text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=8)
    ax[k].set_xticks(xx); ax[k].set_xticklabels([f"τ={t}" for t in taus]); ax[k].set_ylim(0, 1.15)
    ax[k].grid(axis="x", visible=False); ax[k].set_title(f"{ds.upper()}（K=2, α=0.05）", loc="left", fontsize=10)
ax[0].legend(frameon=False, fontsize=8.5, ncol=1, loc="lower left"); ax[0].set_ylabel("比例")
fig.suptitle("图2  区间保守判定把 τ-critical 漏判基本消除：PJM τ=0.7 的 recall 0.754 → 0.994，代价是 precision 0.99 → 0.87\n"
             "安全语义上这是正确方向（宁可误报不可漏报）；precision 的下降由后续重训认证复核吸收",
             x=0.005, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .90]); fig.savefig(F / "fig2_critical_interval.png", bbox_inches="tight"); plt.close(fig)

# ---------- 图3：A3 K 递进口径 + A2 元训练定位 ----------
fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.3))
for k, ds in enumerate(["pjm", "caiso"]):
    d = E[E.数据集 == ds].set_index("K")
    kk = np.arange(4)
    ax[k].plot(kk[:3], d.loc[0:2, "正式三攻击器"], "o-", color=BLUE, ms=7, lw=2.4, label="正式三攻击器（正文 K≤2）")
    ax[k].plot(kk[:3], d.loc[0:2, "多目标DNN+树"], "s--", color=ORANGE, ms=5, lw=1.5, label="多目标DNN + 树")
    ax[k].plot(kk, d.loc[0:3, "仅多目标DNN"], "^:", color=GRAY, ms=5, lw=1.5, label="仅多目标DNN（唯一可得 K=3）")
    ax[k].axvspan(2.5, 3.4, color="#f2efe6", zorder=0)
    ax[k].text(2.95, 0.94, "附录\n口径不同", ha="center", va="top", fontsize=8, color="#8a7a4a", transform=ax[k].get_xaxis_transform())
    gap = float(S[S.数据集 == ds].K2处口径差.iloc[0]); inc = float(S[S.数据集 == ds].K2到K3增量_旧口径.iloc[0])
    ax[k].annotate("", (2, d.loc[2, "正式三攻击器"]), (2, d.loc[2, "仅多目标DNN"]),
                   arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
    ax[k].text(1.62, (d.loc[2, "正式三攻击器"] + d.loc[2, "仅多目标DNN"]) / 2, f"口径差\n{gap:.4f}", fontsize=8, color=RED, ha="center")
    ax[k].annotate("", (3, d.loc[3, "仅多目标DNN"]), (3, d.loc[2, "仅多目标DNN"]),
                   arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax[k].text(3.02, (d.loc[3, "仅多目标DNN"] + d.loc[2, "仅多目标DNN"]) / 2, f"K2→K3\n{inc:.4f}", fontsize=8, color=INK)
    ax[k].set_xticks(kk); ax[k].set_xticklabels([f"K={i}" for i in kk]); ax[k].grid(axis="x", visible=False)
    ax[k].set_title(f"{ds.upper()}：口径差 {gap:.4f} > K2→K3 增量 {inc:.4f}", loc="left", fontsize=9.8)
ax[0].set_ylabel("平均 M（41 字段 × 12 目标）"); ax[0].legend(frameon=False, fontsize=7.8, loc="lower right")
# A2 元训练
xm = np.arange(3); wm = .2
d1 = MT[MT.估计器.str.startswith("主口径")].set_index("K"); d2 = MT[MT.估计器.str.startswith("元训练")].set_index("K")
ax[2].bar(xm - 1.5 * wm, d2.认证下界比_top1, wm * .92, color=GRAY, label="元训练 top-1")
ax[2].bar(xm - .5 * wm, d2.认证下界比_top3, wm * .92, color="#cfcec8", label="元训练 top-3")
ax[2].bar(xm + .5 * wm, d1.认证下界比_top1, wm * .92, color="#9fc3ea", label="主口径 top-1")
ax[2].bar(xm + 1.5 * wm, d1.认证下界比_top3, wm * .92, color=BLUE, label="主口径 top-3")
for K_ in [1, 2]:
    ax[2].plot([K_ - 1.5 * wm, K_ + 1.5 * wm], [d2.认证下界比_top1[K_]] * 2, color=RED, lw=1.1, ls=":")
ax[2].set_xticks(xm); ax[2].set_xticklabels([f"K={i}" for i in xm]); ax[2].set_ylim(.78, 1.04)
ax[2].set_ylabel("认证下界比"); ax[2].legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), columnspacing=1.0)
ax[2].grid(axis="x", visible=False)
ax[2].set_title("C. 主口径 top-3 已超过元训练 top-1（红虚线）\n=> 元训练可降级到附录，无需补跑 CAISO", loc="left", fontsize=9.6)
fig.suptitle("图3  A3：K=3 只有「仅多目标 DNN」口径，其与正文口径的差距已大于 K=2→K=3 的阶数增量 => 正文只画到 K=2；"
             "A2：元训练 φ 降级附录", x=0.005, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(F / "fig3_escalation_metatrain.png", bbox_inches="tight"); plt.close(fig)
print("3 张图已写入", F)
