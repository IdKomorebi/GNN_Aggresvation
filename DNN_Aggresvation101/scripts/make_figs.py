# -*- coding: utf-8 -*-
"""101 号图表：读 outputs/analysis/certify_all.csv（6 次 = 3 rep × 2 方向），均值 ± 95% t 区间。"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[1]; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#2b2b2a"
D = pd.read_csv(ROOT / "outputs/analysis/certify_all.csv")


def ms(ds, K, col):
    v = D[(D.ds == ds) & (D.K == K)][col].dropna().values
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) * tdist.ppf(0.975, len(v) - 1)) if len(v) > 1 else 0


fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
# A winner's curse
w = 0.13; x = np.arange(2)
items = [("C1_估计减audit", "估计 M_hat(discovery) − 真 M(audit)", ORANGE), ("C1_disc真值max减其在audit上", "discovery 真值取 max − 同一见证在 audit 上", BLUE),
         ("C1_同数据认证减独立认证", "同数据认证 − 独立认证", GRAY)]
for d_i, ds in enumerate(["pjm", "caiso"]):
    for k_i, (col, lab, color) in enumerate(items):
        m = [ms(ds, K, col) for K in (1, 2)]
        xs = x + d_i * 0.45 + (k_i - 1) * w - 0.22
        ax[0, 0].bar(xs, [a for a, _ in m], w * 0.9, yerr=[b for _, b in m], color=color, alpha=1 if ds == "pjm" else 0.55,
                     label=lab if ds == "pjm" else None, capsize=2, error_kw=dict(lw=0.8))
ax[0, 0].set_xticks([0 - 0.22 + 0, 0 + 0.23, 1 - 0.22, 1 + 0.23]); ax[0, 0].set_xticklabels(["PJM K=1", "CAISO K=1", "PJM K=2", "CAISO K=2"])
ax[0, 0].axhline(0, color="#8a8984", lw=0.8); ax[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left"); ax[0, 0].grid(axis="x", visible=False)
ax[0, 0].set_title("A. winner's curse：max 操作在同数据上的正偏（深=PJM，浅=CAISO）", loc="left", fontsize=10.5)
# B 独立认证下界
for ds, color in [("pjm", BLUE), ("caiso", ORANGE)]:
    for col, ls, lab in [("C2_L除以M_audit", "-", "认证下界 L/M"), ("C2_召回", "--", "召回 L≥M−0.02"), ("C5_档位一致", ":", "档位一致")]:
        m = [ms(ds, K, col) for K in (0, 1, 2)]
        ax[0, 1].errorbar([0, 1, 2], [a for a, _ in m], yerr=[b for _, b in m], color=color, ls=ls, marker="o", capsize=3, lw=1.8, label=f"{ds.upper()} {lab}")
ax[0, 1].set_xticks([0, 1, 2]); ax[0, 1].set_xlabel("K"); ax[0, 1].set_ylim(0, 1.05); ax[0, 1].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
ax[0, 1].set_title("B. 独立数据认证：见证下界、召回与档位一致", loc="left", fontsize=10.5)
# C 阈值与交互复现
cats = [("C3_τ0.7_跨越背景复现率", "τ=0.7 跨越背景\n在 audit 复现", 2), ("C3_τ0.7_关键字段召回", "τ=0.7 关键字段\n召回", 2),
        ("C3_τ0.5_跨越背景复现率", "τ=0.5 跨越背景\n在 audit 复现", 2), ("C4_audit上I大于0", "强交互(I≥0.05)\naudit 上 I>0", 1),
        ("C4_audit上I大于002", "强交互\naudit 上 I>0.02", 1), ("C4_audit强交互召回", "audit 强交互\n被召回", 1)]
x = np.arange(len(cats)); w = 0.38
for d_i, (ds, color) in enumerate([("pjm", BLUE), ("caiso", ORANGE)]):
    m = [ms(ds, K, col) for col, _, K in cats]
    ax[1, 0].bar(x + (d_i - 0.5) * w, [a for a, _ in m], w * 0.9, yerr=[b for _, b in m], color=color, capsize=2, label=ds.upper())
    for xi, (a, b) in zip(x + (d_i - 0.5) * w, m): ax[1, 0].text(xi, a + b + 0.015, f"{a:.2f}", ha="center", fontsize=7.5)
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels([c[1] for c in cats], fontsize=8); ax[1, 0].set_ylim(0, 1.18)
ax[1, 0].legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0)); ax[1, 0].grid(axis="x", visible=False)
ax[1, 0].set_title("C. 阈值跨越（K=2）与强交互在独立数据上的复现", loc="left", fontsize=10.5)
# D 跨数据上界
for ds, color in [("pjm", BLUE), ("caiso", ORANGE)]:
    for a_, ls in [(0.05, "-"), (0.01, "--")]:
        m = [ms(ds, K, f"C6_α{a_}_audit覆盖率") for K in (1, 2)]
        ax[1, 1].errorbar([1, 2], [a for a, _ in m], yerr=[b for _, b in m], color=color, ls=ls, marker="o", capsize=3, lw=1.8, label=f"{ds.upper()} α={a_}")
ax[1, 1].axhline(0.95, color=GRAY, lw=1); ax[1, 1].axhline(0.99, color=GRAY, lw=1, ls="--")
ax[1, 1].set_xticks([1, 2]); ax[1, 1].set_xlim(0.7, 2.3); ax[1, 1].set_ylim(0.6, 1.02); ax[1, 1].set_xlabel("K"); ax[1, 1].legend(frameon=False, fontsize=8.5, loc="lower left")
ax[1, 1].set_title("D. 跨数据上界：discovery 校准 → audit 覆盖率（灰线为名义 95%/99%）", loc="left", fontsize=10.5)
fig.suptitle("图1  101 号独立认证（discovery / audit 完全不相交，3 次随机对半 × 2 方向，误差条为 95% t 区间）", x=0.01, ha="left", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(F / "fig1_independent_certification.png", bbox_inches="tight"); print("-> fig1_independent_certification.png")
