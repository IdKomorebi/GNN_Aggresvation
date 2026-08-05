# -*- coding: utf-8 -*-
"""91 号出图：摊销边界的定位与闭式读出的修复效果。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
BANDS = [-9, 0.05, 0.10, 0.20, 0.35, 9]
LAB = ["<.05", ".05-.1", ".1-.2", ".2-.35", ">.35"]
C = {"oracle": "#e74c3c", "oracle_affine": "#e67e22", "poly2": "#95a5a6",
     "full": "#7f8c8d", "last": "#27ae60", "cat3": "#2980b9", "L0": "#27ae60"}

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

# ---- (a) 核心图：gap 随真值强度的走势（二阶）----
ax = axes[0, 0]
d = pd.read_csv(OUT / "order2_pairs.csv")
d["band"] = pd.cut(d.truth, BANDS, labels=LAB)
for k in ["oracle", "oracle_affine", "poly2", "last"]:
    g = d[d.kind == k]
    if not len(g):
        continue
    s = g.groupby("band", observed=True).apply(
        lambda x: (x.truth - x.est).mean(), include_groups=False)
    ax.plot(range(len(s)), s.values, "o-", color=C[k], label=k, lw=2, ms=6)
ax.set_xticks(range(len(LAB)))
ax.set_xticklabels(LAB)
ax.set_xlabel("重训真值 syn 所在档")
ax.set_ylabel("低估量 gap = 真值 − 估计")
ax.set_title("(a) ★摊销 gap 随协同强度单调放大\n闭式读出把最强档的 0.382 压到 0.018", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.6)

# ---- (b) 真 top10 的估计排名 ----
ax = axes[0, 1]
o = d[d.kind == "oracle"].reset_index(drop=True)
top = o.nlargest(10, "truth").index
names, vals = [], []
for k in ["oracle", "oracle_affine", "poly2", "last"]:
    g = d[d.kind == k].reset_index(drop=True)
    if not len(g):
        continue
    names.append(k)
    vals.append(np.median((-g.est).rank()[top]))
b = ax.bar(names, vals, color=[C[k] for k in names])
ax.set_yscale("log")
ax.set_ylabel("真值 top10 的估计排名中位数（越小越好，共 11352 条）")
ax.set_title("(b) 共享读出把最强协同排到 6527 名\n闭式重解拉回第 4.5 名", fontsize=10)
for r, v in zip(b, vals):
    ax.annotate(f"{v:.0f}", (r.get_x() + r.get_width() / 2, v),
                ha="center", va="bottom", fontsize=9)
ax.grid(alpha=0.3, axis="y")

# ---- (c) 尾部 Spearman（三个 seed）----
ax = axes[0, 2]
frames = []
for tag in ["", "_s1", "_s2"]:
    p = OUT / f"order2_truth{tag}.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
t = pd.concat(frames)
ks = ["oracle", "oracle_affine", "poly2", "last", "cat3"]
x = np.arange(len(ks))
for i, (metric, mk) in enumerate([("rho", "o"), ("rho_tail", "s")]):
    m = [t[t.kind == k][metric].mean() for k in ks]
    e = [t[t.kind == k][metric].std() for k in ks]
    ax.errorbar(x + i * 0.08, m, yerr=e, fmt=mk + "-", capsize=3,
                label="全体 ρ" if metric == "rho" else "尾部 ρ(真值top100)", lw=2)
ax.set_xticks(x)
ax.set_xticklabels(ks, rotation=20, fontsize=8)
ax.set_ylabel("Spearman ρ")
ax.set_title("(c) 三个 oracle seed 的稳健性\n尾部：共享读出 0.12 → 闭式 0.78", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---- (d) ★E4：闭式读出 vs 微调 ----
ax = axes[1, 0]
s = pd.read_csv(OUT / "readout_vs_feature_summary.csv")
order = ["oracle", "ft5", "ft25", "ft50", "L0", "ft25+L0"]
s = s.set_index("method").reindex(order).reset_index()
cols = ["#e74c3c", "#f39c12", "#f39c12", "#f39c12", "#27ae60", "#16a085"]
ax.bar(s.method, s.gap_strong, color=cols)
ax2 = ax.twinx()
ax2.plot(s.method, s.rho, "ko--", ms=5, lw=1.2, label="ρ（右轴）")
ax2.set_ylabel("Spearman ρ")
ax2.legend(fontsize=8, loc="upper center")
ax.set_ylabel("最强档低估量 gap")
ax.set_title("(d) ★零训练的闭式读出 > 50 步微调\nL0 gap 0.016 vs ft50 0.036；ρ 0.904 vs 0.899",
             fontsize=10)
ax.tick_params(axis="x", rotation=20)
ax.grid(alpha=0.3, axis="y")

# ---- (e) 三阶：手工字典 vs 学出来的特征 ----
ax = axes[1, 1]
t3 = pd.read_csv(OUT / "order3_truth.csv")
ks3 = [k for k in ["oracle", "poly2", "full", "last", "cat3"] if k in set(t3.kind)]
x = np.arange(len(ks3))
ax.bar(x - 0.2, [t3[t3.kind == k].rho.iloc[0] for k in ks3], 0.4,
       label="全体 ρ", color="#bdc3c7")
ax.bar(x + 0.2, [t3[t3.kind == k].rho_tail.iloc[0] for k in ks3], 0.4,
       label="尾部 ρ", color="#2980b9")
ax.set_xticks(x)
ax.set_xticklabels(ks3, rotation=20, fontsize=8)
ax.set_ylabel("Spearman ρ（对无偏三阶重训真值）")
ax.set_title("(e) 三阶：手工全交互字典(full)的尾部\n反而不如学出来的特征", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# ---- (f) 诚实的负结果：注入的合成纯四阶 ----
ax = axes[1, 2]
frames = [pd.read_csv(p) for p in
          [OUT / "inject_l0_summary_smoke.csv", OUT / "inject_l0_summary_spd.csv",
           OUT / "inject_l0_summary_spd3.csv", OUT / "inject_l0_summary_p2.csv"]
          if p.exists()]
inj = pd.concat(frames).drop_duplicates(subset=["kind", "alpha"])
for k in ["poly2", "full", "last", "cat3", "t_test"]:
    g = inj[inj.kind == k].sort_values("alpha")
    if not len(g):
        continue
    ax.plot(g.alpha, g.pct, "o-", label=k,
            color=C.get(k, "#8e44ad"), lw=2 if k in ("last", "cat3") else 1.2)
ax.axhline(50, color="k", ls=":", lw=1)
ax.annotate("随机水平", (0.11, 52), fontsize=8, color="#7f8c8d")
ax.set_xlabel("注入信号强度 α")
ax.set_ylabel("注入集合在 400 个随机四元组中的百分位")
ax.set_title("(f) 诚实的负结果：对**合成**纯四阶\nφ 不是通用基，输给手工 full 字典", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("91 号：摊销的层次错了——共享读出是尾部失明的原因，闭式重解是零训练的修复",
             fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.975))
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "amortization_boundary.png", dpi=155)
print(f"已保存 {FIG / 'amortization_boundary.png'}")
