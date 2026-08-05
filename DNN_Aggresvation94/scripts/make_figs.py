# -*- coding: utf-8 -*-
"""94 号出图：真值版衰减律 / o4 三组对照 / 反层级 / 成本。"""
from __future__ import annotations

import ast
import glob
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
R93 = ROOT.parent / "DNN_Aggresvation93"
FIG = ROOT / "figures"


def load_cert(pattern):
    fs = glob.glob(str(R93 / f"outputs/{pattern}"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    return d


fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

# ---- (a) 真值版衰减律 ----
ax = axes[0]
orders = [3, 4, 5]
maxima = [0.455, 0.196, 0.154]
poly2 = [0.4915, 0.2553, 0.3650]
fullv = [0.5031, 0.3721, 0.5137]
ax.plot(orders, maxima, "o-", color="#27ae60", lw=2.5, ms=8, label="认证真值（本工作）", zorder=5)
ax.plot(orders, poly2, "s--", color="#95a5a6", lw=1.3, label="poly2 估计（87号，断崖假象）")
ax.plot(orders, fullv, "^--", color="#e74c3c", lw=1.3, label="full 估计（90号，不衰减假象）")
for o, m in zip(orders, maxima):
    ax.annotate(f"{m:.3f}", (o, m), xytext=(6, 6), textcoords="offset points",
                fontsize=9, color="#27ae60", fontweight="bold")
ax.set_xticks(orders)
ax.set_xlabel("协同阶数")
ax.set_ylabel("该阶最强 syn")
ax.set_title("(a) ★真值版衰减律：温和单调（2.3×,1.3×）\n两种手工字典的版本都是假象", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---- (b) o4 三组对照 ----
ax = axes[1]
l1t = load_cert("certify_o4_l1top_s*.csv")
try:
    ft = load_cert("certify_o4_fulltop_s*.csv")
except ValueError:
    ft = pd.DataFrame(columns=l1t.columns)
groups = [("L1 自选 top", l1t[l1t.group == "strong"], "#27ae60"),
          ("full 字典 top", ft, "#e74c3c"),
          ("随机对照", l1t[l1t.group == "control"], "#95a5a6")]
for i, (lab, g, col) in enumerate(groups):
    y = g.syn_true_audit.values
    if not len(y):
        continue
    ax.scatter(np.random.RandomState(i).normal(i, 0.07, len(y)), y, s=32,
               alpha=0.75, color=col, edgecolors="w", linewidths=0.4)
    ax.hlines(np.median(y), i - 0.24, i + 0.24, color="k", lw=2)
    ax.annotate(f"n={len(y)}\n过0.10: {int((y>0.10).sum())}", (i, -0.022),
                ha="center", fontsize=8, color=col)
ax.axhline(0.10, color="#27ae60", ls=":", lw=1.2)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("专用 DNN 重训真值 syn")
ax.set_title("(b) order-4 复刻 o5 结论\nL1 选的 top 是真的，full 选的不是", fontsize=10)
ax.set_ylim(-0.035, 0.22)
ax.grid(alpha=0.3, axis="y")

# ---- (c) 反层级：认证真协同的最好父集排名 ----
ax = axes[2]
h = pd.read_csv(ROOT / "outputs/hierarchy_check_o5.csv")
ranks = np.sort(h.best_parent_rank.values)
ax.barh(range(len(ranks)), ranks, color="#2980b9", alpha=0.85)
for B, col in [(100, "#27ae60"), (1000, "#e67e22"), (5000, "#e74c3c")]:
    ax.axvline(B, color=col, ls="--", lw=1.4)
    ax.annotate(f"B={B}", (B * 1.1, len(ranks) - 0.5), fontsize=8, color=col)
ax.set_xscale("log")
ax.set_xlabel("最好四阶父集在 o4 全扫中的 syn 排名（对数轴）")
ax.set_ylabel("10 个已认证真五阶协同")
ax.set_title("(c) ★反层级的认证级测量\nbeam B=1000 漏 4/10、B=100 全漏 ⟹ 全枚举消解", fontsize=10)
ax.grid(alpha=0.3, axis="x")

# ---- (d) 成本（对数轴） ----
ax = axes[3]
items = [("暴力认证\no5 全空间", 424 * 24, "#e74c3c"),
         ("ft25 扫描\no5 全空间", 18.8, "#e67e22"),
         ("L1 管线\n(扫描+认证top)", 1.8, "#27ae60")]
b = ax.bar([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items])
ax.set_yscale("log")
ax.set_ylabel("GPU·小时（对数轴）")
for r, (_, v, _) in zip(b, items):
    ax.annotate(f"{v:g} h" if v < 100 else f"{v/24:.0f} 天",
                (r.get_x() + r.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
ax.set_title("(d) 成本：全管线 ≈5,600× 于暴力认证\n查询级 2.5ms vs 微调 55.5ms(25×)", fontsize=10)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("94 号（收束）：真值版衰减律 + 全面领先判定 + 反层级认证级回答 + 成本",
             fontsize=12.5, y=1.00)
fig.tight_layout(rect=(0, 0, 1, 0.94))
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "final_wrapup.png", dpi=155)
print(f"已保存 {FIG / 'final_wrapup.png'}")
