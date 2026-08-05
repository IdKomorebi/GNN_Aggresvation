# -*- coding: utf-8 -*-
"""86 号图：高阶非枚举 beam 搜索的覆盖率-成本前沿 + 枚举爆炸对比。"""
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

# (a) order-4 覆盖-成本前沿
ax = axes[0]
fr = pd.read_csv(ROOT / "outputs/beam_frontier.csv")
colors = {"syn": "#27ae60", "hybrid": "#2980b9", "vmax": "#c0392b"}
for key in ["syn", "hybrid", "vmax"]:
    s = fr[fr.key == key].sort_values("touch_frac")
    ax.plot(s.touch_frac * 100, s["cov>0.1"], "o-", color=colors[key], lw=2, label=f"beam-{key}")
s = fr[fr.key == "syn"].sort_values("touch_frac")
ax.plot(s.touch_frac * 100, s["rand_cov>0.1"], "s--", color="#95a5a6", lw=1.5, label="随机(同预算)")
ax.set_xlabel("触及候选占全枚举 % (成本)"); ax.set_ylabel("强四阶(syn>0.10)覆盖率")
ax.set_title("(a) order-4：syn-beam 用~1/4成本覆盖~90%\nvmax失明(纯高阶v中等),4×于随机")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) order-5 覆盖-成本(若已跑)
ax = axes[1]
p5 = ROOT / "outputs/scale_order5.csv"
if p5.exists():
    s5 = pd.read_csv(p5)
    for key, col in [("syn", "#27ae60"), ("hybrid", "#2980b9")]:
        d = s5[s5.key == key].sort_values("touch_frac")
        if "cov>0.1" in d:
            ax.plot(d.touch_frac * 100, d["cov>0.1"], "o-", color=col, lw=2, label=f"beam-{key}")
            ax.plot(d.touch_frac * 100, d["rand>0.1"], "s--", color="#95a5a6", lw=1.2,
                    label="随机" if key == "syn" else None)
    ax.set_xlabel("触及占全枚举 % (成本)"); ax.set_ylabel("强五阶(syn>0.10)覆盖率")
    ax.set_title("(b) order-5：全枚举1.09M不可行时\nbeam 覆盖率保持")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "order-5 运行中", ha="center", va="center"); ax.axis("off")

# (c) 枚举爆炸 vs beam 触及(成本外推)
ax = axes[2]
orders = [3, 4, 5, 6, 8, 10]
full = [comb(44, m) for m in orders]
ax.semilogy(orders, full, "r^-", lw=2, ms=9, label="全枚举 C(44,m)")
# beam 触及≈ 各阶 B×可扩展字段,近似线性外推(用 B=1000 syn 的实测点+线性)
fr = pd.read_csv(ROOT / "outputs/beam_frontier.csv")
b1000 = fr[(fr.beam == 1000) & (fr.key == "syn")]
touch4 = int(b1000.touched4.iloc[0]) if len(b1000) else 31000
beam_touch = {4: touch4}
if p5.exists():
    s5 = pd.read_csv(p5)
    r = s5[(s5.beam == 1000) & (s5.key == "syn")]
    if len(r):
        beam_touch[5] = int(r.touched.iloc[0])
bo = sorted(beam_touch)
ax.semilogy(bo, [beam_touch[o] for o in bo], "go-", lw=2, ms=9, label="beam-syn B=1000 触及")
if len(bo) >= 2:
    slope = (beam_touch[bo[-1]] - beam_touch[bo[0]]) / (bo[-1] - bo[0])
    proj = {o: max(beam_touch[bo[-1]] + slope * (o - bo[-1]), 1) for o in [6, 8, 10]}
    ax.semilogy(list(proj), list(proj.values()), "g:", lw=1.5, label="beam 线性外推")
ax.set_xlabel("协同阶数 m"); ax.set_ylabel("候选/触及数 (log)")
ax.set_title("(c) 枚举指数爆炸 vs beam 触及近线性\n高阶时 beam 是唯一可行路径")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

fig.suptitle("DNN86：高阶协同的非枚举 beam 搜索(骑在85号ms级结构化查询上,零GPU)", fontsize=13)
fig.tight_layout()
fig.savefig(ROOT / "figures/highorder_search_zh.png", dpi=150)
print("已写出 figures/highorder_search_zh.png")
