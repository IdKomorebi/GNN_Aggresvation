# -*- coding: utf-8 -*-
"""87 号图：纯高阶协同衰减律 + 覆盖率证书 + 枚举/beam 成本对比。"""
import glob
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

# ---- (a) 衰减律：max_syn 与强协同密度随阶数 ----
ax = axes[0]
p = sorted(glob.glob(str(ROOT / "outputs/decay_cost_B*.csv")))
if p:
    d = pd.read_csv(p[-1]).sort_values("order")
    ax.plot(d.order, d.max_syn, "o-", color="#c0392b", lw=2.5, ms=9, label="max syn（该阶最强纯高阶）")
    ax.axhline(0.20, color="gray", ls=":", lw=1.2)
    ax.annotate("三阶最强真值 0.455\n(82号)", xy=(3, 0.30), fontsize=8, color="#7f8c8d")
    for _, r in d.iterrows():
        tag = "全枚举" if r.source == "全枚举" else f"抽样{int(r.n_eval/1000)}k"
        ax.annotate(f"{r.max_syn:.3f}\n{tag}", (r.order, r.max_syn), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7)
    ax2 = ax.twinx()
    dens = d["dens>0.1"].replace(0, np.nan)
    ax2.semilogy(d.order, dens, "s--", color="#2980b9", lw=2, ms=8, label="密度 syn>0.10")
    hi = d["dens_hi95>0.1"]
    zero = d[d["dens>0.1"] == 0]
    if len(zero):
        ax2.semilogy(zero.order, zero["dens_hi95>0.1"], "v", color="#2980b9", ms=11,
                     label="密度=0 的95%置信上界")
    ax2.set_ylabel("强协同密度 (log)", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax.set_xlabel("协同阶数 m"); ax.set_ylabel("max syn", color="#c0392b")
    ax.set_title("(a) 纯高阶协同衰减律\n三阶→四阶强度断崖(0.47→0.18),密度持续指数衰减")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

# ---- (b) 覆盖率证书：成本 vs 95% 置信下界 ----
ax = axes[1]
c = pd.read_csv(ROOT / "outputs/coverage_certificate.csv")
for m, col in [(4, "#27ae60"), (5, "#8e44ad")]:
    s = c[c.order == m].sort_values("cost_frac")
    ax.plot(s.cost_frac * 100, s["cov>0.1"], "o-", color=col, lw=2, label=f"order-{m} 覆盖率")
    ax.fill_between(s.cost_frac * 100, s["lo95>0.1"], s["cov>0.1"], color=col, alpha=0.18)
    ax.plot(s.cost_frac * 100, s["lo95>0.1"], ":", color=col, lw=1.5,
            label=f"order-{m} 95%置信下界")
ax.set_xlabel("触及候选占全枚举 % (成本)"); ax.set_ylabel("强协同(syn>0.10)覆盖率")
ax.set_title("(b) 覆盖率证书（阴影=到95%下界）\n阶数越高,同覆盖率的相对成本越低")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- (c) 枚举爆炸 vs beam 触及 ----
ax = axes[2]
if p:
    d = pd.read_csv(p[-1]).sort_values("order")
    ax.semilogy(d.order, d.full_comb, "r^-", lw=2, ms=9, label="全枚举 C(44,m)")
    ax.semilogy(d.order, d.touched, "go-", lw=2, ms=9, label="beam-syn B=1000 触及")
    for _, r in d.iterrows():
        ax.annotate(f"{r.touch_frac*100:.2f}%", (r.order, r.touched), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="#27ae60")
    ax.set_xlabel("协同阶数 m"); ax.set_ylabel("集合数 (log)")
    ax3 = ax.twinx()
    cov = d["cov>0.1"]
    ax3.plot(d.order, cov, "d-.", color="#e67e22", lw=2, ms=9, label="beam 覆盖率 syn>0.10")
    for _, r in d.iterrows():
        if not np.isnan(r["cov>0.1"]):
            ax3.annotate(f"{r['cov>0.1']:.2f}\n(n={int(r['n>0.1'])})", (r.order, r["cov>0.1"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7, color="#e67e22")
    ax3.set_ylabel("beam 覆盖率", color="#e67e22"); ax3.set_ylim(-0.05, 1.15)
    ax3.tick_params(axis="y", labelcolor="#e67e22")
    ax.set_title("(c) 触及数近恒定(~3万) vs 枚举指数爆炸\n★但覆盖率随阶下降:高阶强协同不再嵌套于强低阶")
    h1, l1 = ax.get_legend_handles_labels(); h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, fontsize=8, loc="center left")
    ax.grid(alpha=0.3, which="both")

fig.suptitle("DNN87：高阶协同的衰减律与 beam 搜索的覆盖率证书（零 GPU）", fontsize=13)
fig.tight_layout()
fig.savefig(ROOT / "figures/decay_and_certificate_zh.png", dpi=150)
print("已写出 figures/decay_and_certificate_zh.png")
