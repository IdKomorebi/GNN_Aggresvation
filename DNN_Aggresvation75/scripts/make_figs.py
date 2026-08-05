# -*- coding: utf-8 -*-
"""DNN75 主结果图。

图 1 采样阶梯：各方案的 K=0 MAE 分三个查询区间（扫描/中段/防护），
              含用户要求的 none / bern50 阶梯底。
图 2 尺寸分布 vs 效果：上=各方案训练掩码的尺寸直方图，下=对应的分带 MAE。
图 3 协同检出：Spearman / top-20 命中率 / 真值 top-10 的估计排名中位数。
图 4 路由：三区间同时不劣于 uniform 的组合。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R75 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R75 / "scripts"))
sys.path.insert(0, str(R75 / "src"))
from analyze import BANDS, REGIONS          # noqa: E402
from samplers import ALL, size_histogram    # noqa: E402

for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
          "Source Han Sans CN", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

ORDER = ["none", "bern50", "uniform", "logunif", "small50", "small80",
         "workload", "workload_hard", "large50"]
COL = {"none": "#7f8c8d", "bern50": "#95a5a6", "uniform": "#c0392b"}


def color(sc):
    return COL.get(sc, "#2980b9")


def main():
    s = pd.read_csv(R75 / "outputs/scheme_comparison.csv", index_col=0)
    s = s.reindex([x for x in ORDER if x in s.index])
    x = np.arange(len(s))
    cols = [color(i) for i in s.index]

    # ---------------- 图 1：三区间 MAE ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (_, _, nm) in zip(axes, REGIONS):
        c = f"mae_{nm}"
        ax.bar(x, s[c], color=cols)
        if "uniform" in s.index:
            ax.axhline(s.loc["uniform", c], color="#c0392b", ls="--", lw=1.8,
                       label=f"uniform 基准 {s.loc['uniform', c]:.4f}")
        ax.set_xticks(x); ax.set_xticklabels(s.index, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{nm}"); ax.set_ylabel("K=0 MAE"); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("DNN75 图1：训练掩码尺寸分配 → 三个查询区间的摊销保真度", fontsize=13)
    fig.tight_layout(); fig.savefig(R75 / "figures/regions_zh.png", dpi=150)

    # ---------------- 图 2：尺寸分布 vs 分带 MAE ----------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    ax = axes[0]
    for sc in s.index:
        h = size_histogram(sc, 44, 100_000, seed=0)
        ax.plot(np.arange(1, 45), h[1:], lw=2, alpha=0.85, color=color(sc), label=sc)
    ax.set_yscale("log"); ax.set_xlabel("训练掩码的可见字段数 k"); ax.set_ylabel("概率")
    ax.set_title("(a) 各方案的训练掩码尺寸分布"); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)

    ax = axes[1]
    bcols = [f"band_{nm}" for _, _, nm in BANDS if f"band_{nm}" in s.columns]
    xb = np.arange(len(bcols))
    for sc in s.index:
        ax.plot(xb, s.loc[sc, bcols].values, "o-", lw=2, alpha=0.85,
                color=color(sc), label=sc)
    ax.set_xticks(xb); ax.set_xticklabels([c.replace("band_", "") for c in bcols])
    ax.set_xlabel("评测子集的规模带"); ax.set_ylabel("K=0 MAE")
    ax.set_title("(b) 对应的分带保真度（评测集每带 ≥10，中大带已从每带 10 扩到 44/49/52/40）")
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(R75 / "figures/size_alloc_zh.png", dpi=150)

    # ---------------- 图 3：协同检出 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (c, lab, better) in zip(axes, [
            ("syn_rho", "二阶协同 Spearman", "高"),
            ("syn_top20", "真值 top-20 命中率", "高"),
            ("syn_top10_medrank", "真值 top-10 的估计排名中位数", "低")]):
        if c not in s.columns:
            continue
        ax.bar(x, s[c], color=cols)
        if "uniform" in s.index:
            ax.axhline(s.loc["uniform", c], color="#c0392b", ls="--", lw=1.8, label="uniform")
        ax.set_xticks(x); ax.set_xticklabels(s.index, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{lab}（越{better}越好）"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        if c == "syn_top10_medrank":
            ax.set_yscale("log")
            ax.axhline(1000, color="#27ae60", ls=":", lw=1.6, label="失明缓解线 1000")
            ax.legend(fontsize=8)
    fig.suptitle("DNN75 图3：二阶协同检出质量（946 对 × 12 conf，对 68 号重训真值）", fontsize=13)
    fig.tight_layout(); fig.savefig(R75 / "figures/synergy_zh.png", dpi=150)

    # ---------------- 图 4：路由 ----------------
    p = R75 / "outputs/routing_comparison.csv"
    if p.exists():
        r = pd.read_csv(p).sort_values("K0_mae").head(12)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        lbl = [f"{k}:{a}|{m}|{b}\nt={t1},{t2}" for k, a, m, b, t1, t2 in
               zip(r["kind"], r["small"], r["mid"], r["large"], r["t1"], r["t2"])]
        xr = np.arange(len(r))
        w = 0.27
        for i, (_, _, nm) in enumerate(REGIONS):
            ax.bar(xr + (i - 1) * w, r[f"mae_{nm}"], w, label=nm)
        for i, (_, _, nm) in enumerate(REGIONS):
            ax.axhline(s.loc["uniform", f"mae_{nm}"], ls="--", lw=1.3,
                       color=f"C{i}", alpha=0.7)
        ax.set_xticks(xr); ax.set_xticklabels(lbl, rotation=0, ha="center", fontsize=6.5)
        ax.set_ylabel("K=0 MAE"); ax.legend(fontsize=9)
        ax.set_title("DNN75 图4：路由双模型（虚线 = uniform 单模型在对应区间的基准）")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(R75 / "figures/routing_zh.png", dpi=150)

    print("已写出 figures/{regions,size_alloc,synergy,routing}_zh.png")


if __name__ == "__main__":
    main()
