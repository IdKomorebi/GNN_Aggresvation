# -*- coding: utf-8 -*-
"""76 号主结果图。

图 1 相变曲线：三种方案的 med_rank10 / 协同 ρ / 强协同检出 随 K 的变化，标出 K*。
图 2 F-2 帕累托：平均步数 vs 强协同召回，自适应规则 vs 固定 K。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R76 = Path(__file__).resolve().parents[1]
for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
          "Source Han Sans CN", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

COL = {"uniform": "#c0392b", "bern50": "#f39c12", "none": "#7f8c8d"}
LBL = {"uniform": "两段式失活（本文）", "bern50": "伯努利(0.5)", "none": "无随机失活"}


def main():
    agg = pd.read_csv(R76 / "outputs/f1_curves.csv")
    raw = pd.read_csv(R76 / "outputs/f1_by_seed.csv")
    ks = json.load(open(R76 / "outputs/kstar.json")) if (R76 / "outputs/kstar.json").exists() else {}

    # ---------------- 图 1：相变曲线 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [("med_rank10", "真 top10 协同的估计排名中位数", "log"),
              ("rho", "二阶协同 Spearman", None),
              ("recall", "强协同检出率（syn>0.2）", None)]
    for ax, (col, title, scale) in zip(axes, panels):
        for sc in ["none", "bern50", "uniform"]:
            a = agg[agg.scheme == sc].sort_values("K")
            if not len(a):
                continue
            s = raw[raw.scheme == sc].groupby("K")[col].std().reindex(a.K).values
            ax.plot(a.K, a[col], "o-", color=COL[sc], lw=2.2, ms=4, label=LBL[sc])
            ax.fill_between(a.K, a[col] - np.nan_to_num(s), a[col] + np.nan_to_num(s),
                            color=COL[sc], alpha=0.15)
        if scale:
            ax.set_yscale(scale)
        if col == "med_rank10":
            ax.axhline(50, color="#27ae60", ls=":", lw=1.8, label="K* 判据线（排名 50）")
            for sc, k in ks.items():
                if k is not None:
                    ax.axvline(k, color=COL.get(sc, "k"), ls="--", lw=1.2, alpha=0.7)
                    ax.annotate(f"K*={k}", (k, ax.get_ylim()[1] * 0.4),
                                color=COL.get(sc, "k"), fontsize=9, rotation=90,
                                ha="right", va="top")
        ax.set_xlabel("微调步数 K")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("DNN76 图1：二阶协同的发现相变（990 低阶子集，946 对 × 12 conf，3 seed）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(R76 / "figures/f1_phase_zh.png", dpi=150)

    # ---------------- 图 2：F-2 帕累托 ----------------
    p = R76 / "outputs/f2_earlystop.csv"
    if p.exists():
        d = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(9, 6))
        fx = d[d.rule == "fixed"].sort_values("mean_steps")
        ax.plot(fx.mean_steps, fx.recall, "s-", color="#7f8c8d", lw=2,
                label="固定 K 基线", zorder=3)
        ad = d[d.rule == "adaptive"]
        sc = ax.scatter(ad.mean_steps, ad.recall, c=ad.fp, cmap="viridis",
                        s=55, edgecolor="k", linewidth=0.4, label="自适应规则", zorder=4)
        plt.colorbar(sc, ax=ax, label="误报率")
        b50 = fx[fx.K == 50]
        if len(b50):
            ax.axhline(float(b50.recall.iloc[0]), color="#c0392b", ls="--", lw=1.5,
                       label=f"固定 K=50 的召回 {float(b50.recall.iloc[0]):.1%}")
            ax.axvline(50, color="#c0392b", ls=":", lw=1.2)
        ax.set_xlabel("平均微调步数（越小越省）")
        ax.set_ylabel("强协同召回率（syn>0.2，越高越好）")
        ax.set_title("DNN76 图2：自适应早停 vs 固定 K 的帕累托前沿")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(R76 / "figures/f2_pareto_zh.png", dpi=150)

    print("已写出 figures/f1_phase_zh.png" + ("、f2_pareto_zh.png" if p.exists() else ""))


if __name__ == "__main__":
    main()
