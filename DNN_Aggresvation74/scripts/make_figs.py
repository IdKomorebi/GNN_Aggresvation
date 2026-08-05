# -*- coding: utf-8 -*-
"""DNN74：主结果图。

图 1（保真度）：各变体的 K=0 MAE 与 Spearman，横线标出 MLP 目标 / 63 号现状 / 判据门槛。
图 2（机制）：K=0 分尺寸带 MAE + 满输入 R²，看改善集中在哪个规模带。

用法：python scripts/make_figs.py --round A|B|final
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R74 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R74 / "scripts"))
from analyze import BAND_ORDER, GATE, REF  # noqa: E402

for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
          "Source Han Sans CN", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

GNN_BAND_REF = [0.072, 0.133, 0.167, 0.201, 0.138, 0.107]   # 73 号 GNN ours K=0
MLP_BAND_REF = [0.043, 0.089, 0.109, 0.126, 0.076, 0.037]   # 73 号 MLP ours K=0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="A", choices=["A", "B", "final"])
    a = ap.parse_args()

    src = R74 / "outputs" / f"variant_comparison_{a.round}.csv"
    if not src.exists():
        raise SystemExit(f"先跑 analyze.py --round {a.round}")
    df = pd.read_csv(src, index_col=0).sort_values("K0_mae")

    # ---------------- 图 1：保真度 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(df))
    colors = ["#c0392b" if v.startswith("A0") else
              "#7f8c8d" if v.startswith("D_") else "#2980b9" for v in df.index]

    ax = axes[0]
    ax.bar(x, df.K0_mae, yerr=df.K0_mae_sd.fillna(0), color=colors, capsize=3)
    ax.axhline(REF["mlp_ours"]["K0_mae"], color="#27ae60", ls="-", lw=2,
               label=f"MLP-oracle 目标 {REF['mlp_ours']['K0_mae']:.4f}")
    ax.axhline(GATE["promising_mae"], color="#f39c12", ls="--", lw=1.8,
               label=f"「有戏」门槛 {GATE['promising_mae']:.3f}（缩差70%）")
    ax.axhline(GATE["abandon_mae"], color="#e67e22", ls=":", lw=1.8,
               label=f"「放弃」门槛 {GATE['abandon_mae']:.3f}")
    ax.axhline(REF["gnn_ours"]["K0_mae"], color="#c0392b", ls="-.", lw=1.8,
               label=f"63 号 GNN 现状 {REF['gnn_ours']['K0_mae']:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("K=0 MAE（对 394 子集重训真值）")
    ax.set_title("(a) 摊销保真度：越低越好")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, df.K0_rho, yerr=df.K0_rho_sd.fillna(0), color=colors, capsize=3)
    ax.axhline(REF["mlp_ours"]["K0_rho"], color="#27ae60", ls="-", lw=2,
               label=f"MLP-oracle 目标 {REF['mlp_ours']['K0_rho']:.4f}")
    ax.axhline(GATE["promising_rho"], color="#f39c12", ls="--", lw=1.8,
               label=f"「有戏」门槛 {GATE['promising_rho']:.3f}")
    ax.axhline(REF["gnn_ours"]["K0_rho"], color="#c0392b", ls="-.", lw=1.8,
               label=f"63 号 GNN 现状 {REF['gnn_ours']['K0_rho']:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(min(0.85, df.K0_rho.min() - 0.02), 1.0)
    ax.set_ylabel("K=0 Spearman")
    ax.set_title("(b) 排序保真度：越高越好")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"DNN74 Round {a.round}：图 oracle 架构修复后的保真度", fontsize=13)
    fig.tight_layout()
    fig.savefig(R74 / "figures" / f"fidelity_{a.round}_zh.png", dpi=150)

    # ---------------- 图 2：机制 ----------------
    bcols = [f"K0_mae_{b}" for b in BAND_ORDER if f"K0_mae_{b}" in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    xb = np.arange(len(bcols))
    ax.plot(xb, GNN_BAND_REF[:len(bcols)], "o--", color="#c0392b", lw=2, label="63 号 GNN 现状")
    ax.plot(xb, MLP_BAND_REF[:len(bcols)], "s-", color="#27ae60", lw=2, label="MLP-oracle 目标")
    for v in df.index[:4]:
        ax.plot(xb, df.loc[v, bcols].values, "^-", alpha=0.85, label=v)
    ax.set_xticks(xb); ax.set_xticklabels([b for b in BAND_ORDER if f"K0_mae_{b}" in df.columns])
    ax.set_xlabel("集合规模带（可见字段数）"); ax.set_ylabel("K=0 MAE")
    ax.set_title("(a) 改善集中在哪个规模带")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.barh(np.arange(len(df)), df.r2_full, color=colors)
    ax.axvline(REF["mlp_ours"]["r2_full"], color="#27ae60", lw=2,
               label=f"MLP {REF['mlp_ours']['r2_full']:.3f}")
    ax.axvline(REF["gnn_ours"]["r2_full"], color="#c0392b", ls="-.", lw=2,
               label=f"63 号 GNN {REF['gnn_ours']['r2_full']:.3f}")
    ax.axvline(0.8746, color="#8e44ad", ls=":", lw=2, label="52 号 gcn_dynamic 攻击者侧 0.875")
    ax.set_yticks(np.arange(len(df))); ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlim(0.70, 0.92); ax.set_xlabel("满输入 12-conf 平均 R²")
    ax.set_title("(b) 预测准确率（满输入）")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"DNN74 Round {a.round}：机制诊断", fontsize=13)
    fig.tight_layout()
    fig.savefig(R74 / "figures" / f"mechanism_{a.round}_zh.png", dpi=150)
    print(f"已写出 figures/fidelity_{a.round}_zh.png 与 figures/mechanism_{a.round}_zh.png")


if __name__ == "__main__":
    main()
