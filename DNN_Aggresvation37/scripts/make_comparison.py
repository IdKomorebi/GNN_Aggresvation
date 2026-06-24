#!/usr/bin/env python3
"""DNN37 跨 run 汇总：K 专家「分辨率代偿」曲线。

5 臂 experts_k{1,3,4,6,12}：12 个字段共享 K 个专家（只专属 value+输出头），
每字段用全局软分配 π_c 组合。K 是"等价推断 K 个字段"的分辨率旋钮。

读取 outputs/<variant>/<latest_timestamp>/ 下的：
  - tables/r2_probe_vs_model.csv  （每个 confidential 字段的 model_r2）
  - results/summary.json          （best_test_loss）
生成：
  - outputs/experts_comparison.csv          每字段 × 各 K 的 R²
  - outputs/experts_curve.png               K–精度代偿曲线（均值 R² / best_loss）
  - outputs/experts_per_field.png           每字段 × 各 K + single 上限柱状图
并打印关键统计，供 CHANGELOG 引用。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
SINGLE_CSV = (
    Path(__file__).resolve().parents[2]
    / "DNN_Aggresvation32" / "outputs" / "multi_vs_single.csv"
)
KS = [1, 3, 4, 6, 12]
VARIANTS = [f"experts_k{k}" for k in KS]


def latest(variant: str) -> Path | None:
    dirs = sorted(p for p in (OUT / variant).glob("*/") if p.is_dir())
    return dirs[-1] if dirs else None


def main() -> None:
    rows: dict[str, dict[str, float]] = {}
    probe: dict[str, float] = {}
    best_loss: dict[str, float] = {}
    field_order: list[str] = []

    for v in VARIANTS:
        d = latest(v)
        if d is None:
            print(f"[warn] 缺少 variant: {v}")
            continue
        df = pd.read_csv(d / "tables" / "r2_probe_vs_model.csv")
        for _, r in df.iterrows():
            f = r["field"]
            if f not in rows:
                rows[f] = {}
                field_order.append(f)
            rows[f][v] = float(r["model_r2"])
            probe[f] = float(r["probe_r2"])
        sj = json.loads((d / "results" / "summary.json").read_text())
        best_loss[v] = float(sj.get("best_test_loss", float("nan")))

    single_r2: dict[str, float] = {}
    if SINGLE_CSV.exists():
        sdf = pd.read_csv(SINGLE_CSV)
        single_r2 = dict(zip(sdf["field"], sdf["single_r2"]))
    else:
        print(f"[warn] 缺少 single 上限文件: {SINGLE_CSV}")

    table = pd.DataFrame(
        {v: [rows[f].get(v, np.nan) for f in field_order] for v in VARIANTS},
        index=field_order,
    )
    table.insert(0, "single_r2", [single_r2.get(f, np.nan) for f in field_order])
    table.insert(0, "probe_r2", [probe[f] for f in field_order])
    out_csv = OUT / "experts_comparison.csv"
    table.to_csv(out_csv)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n===== per-field model R² =====")
    print(table.round(4).to_string())

    # 代偿曲线统计：每个 K 的均值 R²、best_loss、距 single 的均值差
    mean_r2 = {v: float(np.nanmean(table[v])) for v in VARIANTS}
    single_mean = float(np.nanmean(table["single_r2"]))
    print("\n===== K–精度代偿（均值 model R²；single 上限 = %.4f）=====" % single_mean)
    for k, v in zip(KS, VARIANTS):
        gap = mean_r2[v] - single_mean
        print(f"  K={k:2d}: mean_R²={mean_r2[v]:.4f}  best_loss={best_loss.get(v, float('nan')):.4f}"
              f"  距single均值={gap:+.4f}")

    # ---- 图1：K–精度代偿曲线 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [k for k, v in zip(KS, VARIANTS) if v in mean_r2]
        ys = [mean_r2[f"experts_k{k}"] for k in ks]
        losses = [best_loss[f"experts_k{k}"] for k in ks]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ks, ys, "o-", color="#55a868", label="mean model R² (12 fields)")
        ax1.axhline(single_mean, ls="--", color="#9e9e9e",
                    label=f"single upper bound ({single_mean:.3f})")
        ax1.set_xlabel("K = number of experts  (≈ effective #fields)")
        ax1.set_ylabel("mean model R²", color="#55a868")
        ax1.set_xticks(ks)
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(ks, losses, "s--", color="#4c72b0", alpha=0.7, label="best_test_loss")
        ax2.set_ylabel("best_test_loss", color="#4c72b0")
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=8)
        ax1.set_title("DNN37: K-expert compensation curve")
        fig.tight_layout()
        fig.savefig(OUT / "experts_curve.png", dpi=130)
        print(f"\n曲线已保存: {OUT / 'experts_curve.png'}")

        # ---- 图2：每字段柱状图（single / K=1 / K=4 / K=12）----
        show = ["single_r2", "experts_k1", "experts_k4", "experts_k12"]
        labels = ["single (upper bound)", "K=1 (multi)", "K=4", "K=12"]
        colors = ["#9e9e9e", "#c44e52", "#dd8452", "#55a868"]
        show = [c for c in show if c in table.columns]
        short = [f.replace("_mw", "").replace("da_as_total_", "as_")[:18]
                 for f in field_order]
        x = np.arange(len(field_order)); w = 0.2
        offs = np.linspace(-1.5 * w, 1.5 * w, len(show))
        fig2, ax = plt.subplots(figsize=(16, 6.5))
        for off, col, lab, color in zip(offs, show, labels, colors):
            ax.bar(x + off, table[col], w, label=lab, color=color)
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("model R²")
        ax.set_title("DNN37: per-field R² across K, with single upper bound")
        ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(OUT / "experts_per_field.png", dpi=130)
        print(f"每字段图已保存: {OUT / 'experts_per_field.png'}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")

    print(f"\n对比表已保存: {out_csv}")


if __name__ == "__main__":
    main()
