# -*- coding: utf-8 -*-
"""生成 D85 的统一对比表和结论图。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R63 = REPO / "DNN_Aggresvation63"
R83 = REPO / "DNN_Aggresvation83"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def main() -> None:
    kgrid = pd.read_csv(R83 / "outputs" / "corrected_kgrid_comparison.csv")
    kgrid = kgrid[
        (kgrid.scheme == "uniform")
        & (kgrid["mode"] == "single")
        & kgrid.K.isin([0, 25, 50])
    ]
    rows = []
    for row in kgrid.itertuples():
        rows.append(
            {
                "method": f"uniform_K{row.K}",
                "parent_mae": row.test_parent_mae,
                "syn_mae": row.test_syn_mae,
                "s1_spearman": row.test_s1_spearman,
                "global_top30_recall": row.test_top30_recall,
                "extreme_syn_est_mean": row.extreme_syn_est_mean,
                "extreme_negative_count": row.extreme_negative_count,
                "measured_scan_seconds": (
                    3.7347824722528458 if row.K == 0 else np.nan
                ),
                "cost_note": (
                    "D85 K0 forward"
                    if row.K == 0
                    else "D79 K-grid total to K50: 510s on 4 GPUs"
                ),
            }
        )
    cached = json.loads(
        (OUT / "cached_poly2_benchmark_warm.json").read_text(encoding="utf-8")
    )
    rows.append(
        {
            "method": "cached_poly2",
            "parent_mae": cached["test_parent_mae"],
            "syn_mae": cached["test_syn_mae"],
            "s1_spearman": cached["test_s1_spearman"],
            "global_top30_recall": cached["global_top30_recall"],
            "extreme_syn_est_mean": cached["extreme_syn_est_mean"],
            "extreme_negative_count": cached["extreme_negative_count"],
            "measured_scan_seconds": cached["cache_load_seconds"]
            + cached["scan_seconds"],
            "cost_note": "CPU warm cache, all pairs+triples",
        }
    )
    for suffix, model in (
        ("_all_poly2", "k0_residual_poly2"),
        ("_all", "k0_residual_arith"),
    ):
        summary = pd.read_csv(OUT / f"residual_ridge_summary{suffix}.csv")
        item = summary[summary.model == model].iloc[0]
        timing = json.loads(
            (
                OUT / f"residual_ridge_benchmark{suffix}.json"
            ).read_text(encoding="utf-8")
        )
        rows.append(
            {
                "method": model,
                "parent_mae": item.test_parent_mae,
                "syn_mae": item.test_syn_mae,
                "s1_spearman": item.test_s1_spearman,
                "global_top30_recall": item.global_top30_recall,
                "extreme_syn_est_mean": item.extreme_syn_est_mean,
                "extreme_negative_count": item.extreme_negative_count,
                "measured_scan_seconds": timing["total_seconds"],
                "cost_note": "GPU K0 forward + CPU closed-form residual",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "low_order_method_comparison.csv", index=False)

    curve = pd.read_csv(OUT / "global_recall_curve.csv")
    curve = curve[
        (curve.threshold == 0.10)
        & (curve.source == "all")
        & curve.method.isin(
            [
                "uniform_k0_s1",
                "uniform_k25_s1",
                "uniform_k50_s1",
                "cached_poly2_s1",
                "residual_poly2_s1",
                "residual_arith_s1",
                "rankmean_residual_poly2",
            ]
        )
    ]
    labels = {
        "uniform_k0_s1": "Universal K0",
        "uniform_k25_s1": "Universal K25",
        "uniform_k50_s1": "Universal K50",
        "cached_poly2_s1": "Cached poly2",
        "residual_poly2_s1": "K0 + poly2 residual",
        "residual_arith_s1": "K0 + arithmetic residual",
        "rankmean_residual_poly2": "Residual/poly2 rank fusion",
    }

    high = pd.read_csv(OUT / "high_order_fidelity_summary.csv")
    bands = ["1-4", "5-8", "9-16", "17-32", "33-44"]
    sparse = high[
        (high.model == "sparse_poly2_top200") & high["slice"].isin(bands)
    ].set_index("slice")
    d63 = json.loads(
        (R63 / "outputs" / "fidelity_report.json").read_text(encoding="utf-8")
    )
    mlp_mae = [d63["by_band"]["mlp"][band]["mae"] for band in bands]

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for method, group in curve.groupby("method"):
        axes[0].plot(
            100 * group.keep,
            group.test_recall,
            marker="o",
            label=labels[method],
        )
    axes[0].set_xlabel("Candidate retention (%)")
    axes[0].set_ylabel("Recall for true syn3 > 0.10")
    axes[0].set_title("Full-space triple search")
    axes[0].set_ylim(0.3, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    cost = comparison[comparison.method.isin(
        ["uniform_K0", "cached_poly2", "k0_residual_poly2", "k0_residual_arith"]
    )]
    axes[1].scatter(
        cost.measured_scan_seconds,
        cost.global_top30_recall,
        s=65,
    )
    for row in cost.itertuples():
        axes[1].annotate(
            row.method.replace("k0_residual_", "residual_"),
            (row.measured_scan_seconds, row.global_top30_recall),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Measured all-pair/triple time (s, log)")
    axes[1].set_ylabel("Top-30% recall")
    axes[1].set_title("Accuracy-cost frontier")
    axes[1].grid(alpha=0.25)

    x = np.arange(len(bands))
    width = 0.38
    axes[2].bar(x - width / 2, mlp_mae, width, label="Universal MLP K0")
    axes[2].bar(
        x + width / 2,
        [sparse.loc[band, "mae"] for band in bands],
        width,
        label="Sparse cached poly2",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(bands)
    axes[2].set_xlabel("Subset size")
    axes[2].set_ylabel("MAE vs retrained attack")
    axes[2].set_title("Arbitrary-size fidelity (105 points)")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "d85_search_and_fidelity.png", dpi=180)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
