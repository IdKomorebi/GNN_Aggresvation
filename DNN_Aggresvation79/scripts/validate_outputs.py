# -*- coding: utf-8 -*-
"""79 号最终输出的可复现一致性检查。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "outputs"
KGRID = {0, 1, 5, 10, 25, 50}
N_ALL = 13244


def max_difference(left: pd.DataFrame, right: pd.DataFrame, keys, left_value, right_value):
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    return len(merged), float((merged[left_value] - merged[right_value]).abs().max())


def main() -> None:
    report = {"status": "ready", "checks": {}}
    low = pd.read_csv(next(OUT.glob("kgrid_low_uniform_seed0_shard*.csv.gz")))
    triple = pd.concat(
        [pd.read_csv(path) for path in sorted(OUT.glob("kgrid_triple_uniform_seed0_shard*.csv.gz"))],
        ignore_index=True,
    )
    assert len(low) == 990 * 6 * 12
    assert len(triple) == N_ALL * 6 * 12
    assert set(low.K.unique()) == KGRID
    assert set(triple.K.unique()) == KGRID
    assert triple.groupby(["i", "j", "k"]).ngroups == N_ALL
    report["checks"]["raw_grid"] = {
        "low_rows": len(low),
        "triple_rows": len(triple),
        "unique_triples": triple.groupby(["i", "j", "k"]).ngroups,
        "kgrid": sorted(KGRID),
    }

    # K0 新旧逐项一致，证明79号没有悄悄改变oracle查询口径。
    old_tri0 = pd.read_csv(REPO / "DNN_Aggresvation77" / "outputs" / "triples_k0.csv")
    new_tri0 = triple[triple.K == 0]
    n, diff = max_difference(
        new_tri0, old_tri0, ["i", "j", "k", "conf"], "est_x", "est_y"
    )
    assert n == N_ALL * 12 and diff < 1e-5

    old_low0 = pd.read_csv(REPO / "DNN_Aggresvation77" / "outputs" / "lowfour_k0.csv")
    new_low0 = low[low.K == 0].copy()
    new_low0["key"] = [
        f"{row.i:02d}" if row.size == 1 else f"{row.i:02d}_{row.j:02d}"
        for row in new_low0.itertuples()
    ]
    n_low, diff_low = max_difference(
        new_low0, old_low0, ["key", "size", "conf"], "est_x", "est_y"
    )
    assert n_low == 990 * 12 and diff_low < 1e-5
    report["checks"]["k0_matches_77"] = {
        "triple_entries": n,
        "triple_max_abs_diff": diff,
        "low_entries": n_low,
        "low_max_abs_diff": diff_low,
    }

    # K25与77号第二层重叠候选逐项一致。
    old_tri25 = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(
                (REPO / "DNN_Aggresvation77" / "outputs").glob(
                    "triples_kstar_shard*.csv"
                )
            )
        ],
        ignore_index=True,
    )
    new_tri25 = triple[triple.K == 25]
    n25, diff25 = max_difference(
        new_tri25, old_tri25, ["i", "j", "k", "conf"], "est_x", "est_y"
    )
    assert n25 == len(old_tri25) and diff25 < 1e-5
    report["checks"]["k25_matches_77_overlap"] = {
        "entries": n25,
        "max_abs_diff": diff25,
    }

    truth = pd.read_csv(OUT / "truth_design_split.csv")
    assert abs(truth.weight.sum() - N_ALL) < 1e-6
    assert set(truth.split) == {"tune", "test"}
    report["checks"]["truth_design"] = {
        "sample_triples": len(truth),
        "weighted_population": float(truth.weight.sum()),
        "strong_sample": int(truth.strong.astype(bool).sum()),
    }

    metrics = pd.read_csv(OUT / "kgrid_triple_metrics.csv")
    ranks = pd.read_csv(OUT / "kgrid_recall_precision.csv")
    assert not metrics.isna().any().any()
    assert not ranks.isna().any().any()
    assert metrics[["roc_auc", "average_precision"]].to_numpy().min() >= 0
    assert metrics[["roc_auc", "average_precision"]].to_numpy().max() <= 1
    assert ranks[["recall", "precision", "f1"]].to_numpy().min() >= 0
    assert ranks[["recall", "precision", "f1"]].to_numpy().max() <= 1
    report["checks"]["metric_ranges"] = "passed"

    staged = pd.read_csv(OUT / "staged_rescue_top30.csv")
    expected = staged[
        (staged.first_k == 10)
        & (staged.first_retention == 0.60)
        & (staged.variant == "best_checkpoint_rank")
    ].iloc[0]
    tiers = pd.read_csv(OUT / "final_fidelity_tiers.csv")
    actual = tiers[tiers.tier == "recommended_hierarchy"].iloc[0]
    assert abs(expected.all_recall - actual.population_recall) < 1e-12
    assert abs(expected.test_recall - actual.test_recall) < 1e-12
    assert abs(expected.avg_updates_continuation - actual.avg_updates_continuation) < 1e-12
    report["checks"]["recommended_hierarchy"] = {
        "population_recall": float(actual.population_recall),
        "test_recall": float(actual.test_recall),
        "avg_updates_continuation": float(actual.avg_updates_continuation),
    }

    (OUT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
