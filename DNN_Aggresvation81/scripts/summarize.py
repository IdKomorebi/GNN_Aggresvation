# -*- coding: utf-8 -*-
"""只用tune选择协议，再在test上做一次最终比较和配对bootstrap。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    N_ALL,
    best_checkpoint_score,
    fixed_force_fill,
    load_scores,
    load_truth,
    score_series,
    top_n,
)
from runlog import Timer, log  # noqa: E402

BOOTSTRAP_SEED = 8101
N_BOOT = 5000


def f1(recall, precision):
    return 2 * recall * precision / (recall + precision)


def reconstruct_baseline(scores):
    s1_k10 = score_series(scores, 10, "s1_syn3")
    active = top_n(s1_k10, round(N_ALL * 0.60))
    best = best_checkpoint_score(scores, [0, 10, 25], active)
    return top_n(best, round(N_ALL * 0.30))


def reconstruct_selected(scores, row):
    first_k = int(row.first_k)
    s1 = score_series(scores, first_k, "s1_syn3")
    s2_column = (
        "s2_parent_any" if row.first_s2 == "any" else "s2_parent_aligned"
    )
    s2 = score_series(scores, first_k, s2_column)
    baseline_active = top_n(s1, round(N_ALL * row.first_retention))
    active = fixed_force_fill(
        s1,
        s2,
        row.first_retention,
        row.first_parameter,
    )
    rescued = active - baseline_active
    protect_n = min(
        round(N_ALL * row.protect_fraction),
        len(rescued),
    )
    protected = top_n(s2, protect_n, rescued)
    best = best_checkpoint_score(
        scores, [0, first_k, int(row.final_k)], active
    )
    remainder = best.drop(index=list(protected), errors="ignore")
    final_n = round(N_ALL * row.final_retention)
    return protected | top_n(remainder, final_n - len(protected))


def bootstrap_test(truth, baseline_keep, selected_keep):
    test = truth[truth.split == "test"].copy().reset_index(drop=True)
    test["base"] = test.indices.isin(baseline_keep)
    test["selected"] = test.indices.isin(selected_keep)
    strata = [
        group.index.to_numpy()
        for _, group in test.groupby(["source", "strong"], sort=False)
    ]
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    rows = []
    for _ in range(N_BOOT):
        sample_idx = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in strata]
        )
        sample = test.iloc[sample_idx]
        weight = sample.weight.to_numpy(float)
        strong = sample.strong.to_numpy(bool)
        base = sample.base.to_numpy(bool)
        selected = sample.selected.to_numpy(bool)

        def metrics(hit):
            recall = weight[strong & hit].sum() / weight[strong].sum()
            precision = weight[strong & hit].sum() / weight[hit].sum()
            return recall, precision

        br, bp = metrics(base)
        sr, sp = metrics(selected)
        rows.append(
            {
                "delta_recall": sr - br,
                "delta_precision": sp - bp,
                "baseline_recall": br,
                "selected_recall": sr,
            }
        )
    boot = pd.DataFrame(rows)
    return boot


def main() -> None:
    log("SUMMARY", "START", note="tune选协议；test只作一次最终验证")
    with Timer() as timer:
        staged = pd.read_csv(ROOT / "outputs" / "staged_sweep.csv")
        protected = pd.read_csv(ROOT / "outputs" / "protected_lane_sweep.csv")
        scores = load_scores()
        truth = load_truth()
        estimated_total_strong = float(
            truth.loc[truth.strong, "weight"].sum()
        )

        baseline = staged[
            (staged.first_k == 10)
            & (staged.first_retention == 0.60)
            & (staged.first_family == "s1_only")
            & (staged.final_k == 25)
            & (staged.final_retention == 0.30)
            & (staged.final_family == "best_s1")
        ].iloc[0].copy()
        baseline["tune_f1"] = f1(
            baseline.tune_recall, baseline.tune_precision
        )
        baseline["test_f1"] = f1(
            baseline.test_recall, baseline.test_precision
        )
        baseline["all_precision"] = (
            estimated_total_strong * baseline.all_recall
            / round(N_ALL * 0.30)
        )

        eligible = protected[
            (protected.final_retention == 0.30)
            & (protected.avg_updates_continuation <= 19.0 + 1e-12)
        ].copy()
        eligible["tune_f1"] = f1(
            eligible.tune_recall, eligible.tune_precision
        )
        eligible["test_f1"] = f1(
            eligible.test_recall, eligible.test_precision
        )
        # 唯一选择规则：最大tune F1；test完全不参与排序。
        selected = eligible.loc[eligible.tune_f1.idxmax()].copy()
        recall_only = eligible.loc[eligible.tune_recall.idxmax()].copy()
        for row in (selected, recall_only):
            row["all_precision"] = (
                estimated_total_strong * row.all_recall
                / round(N_ALL * 0.30)
            )

        baseline_keep = reconstruct_baseline(scores)
        selected_keep = reconstruct_selected(scores, selected)
        assert len(baseline_keep) == len(selected_keep) == round(N_ALL * 0.30)

        boot = bootstrap_test(truth, baseline_keep, selected_keep)
        boot.to_csv(ROOT / "outputs" / "bootstrap_test_delta.csv", index=False)
        ci = {
            "n_bootstrap": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "delta_recall_mean": float(boot.delta_recall.mean()),
            "delta_recall_ci95": [
                float(boot.delta_recall.quantile(0.025)),
                float(boot.delta_recall.quantile(0.975)),
            ],
            "delta_precision_mean": float(boot.delta_precision.mean()),
            "delta_precision_ci95": [
                float(boot.delta_precision.quantile(0.025)),
                float(boot.delta_precision.quantile(0.975)),
            ],
        }
        (ROOT / "outputs" / "bootstrap_summary.json").write_text(
            json.dumps(ci, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        keep_columns = [
            "avg_updates_continuation",
            "tune_recall",
            "test_recall",
            "all_recall",
            "tune_precision",
            "test_precision",
            "all_precision",
            "tune_f1",
            "test_f1",
            "tune_b10_12_recall",
            "test_b10_12_recall",
            "all_b10_12_recall",
            "tune_b12_15_recall",
            "test_b12_15_recall",
            "all_b12_15_recall",
            "tune_b15_20_recall",
            "test_b15_20_recall",
            "all_b15_20_recall",
            "tune_b_gt20_recall",
            "test_b_gt20_recall",
            "all_b_gt20_recall",
            "tune_b_gt20_raw_hit",
            "test_b_gt20_raw_hit",
            "all_b_gt20_raw_hit",
        ]
        summary_rows = []
        for name, row in (
            ("D79_S1_baseline", baseline),
            ("D81_tune_F1_selected", selected),
            ("D81_tune_recall_only", recall_only),
        ):
            record = {"protocol": name}
            for key in keep_columns:
                record[key] = row.get(key, np.nan)
            for key in (
                "first_k",
                "first_retention",
                "first_family",
                "first_s2",
                "first_parameter",
                "final_k",
                "final_retention",
                "protect_fraction",
                "n_rescued_stage1",
                "n_protected",
            ):
                record[key] = row.get(key, np.nan)
            summary_rows.append(record)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(
            ROOT / "outputs" / "selected_protocols.csv", index=False
        )

    log(
        "SUMMARY",
        "DONE",
        note="主方案由tune F1选中；完成test配对bootstrap",
        elapsed_s=timer.elapsed,
        selected_first_k=int(selected.first_k),
        selected_first_s2=selected.first_s2,
        selected_first_parameter=float(selected.first_parameter),
        selected_protect=float(selected.protect_fraction),
    )
    print(summary.to_string(index=False))
    print(json.dumps(ci, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
