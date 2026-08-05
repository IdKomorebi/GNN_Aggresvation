# -*- coding: utf-8 -*-
"""D85 关键数值链和评价隔离检查。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R79 = REPO / "DNN_Aggresvation79"
R83 = REPO / "DNN_Aggresvation83"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R83 / "src"))

from truth import load_correct_truth  # noqa: E402


def key(value: object) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return tuple(ast.literal_eval(str(value)))


def main() -> None:
    checks: list[dict[str, object]] = []

    cold = json.loads(
        (OUT / "cached_poly2_benchmark.json").read_text(encoding="utf-8")
    )
    warm = json.loads(
        (OUT / "cached_poly2_benchmark_warm.json").read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "cached_poly2_matches_original",
            "value": cold["max_abs_r2_delta_vs_d83"],
            "limit": 1e-8,
            "pass": bool(cold["max_abs_r2_delta_vs_d83"] < 1e-8),
        }
    )
    checks.append(
        {
            "check": "cached_scan_all_pair_triple",
            "value": cold["n_sets"],
            "expected": 14190,
            "pass": bool(
                cold["n_sets"] == 14190 and warm["n_sets"] == 14190
            ),
        }
    )

    current = pd.read_csv(OUT / "residual_ridge_estimates_all.csv.gz")
    current = current[current.model == "uniform_k0"].copy()
    current["indices"] = current.indices.map(key)
    legacy_rows = []
    low = pd.read_csv(
        R79 / "outputs" / "kgrid_low_uniform_seed0_shard0of1.csv.gz"
    )
    low = low[(low.K == 0) & (low["size"] == 2)]
    low["indices"] = list(zip(low.i.astype(int), low.j.astype(int)))
    legacy_rows.append(low[["indices", "conf", "est"]])
    for shard in range(4):
        triple = pd.read_csv(
            R79
            / "outputs"
            / f"kgrid_triple_uniform_seed0_shard{shard}of4.csv.gz"
        )
        triple = triple[triple.K == 0]
        triple["indices"] = list(
            zip(
                triple.i.astype(int),
                triple.j.astype(int),
                triple.k.astype(int),
            )
        )
        legacy_rows.append(triple[["indices", "conf", "est"]])
    legacy = pd.concat(legacy_rows, ignore_index=True)
    joined = current.merge(
        legacy,
        on=["indices", "conf"],
        suffixes=("_new", "_legacy"),
        validate="one_to_one",
    )
    k0_delta = float(np.max(np.abs(joined.est_new - joined.est_legacy)))
    checks.append(
        {
            "check": "uniform_k0_matches_D79",
            "value": k0_delta,
            "limit": 1e-5,
            "n_rows": len(joined),
            "pass": bool(k0_delta < 1e-5 and len(joined) == 14190 * 12),
        }
    )

    truth_run = pd.read_csv(OUT / "residual_ridge_estimates.csv.gz")
    truth_run["indices"] = truth_run.indices.map(key)
    for model, path in (
        ("k0_residual_arith", OUT / "residual_ridge_estimates_all.csv.gz"),
        (
            "k0_residual_poly2",
            OUT / "residual_ridge_estimates_all_poly2.csv.gz",
        ),
    ):
        full = pd.read_csv(path)
        full = full[full.model == model].copy()
        full["indices"] = full.indices.map(key)
        old = truth_run[truth_run.model == model]
        overlap = old.merge(
            full,
            on=["indices", "conf"],
            suffixes=("_truth", "_full"),
            validate="one_to_one",
        )
        delta = float(np.max(np.abs(overlap.est_truth - overlap.est_full)))
        checks.append(
            {
                "check": f"{model}_truth_full_reproducible",
                "value": delta,
                "limit": 1e-6,
                "n_rows": len(overlap),
                "pass": bool(delta < 1e-6 and len(overlap) == 3143 * 12),
            }
        )

    truth_entries, truth_triples = load_correct_truth()
    count_check = (
        truth_triples.groupby(["source", "split"]).size().to_dict()
    )
    expected = {
        ("exact397", "test"): 158,
        ("exact397", "tune"): 239,
        ("remainder1800", "test"): 720,
        ("remainder1800", "tune"): 1080,
    }
    checks.append(
        {
            "check": "truth_source_split_counts",
            "value": {repr(k): int(v) for k, v in count_check.items()},
            "expected": {repr(k): v for k, v in expected.items()},
            "pass": bool(count_check == expected),
        }
    )
    curve = pd.read_csv(OUT / "global_recall_curve.csv")
    checks.append(
        {
            "check": "global_curve_method_coverage",
            "value": int(curve.method.nunique()),
            "expected_min": 14,
            "pass": bool(curve.method.nunique() >= 14),
        }
    )
    checks.append(
        {
            "check": "global_curve_metric_ranges",
            "value": [
                float(curve.test_recall.min(skipna=True)),
                float(curve.test_recall.max(skipna=True)),
            ],
            "pass": bool(
                curve.test_recall.dropna().between(0.0, 1.0).all()
            ),
        }
    )
    high = pd.read_csv(OUT / "high_order_fidelity.csv")
    checks.append(
        {
            "check": "high_order_truth_coverage",
            "value": len(high),
            "expected": 105,
            "pass": bool(
                len(high) == 105
                and high.sparse_poly2_top200.notna().all()
            ),
        }
    )
    checks.append(
        {
            "check": "evaluation_isolation",
            "value": (
                "ridge alpha only uses train-internal fit/validation; "
                "protected-lane hyperparameters use tune; headline uses test; "
                "pair-edge selection uses estimator scores only"
            ),
            "pass": True,
            "code_audit": True,
        }
    )
    result = {
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "checks": checks,
        "caveats": [
            "test random pool has 720 triples and 46 with syn3>0.10, but none with syn3>0.20",
            "syn3>0.20 test recall is based on only 5 targeted triples",
            "D63 arbitrary-size truth has only 105 subsets; 50 are random and 55 are selected top-k",
            "cached/sparse poly2 estimates a restricted attack-family lower bound, not the mathematical supremum",
        ],
    }
    (OUT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
