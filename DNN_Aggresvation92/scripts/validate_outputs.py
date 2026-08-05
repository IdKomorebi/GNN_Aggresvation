#!/usr/bin/env python3
"""Reproducibility and leakage checks for DNN92."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def main() -> None:
    design = json.loads((ROOT / "outputs/design.json").read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in design["tasks"]}
    split = design["splits"]
    checks = {}

    shard_paths = sorted((ROOT / "outputs/trajectories").glob("shard*of3.npz"))
    arrays = [np.load(path) for path in shard_paths]
    all_ids = np.concatenate([z["task_ids"] for z in arrays]).astype(str)
    checks["trajectory_shards"] = {
        "passed": len(shard_paths) == 3 and len(all_ids) == len(set(all_ids)) == len(tasks),
        "n_shards": len(shard_paths),
        "n_tasks": len(all_ids),
        "n_unique": len(set(all_ids)),
    }

    basis = set(split["basis_train"])
    val_targets = set(split["validation_targets"])
    audit_targets = set(split["audit_targets"])
    forbidden_parents = set()
    for task_id in split["validation_triples"] + split["audit_triples"]:
        indices = tasks[task_id]["indices"]
        for parent in combinations(indices, 2):
            forbidden_parents.add(
                "q_" + "_".join(f"{x:02d}" for x in sorted(parent))
            )
    exact_leak = basis & (val_targets | audit_targets | forbidden_parents)
    checks["exact_task_leakage"] = {
        "passed": not exact_leak,
        "n_basis": len(basis),
        "n_forbidden_overlap": len(exact_leak),
        "overlap": sorted(exact_leak),
    }

    estimates = pd.read_csv(ROOT / "outputs/all_estimates.csv.gz")
    checks["finite_estimates"] = {
        "passed": bool(
            np.isfinite(estimates[["est", "truth"]].to_numpy()).all()
            and (estimates["est"] >= 0).all()
        ),
        "n_rows": len(estimates),
        "n_methods": int(estimates[["method", "rank"]].drop_duplicates().shape[0]),
    }

    # Exact reproduction of D69 K0 for every evaluation task/confidential output.
    k0 = estimates[
        (estimates["method"] == "full_K0") & (estimates["rank"] == 0)
    ].drop_duplicates(["task_id", "conf"])
    differences = []
    for row in k0.itertuples(index=False):
        old = json.loads(
            (
                REPO
                / "DNN_Aggresvation69/outputs/est/mlp"
                / f"{tasks[row.task_id]['sid69']}.json"
            ).read_text(encoding="utf-8")
        )
        reference = float(old["by_k"]["0"]["per_conf_r2"][row.conf])
        differences.append(abs(float(row.est) - reference))
    checks["d69_k0_reproduction"] = {
        "passed": max(differences, default=np.inf) < 2e-6,
        "n_compared": len(differences),
        "max_abs_difference": max(differences, default=np.nan),
    }

    diagnostics = pd.read_csv(ROOT / "outputs/subspace_diagnostics.csv")
    monotonic = True
    for _, frame in diagnostics.groupby("split"):
        values = frame.sort_values("rank")["mean"].to_numpy()
        monotonic &= bool(np.all(np.diff(values) >= -1e-7))
    checks["subspace_energy_monotonic"] = {
        "passed": monotonic,
        "max_rank": int(diagnostics["rank"].max()),
    }

    selections = json.loads(
        (ROOT / "outputs/validation_selections.json").read_text(encoding="utf-8")
    )
    all_methods = set(zip(estimates["method"], estimates["rank"]))
    selected_present = all(
        (record["method"], record["rank"]) in all_methods
        for record in selections.values()
    )
    checks["validation_selected_methods_exist"] = {
        "passed": selected_present,
        "selections": selections,
    }

    lora = estimates[estimates["method"].str.startswith("lora_")]
    lora_coverage = lora.groupby(["method", "rank"])["task_id"].nunique()
    checks["lora_task_coverage"] = {
        "passed": bool((lora_coverage == 481).all()),
        "min_tasks": int(lora_coverage.min()),
        "max_tasks": int(lora_coverage.max()),
        "n_configurations": int(len(lora_coverage)),
    }

    required = [
        "subspace_spectrum.csv",
        "subspace_diagnostics.csv",
        "all_direct_metrics.csv",
        "all_synergy_metrics.csv",
        "main_comparison.csv",
        "subspace_summary.png",
        "cost_summary.csv",
    ]
    missing = [name for name in required if not (ROOT / "outputs" / name).exists()]
    checks["required_outputs"] = {"passed": not missing, "missing": missing}
    passed = all(record["passed"] for record in checks.values())
    report = {"passed": passed, "checks": checks}
    (ROOT / "outputs/validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
