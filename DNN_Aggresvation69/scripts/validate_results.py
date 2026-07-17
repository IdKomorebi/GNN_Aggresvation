#!/usr/bin/env python3
"""Independent reproducibility and completeness checks for DNN69 outputs."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
KGRID = {0, 10, 50, 200}


def main() -> None:
    registry = json.loads((ROOT / "outputs/subsets.json").read_text())
    audit = json.loads((ROOT / "outputs/gcn_audit_subsets.json").read_text())
    checks = {}

    # Universal estimate completeness and schema.
    arch_counts = {}
    bad_k = []
    bad_fields = []
    for arch in ["mlp", "gnn"]:
        files = sorted((ROOT / f"outputs/est/{arch}").glob("*.json"))
        arch_counts[arch] = len(files)
        for path in files:
            record = json.loads(path.read_text())
            sid = record["subset_id"]
            if {int(k) for k in record["by_k"]} != KGRID:
                bad_k.append(f"{arch}/{sid}")
            if record["fields"] != registry[sid]["fields"]:
                bad_fields.append(f"{arch}/{sid}")
    checks["estimate_counts"] = arch_counts
    checks["estimate_expected_per_arch"] = len(registry)
    checks["estimate_k_grid_mismatches"] = bad_k
    checks["estimate_field_mismatches"] = bad_fields

    # D69 MLP path should exactly replay D67 for the shared 990 subsets.
    diffs = []
    shared = 0
    for old_path in (REPO / "DNN_Aggresvation67/outputs/est").glob("*.json"):
        sid = old_path.stem
        new_path = ROOT / f"outputs/est/mlp/{sid}.json"
        if not new_path.exists():
            continue
        old = json.loads(old_path.read_text())
        new = json.loads(new_path.read_text())
        for k in KGRID:
            for conf, value in old["by_k"][str(k)]["per_conf_r2"].items():
                diffs.append(abs(float(value) - float(new["by_k"][str(k)]["per_conf_r2"][conf])))
        shared += 1
    checks["d67_mlp_replay_files"] = shared
    checks["d67_mlp_replay_max_abs_diff"] = max(diffs) if diffs else None

    # Pair synergy analysis must replay the published D67 curve exactly.
    old_syn = pd.read_csv(REPO / "DNN_Aggresvation67/outputs/synergy_ksweep.csv").set_index("K")
    new_syn = pd.read_csv(ROOT / "outputs/synergy_metrics.csv")
    new_syn = new_syn[(new_syn.order == 2) & (new_syn.arch == "mlp")].set_index("K")
    syn_diff = {str(k): float(new_syn.loc[k, "spearman"] - old_syn.loc[k, "spearman_raw"])
                for k in sorted(KGRID)}
    checks["d67_pair_synergy_spearman_diff"] = syn_diff

    # Six accidental cross-source duplicates should be exact retrain repeats.
    duplicate_diffs = {}
    for sid, meta in registry.items():
        if not meta.get("is_duplicate_entry"):
            continue
        canonical = meta["canonical_sid"]
        d60 = json.loads((REPO / f"DNN_Aggresvation60/outputs/retrain/{sid}_dnn_seed0.json").read_text())
        d67 = json.loads((REPO / f"DNN_Aggresvation67/outputs/retrain/{canonical}_dnn_seed0.json").read_text())
        duplicate_diffs[f"{sid}->{canonical}"] = float(d60["mean_r2"] - d67["mean_r2"])
    checks["duplicate_retrain_mean_r2_diff"] = duplicate_diffs

    expected_gcn = set(audit)
    actual_gcn = {Path(p).name.removesuffix("_gcn_seed0.json")
                  for p in glob.glob(str(ROOT / "outputs/retrain_gcn/*_gcn_seed0.json"))}
    checks["gcn_audit_expected"] = len(expected_gcn)
    checks["gcn_audit_completed"] = len(actual_gcn & expected_gcn)
    checks["gcn_audit_missing"] = sorted(expected_gcn - actual_gcn)
    checks["gcn_audit_unexpected"] = sorted(actual_gcn - expected_gcn)

    checks["pass"] = (
        arch_counts == {"mlp": len(registry), "gnn": len(registry)}
        and not bad_k and not bad_fields
        and shared == 990 and max(diffs, default=0.0) == 0.0
        and max((abs(v) for v in syn_diff.values()), default=0.0) == 0.0
        and max((abs(v) for v in duplicate_diffs.values()), default=0.0) == 0.0
        and expected_gcn == actual_gcn
    )
    path = ROOT / "outputs/validation_checks.json"
    path.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
