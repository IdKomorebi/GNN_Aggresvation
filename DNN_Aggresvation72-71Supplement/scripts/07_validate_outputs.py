#!/usr/bin/env python3
"""Strict completeness and protocol checks for the supplement."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
checks = {}
candidates = json.loads((OUT / "candidates.json").read_text())["candidates"]
expected_candidates = set(candidates)

validation_files = sorted(glob.glob(str(OUT / "validation_mlp_seed*.csv")))
checks["validation_files"] = len(validation_files)
validation = pd.concat([pd.read_csv(p) for p in validation_files], ignore_index=True)
checks["validation_rows"] = len(validation)
checks["validation_seeds"] = sorted(validation.seed.unique().tolist())
checks["validation_expected_rows"] = 3 * 4 * 7

cross_files = sorted(glob.glob(str(OUT / "cross_oracle_*_seed*.csv")))
checks["cross_files"] = len(cross_files)
cross = pd.concat([pd.read_csv(p) for p in cross_files], ignore_index=True)
checks["cross_rows"] = len(cross)
checks["cross_expected_rows"] = len(expected_candidates) * 2 * 3
checks["cross_candidate_match"] = set(cross.candidate) == expected_candidates
checks["cross_arch_seed_pairs"] = sorted({(r.arch, int(r.seed)) for r in cross.itertuples()})

cert_files = sorted(glob.glob(str(OUT / "clean_retrain/*.json")))
cert = pd.DataFrame(json.loads(Path(p).read_text()) for p in cert_files)
checks["clean_retrain_files"] = len(cert_files)
checks["clean_retrain_expected"] = len(expected_candidates) * 3
checks["clean_retrain_candidate_match"] = set(cert.candidate) == expected_candidates
checks["clean_retrain_seeds"] = sorted(cert.seed.unique().tolist())
checks["all_test_after_retrain_selection"] = bool(cert.test_access_stage.isin(
    ["after_model_selection", "after_retrain_model_selection"]
).all())
checks["candidate_selection_reused_test"] = True
checks["split_sizes"] = sorted({(int(r.outer_development_n), int(r.inner_train_n),
                                  int(r.inner_validation_n), int(r.untouched_test_n)) for r in cert.itertuples()})
checks["all_max_epochs_800"] = bool((cert.max_epochs == 800).all())
checks["all_three_seeds_per_candidate"] = bool((cert.groupby("candidate").seed.nunique() == 3).all())

required = [
    ROOT / "RESULTS.md", OUT / "validation_mlp_summary.csv", OUT / "cross_oracle_summary.csv",
    OUT / "clean_retrain_summary.csv", OUT / "oracle_vs_clean_retrain.csv",
    OUT / "training_cap_sensitivity.csv", OUT / "same_budget_comparison.csv",
    ROOT / "figures/mlp_protection_validation_zh.png",
    ROOT / "figures/mlp_search_clean_certificate_zh.png",
    ROOT / "figures/same_budget_clean_comparison_zh.png",
]
checks["missing_required_outputs"] = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
checks["pass"] = bool(
    checks["validation_rows"] == checks["validation_expected_rows"]
    and checks["cross_rows"] == checks["cross_expected_rows"]
    and checks["cross_candidate_match"]
    and checks["clean_retrain_files"] == checks["clean_retrain_expected"]
    and checks["clean_retrain_candidate_match"]
    and checks["all_test_after_retrain_selection"]
    and checks["all_max_epochs_800"]
    and checks["all_three_seeds_per_candidate"]
    and not checks["missing_required_outputs"]
)
(OUT / "validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(checks, ensure_ascii=False, indent=2))
if not checks["pass"]:
    raise SystemExit(1)
