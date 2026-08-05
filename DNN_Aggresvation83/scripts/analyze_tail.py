# -*- coding: utf-8 -*-
"""用 D82 的独立 tune/test 真值协议评价 tail-aware oracle。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R75 = REPO / "DNN_Aggresvation75"
R82 = REPO / "DNN_Aggresvation82"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R82 / "src"))
sys.path.insert(0, str(ROOT / "src"))

from truth import load_correct_truth  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "d82_analyze", R82 / "scripts" / "analyze_k0.py"
)
assert spec and spec.loader
d82 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d82)


def add_scheme(
    rows: list[dict[str, object]],
    scheme: str,
    seed: int,
    frame_path: Path,
    checkpoint_path: Path,
) -> None:
    frame = d82.load_estimate(frame_path)
    truth_entries, truth_triples = add_scheme.truth  # type: ignore[attr-defined]
    rows.append(
        d82.evaluate(
            scheme,
            seed,
            "single",
            d82.make_entries(frame, frame),
            truth_entries,
            truth_triples,
            checkpoint_path,
        )
    )


def main() -> None:
    add_scheme.truth = load_correct_truth()  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []

    # 已有基线：普通 uniform 及 D82 中“同 mask 32 样本”的 ERM。
    for scheme, seeds in (("uniform", (0, 1)), ("local234_group32", (0,))):
        for seed in seeds:
            add_scheme(
                rows,
                scheme,
                seed,
                R82 / "outputs" / f"k0_{scheme}_seed{seed}.parquet",
                (
                    R75 / "outputs" / f"oracle_{scheme}_seed{seed}.pt"
                    if scheme == "uniform"
                    else R82 / "outputs" / f"oracle_{scheme}_seed{seed}.pt"
                ),
            )

    for path in sorted(OUT.glob("k0_*_seed*.parquet")):
        scheme, seed = d82.scheme_seed(path)
        add_scheme(
            rows,
            scheme,
            seed,
            path,
            OUT / f"oracle_{scheme}_seed{seed}.pt",
        )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "tail_metrics_by_seed.csv", index=False)
    columns = [
        "scheme",
        "seed",
        "test_parent_mae",
        "test_parent_bias",
        "test_syn_mae",
        "test_s1_spearman",
        "test_top30_recall",
        "test_top30_precision",
        "test_top30_f1",
        "extreme_parent_error_mean",
        "extreme_pair_error_mean",
        "extreme_syn_est_mean",
        "extreme_negative_count",
        "extreme_top30_hit",
        "r2_full",
        "epochs_run",
    ]
    summary = detail[columns].sort_values(
        ["extreme_syn_est_mean", "test_s1_spearman"], ascending=False
    )
    summary.to_csv(OUT / "tail_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
