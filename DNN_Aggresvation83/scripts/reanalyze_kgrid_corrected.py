# -*- coding: utf-8 -*-
"""用修正真值重算 D82 的 K-grid；不改写 D82 旧输出。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R82 = REPO / "DNN_Aggresvation82"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R82 / "scripts"))
sys.path.insert(0, str(R82 / "src"))
sys.path.insert(0, str(ROOT / "src"))

from truth import load_correct_truth  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "d82_kgrid", R82 / "scripts" / "analyze_kgrid.py"
)
assert spec and spec.loader
d82 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d82)


def main() -> None:
    truth_entries, truth_triples = load_correct_truth()
    frames = []
    uniform_low, uniform_triple = d82.load_grid("uniform")
    uniform_entries = d82.attach_children(uniform_triple, uniform_low)
    metrics, _ = d82.evaluate_mode(
        "uniform", "single", uniform_entries, truth_entries, truth_triples
    )
    frames.append(metrics)
    for scheme in ("local234_uniform10", "triple50_pair40_uniform10"):
        low, triple = d82.load_grid(scheme)
        for mode, child in (
            ("single", low),
            ("hybrid_uniform_pairs", uniform_low),
        ):
            entries = d82.attach_children(triple, child)
            metrics, _ = d82.evaluate_mode(
                scheme, mode, entries, truth_entries, truth_triples
            )
            frames.append(metrics)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "corrected_kgrid_comparison.csv", index=False)
    show = result[
        (result.scheme == "uniform") & (result["mode"] == "single")
    ][
        [
            "K",
            "test_parent_mae",
            "test_parent_bias",
            "test_syn_mae",
            "test_s1_spearman",
            "test_top30_recall",
            "extreme_syn_est_mean",
        ]
    ]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
