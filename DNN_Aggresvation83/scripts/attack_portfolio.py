# -*- coding: utf-8 -*-
"""把 DNN 重训与结构化攻击者取最大，检查“真值”随攻击族扩展如何变化。"""
from __future__ import annotations

import ast
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from truth import load_correct_truth  # noqa: E402


def main() -> None:
    truth, triples = load_correct_truth()
    estimates = pd.read_csv(OUT / "structured_estimates_all.csv.gz")
    estimates["indices"] = estimates.indices.map(ast.literal_eval)
    pair = estimates[estimates["size"] == 2]
    parent = estimates[estimates["size"] == 3]

    result = truth.copy()
    for model in ("poly2", "arith"):
        pair_map = {
            (row.indices, row.conf): row.est
            for row in pair[pair.model == model].itertuples()
        }
        parent_map = {
            (row.indices, row.conf): row.est
            for row in parent[parent.model == model].itertuples()
        }
        result[f"parent_{model}"] = [
            parent_map[(indices, conf)]
            for indices, conf in zip(result.indices, result.conf)
        ]
        result[f"pair_{model}"] = [
            max(
                pair_map[(tuple(sorted(child)), conf)]
                for child in combinations(indices, 2)
            )
            for indices, conf in zip(result.indices, result.conf)
        ]

    result["pair_portfolio"] = result[
        ["pair_true", "pair_poly2", "pair_arith"]
    ].max(axis=1)
    result["parent_portfolio_raw"] = result[
        ["parent_true", "parent_poly2", "parent_arith"]
    ].max(axis=1)
    # 更大集合可以忽略新增字段，故父值至少等于任何直接子集值。
    result["parent_portfolio"] = result[
        ["parent_portfolio_raw", "pair_portfolio"]
    ].max(axis=1)
    result["syn3_portfolio"] = result.parent_portfolio - result.pair_portfolio
    result.to_csv(OUT / "attack_portfolio_entries.csv.gz", index=False)

    by_triple = (
        result.groupby(["indices", "source", "weight", "split"], as_index=False)
        .agg(
            syn3_dnn=("syn3_true", "max"),
            syn3_portfolio=("syn3_portfolio", "max"),
            parent_gain=("parent_portfolio", lambda x: float("nan")),
        )
        .drop(columns="parent_gain")
    )
    gains = (
        result.groupby("indices", as_index=False)
        .agg(
            parent_gain=(
                "parent_portfolio",
                lambda values: float(
                    np.max(
                        values.to_numpy()
                        - result.loc[values.index, "parent_true"].to_numpy()
                    )
                ),
            ),
            pair_gain=(
                "pair_portfolio",
                lambda values: float(
                    np.max(
                        values.to_numpy()
                        - result.loc[values.index, "pair_true"].to_numpy()
                    )
                ),
            ),
        )
    )
    by_triple = by_triple.merge(gains, on="indices", validate="one_to_one")
    by_triple["strong_dnn"] = by_triple.syn3_dnn > 0.10
    by_triple["strong_portfolio"] = by_triple.syn3_portfolio > 0.10
    by_triple.to_csv(OUT / "attack_portfolio_triples.csv", index=False)

    test = by_triple[by_triple.split == "test"]
    old = set(test.loc[test.strong_dnn, "indices"])
    new = set(test.loc[test.strong_portfolio, "indices"])
    summary = {
        "test_sample_triples": int(len(test)),
        "dnn_strong_sample_count": int(test.strong_dnn.sum()),
        "portfolio_strong_sample_count": int(test.strong_portfolio.sum()),
        "strong_overlap_count": int(len(old & new)),
        "dnn_only_count": int(len(old - new)),
        "portfolio_only_count": int(len(new - old)),
        "syn3_spearman": float(
            spearmanr(test.syn3_dnn, test.syn3_portfolio).correlation
        ),
        "mean_parent_gain": float(
            np.average(test.parent_gain, weights=test.weight)
        ),
        "mean_pair_gain": float(np.average(test.pair_gain, weights=test.weight)),
    }
    pd.Series(summary).to_json(
        OUT / "attack_portfolio_summary.json",
        force_ascii=False,
        indent=2,
    )
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
