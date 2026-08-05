# -*- coding: utf-8 -*-
"""在 D63 的 105 个任意大小重训练点上评估稀疏缓存 poly2。"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R59 = REPO / "DNN_Aggresvation59"
R60 = REPO / "DNN_Aggresvation60"
R63 = REPO / "DNN_Aggresvation63"
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "cached_poly2", ROOT / "scripts" / "cached_poly2.py"
)
assert spec and spec.loader
cached = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cached)

BANDS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]


def band_of(size: int) -> str:
    for low, high in BANDS:
        if low <= size <= high:
            return f"{low}-{high}"
    raise ValueError(size)


def load_truth(general: list[str]) -> pd.DataFrame:
    name_to_index = {name: index for index, name in enumerate(general)}
    rows = []
    subsets = json.loads(
        (R60 / "outputs" / "subsets.json").read_text(encoding="utf-8")
    )
    random_values = pd.read_csv(
        R60 / "outputs" / "random_eval.csv"
    ).set_index("subset_id")
    for subset_id, info in subsets.items():
        if info["group"] != "random_eval":
            continue
        key = tuple(sorted(name_to_index[field] for field in info["fields"]))
        rows.append(
            {
                "subset_id": subset_id,
                "indices": key,
                "size": len(key),
                "source": "60_random",
                "v_truth": float(random_values.loc[subset_id, "v_best"]),
            }
        )
    retrain = pd.read_csv(R59 / "outputs" / "retrain" / "all_retrain.csv")
    best = retrain.groupby(["ranking", "k"]).mean_r2.max()
    orders = {
        ranking: json.loads(
            (
                R59 / "outputs" / "retrain" / f"ranking_{ranking}.json"
            ).read_text(encoding="utf-8")
        )["order_fields"]
        for ranking in retrain.ranking.unique()
    }
    for (ranking, size), value in best.items():
        key = tuple(
            sorted(name_to_index[field] for field in orders[ranking][: int(size)])
        )
        rows.append(
            {
                "subset_id": f"{ranking}_k{size}",
                "indices": key,
                "size": int(size),
                "source": "59_topk",
                "v_truth": float(value),
            }
        )
    truth = pd.DataFrame(rows)
    truth["band"] = truth["size"].map(band_of)
    assert len(truth) == 105
    return truth


def columns_for(
    key: tuple[int, ...],
    n_general: int,
    pair_position: dict[tuple[int, int], int],
    selected_edges: set[tuple[int, int]] | None,
) -> list[int]:
    columns = list(key) + [n_general + index for index in key]
    pairs = combinations(key, 2)
    if selected_edges is not None:
        pairs = (pair for pair in pairs if pair in selected_edges)
    columns.extend(
        2 * n_general + pair_position[pair]
        for pair in pairs
    )
    return columns


def metrics(
    frame: pd.DataFrame, model: str, mask: pd.Series, label: str
) -> dict[str, object]:
    selected = frame[mask & frame[model].notna()]
    error = selected[model] - selected.v_truth
    return {
        "model": model,
        "slice": label,
        "n": len(selected),
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "spearman": float(
            spearmanr(selected[model], selected.v_truth).statistic
        ),
    }


def main() -> None:
    with np.load(OUT / "poly2_moments.npz") as payload:
        moments = {name: payload[name] for name in payload.files}
    data = cached.load_data()
    n_general = int(data["n_general"])
    general = data["general"]
    all_pairs = list(combinations(range(n_general), 2))
    pair_position = {pair: index for index, pair in enumerate(all_pairs)}
    truth = load_truth(general)

    single = {}
    for index in range(n_general):
        columns = [index, n_general + index]
        single[index] = cached.query(columns, moments)
    pair_frame = pd.read_csv(OUT / "cached_poly2_estimates.csv.gz")
    pair_frame = pair_frame[pair_frame["size"] == 2].copy()
    pair_frame["indices"] = pair_frame.indices.map(
        lambda value: tuple(int(x.strip()) for x in value.strip("()").split(","))
    )
    conf_names = list(data["confidential"])
    conf_position = {name: index for index, name in enumerate(conf_names)}
    edge_score = {}
    for key, group in pair_frame.groupby("indices"):
        values = np.zeros(len(conf_names))
        for row in group.itertuples():
            values[conf_position[row.conf]] = row.est
        edge_score[key] = float(
            np.max(values - np.maximum(single[key[0]], single[key[1]]))
        )
    ordered_edges = sorted(edge_score, key=edge_score.get, reverse=True)
    edge_table = pd.DataFrame(
        [{"edge": repr(edge), "score": edge_score[edge]} for edge in ordered_edges]
    )
    edge_table.to_csv(OUT / "selected_pair_edges.csv", index=False)

    variants: dict[str, set[tuple[int, int]] | None] = {
        "raw_square": set(),
        "sparse_poly2_top50": set(ordered_edges[:50]),
        "sparse_poly2_top100": set(ordered_edges[:100]),
        "sparse_poly2_top200": set(ordered_edges[:200]),
        "dense_poly2": None,
    }
    timing_rows = []
    result = truth.copy()
    for model, edges in variants.items():
        predictions = []
        for row in truth.itertuples():
            if model == "dense_poly2" and row.size > 16:
                predictions.append(np.nan)
                continue
            columns = columns_for(
                row.indices, n_general, pair_position, edges
            )
            started = time.perf_counter()
            prediction = float(cached.query(columns, moments).mean())
            elapsed = time.perf_counter() - started
            predictions.append(prediction)
            timing_rows.append(
                {
                    "model": model,
                    "subset_id": row.subset_id,
                    "size": row.size,
                    "n_columns": len(columns),
                    "seconds": elapsed,
                }
            )
        result[model] = predictions
    result.to_csv(OUT / "high_order_fidelity.csv", index=False)
    timings = pd.DataFrame(timing_rows)
    timings.to_csv(OUT / "high_order_query_timings.csv", index=False)

    summary_rows = []
    for model in variants:
        available = result[model].notna()
        summary_rows.append(
            metrics(
                result,
                model,
                pd.Series(True, index=result.index),
                "overall_available",
            )
        )
        for source in ("60_random", "59_topk"):
            mask = result.source == source
            summary_rows.append(metrics(result, model, mask, source))
        for low, high in BANDS:
            band = f"{low}-{high}"
            mask = result.band == band
            if (mask & available).sum() >= 3:
                summary_rows.append(metrics(result, model, mask, band))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "high_order_fidelity_summary.csv", index=False)
    timing_summary = (
        timings.groupby(["model", "size"], as_index=False)
        .agg(
            n=("seconds", "size"),
            mean_columns=("n_columns", "mean"),
            median_ms=("seconds", lambda value: 1000 * value.median()),
            p95_ms=("seconds", lambda value: 1000 * value.quantile(0.95)),
        )
    )
    timing_summary.to_csv(OUT / "high_order_timing_summary.csv", index=False)

    rng = np.random.default_rng(20260724)
    benchmark_rows = []
    for model, edges in variants.items():
        for size in (4, 5, 6, 8, 10, 12, 16, 24, 32, 44):
            if model == "dense_poly2" and size > 16:
                continue
            for _ in range(50):
                key = tuple(
                    sorted(rng.choice(n_general, size=size, replace=False))
                )
                columns = columns_for(
                    key, n_general, pair_position, edges
                )
                started = time.perf_counter()
                cached.query(columns, moments)
                benchmark_rows.append(
                    {
                        "model": model,
                        "size": size,
                        "n_columns": len(columns),
                        "seconds": time.perf_counter() - started,
                    }
                )
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark_summary = (
        benchmark.groupby(["model", "size"], as_index=False)
        .agg(
            n=("seconds", "size"),
            mean_columns=("n_columns", "mean"),
            median_ms=("seconds", lambda value: 1000 * value.median()),
            p95_ms=("seconds", lambda value: 1000 * value.quantile(0.95)),
        )
    )
    benchmark_summary["n_combinations"] = benchmark_summary["size"].map(
        lambda size: math.comb(n_general, int(size))
    )
    benchmark_summary["enumerate_hours_at_median"] = (
        benchmark_summary.n_combinations
        * benchmark_summary.median_ms
        / 1000.0
        / 3600.0
    )
    benchmark_summary.to_csv(
        OUT / "high_order_enumeration_benchmark.csv", index=False
    )
    print(summary.to_string(index=False))
    print("\n查询时延：")
    print(timing_summary.to_string(index=False))
    print("\n随机查询与全枚举外推：")
    print(benchmark_summary.to_string(index=False))


if __name__ == "__main__":
    main()
