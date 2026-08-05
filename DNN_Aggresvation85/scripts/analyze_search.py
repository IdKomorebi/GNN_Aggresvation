# -*- coding: utf-8 -*-
"""统一比较全空间三阶筛选、分来源误差与无真值融合方案。"""
from __future__ import annotations

import ast
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R79 = REPO / "DNN_Aggresvation79"
R83 = REPO / "DNN_Aggresvation83"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R83 / "src"))

from truth import load_correct_truth  # noqa: E402


def parse_key(value: object) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    return tuple(ast.literal_eval(str(value)))


def weighted_rank_corr(
    left: np.ndarray, right: np.ndarray, weight: np.ndarray
) -> float:
    left = rankdata(left)
    right = rankdata(right)
    left = left - np.average(left, weights=weight)
    right = right - np.average(right, weights=weight)
    denominator = np.sqrt(
        np.sum(weight * left**2) * np.sum(weight * right**2)
    )
    return float(np.sum(weight * left * right) / denominator)


def estimator_tables(
    path: Path, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame = frame[frame.model == model].copy()
    frame["indices"] = frame.indices.map(parse_key)
    pairs = frame[frame["size"] == 2][["indices", "conf", "est"]].rename(
        columns={"indices": "pair", "est": "pair_est"}
    )
    triples = frame[frame["size"] == 3][
        ["indices", "conf", "est"]
    ].rename(columns={"est": "parent_est"})
    children = []
    for ordinal in range(3):
        child = triples.copy()
        child["pair"] = child.indices.map(
            lambda key, o=ordinal: tuple(combinations(key, 2))[o]
        )
        children.append(child[["indices", "conf", "pair"]])
    child_rows = pd.concat(children, ignore_index=True).merge(
        pairs, on=["pair", "conf"], validate="many_to_one"
    )
    best = (
        child_rows.groupby(["indices", "conf"], as_index=False)
        .pair_est.max()
        .rename(columns={"pair_est": "best_pair_est"})
    )
    per_conf = triples.merge(
        best, on=["indices", "conf"], validate="one_to_one"
    )
    per_conf["syn_est"] = per_conf.parent_est - per_conf.best_pair_est
    scores = (
        per_conf.groupby("indices", as_index=False)
        .agg(s1=("syn_est", "max"), s2=("parent_est", "max"))
        .sort_values("indices")
    )
    assert scores.indices.nunique() == 13244
    return per_conf, scores


def stratified_metrics(
    per_conf: pd.DataFrame,
    scores: pd.DataFrame,
    truth_entries: pd.DataFrame,
    truth_triples: pd.DataFrame,
    method: str,
) -> list[dict[str, object]]:
    joined = truth_entries.merge(
        per_conf,
        on=["indices", "conf"],
        validate="one_to_one",
    )
    triple_join = truth_triples.merge(
        scores, on="indices", validate="one_to_one"
    )
    rows: list[dict[str, object]] = []
    slices = [
        ("test_all", joined.split == "test", triple_join.split == "test"),
        (
            "test_random",
            (joined.split == "test") & (joined.source == "remainder1800"),
            (triple_join.split == "test")
            & (triple_join.source == "remainder1800"),
        ),
        (
            "test_targeted",
            (joined.split == "test") & (joined.source == "exact397"),
            (triple_join.split == "test")
            & (triple_join.source == "exact397"),
        ),
    ]
    for label, entry_mask, triple_mask in slices:
        entry = joined[entry_mask]
        triple = triple_join[triple_mask].copy()
        weight_entry = entry.weight.to_numpy()
        weight_triple = triple.weight.to_numpy()
        keep_n = max(1, round(len(triple) * 0.30))
        keep = set(triple.nlargest(keep_n, "s1").indices)
        strong = triple.syn3_true > 0.10
        rows.append(
            {
                "method": method,
                "slice": label,
                "n_triples": len(triple),
                "n_strong": int(strong.sum()),
                "parent_mae": float(
                    np.average(
                        np.abs(entry.parent_est - entry.parent_true),
                        weights=weight_entry,
                    )
                ),
                "parent_bias": float(
                    np.average(
                        entry.parent_est - entry.parent_true,
                        weights=weight_entry,
                    )
                ),
                "syn_mae": float(
                    np.average(
                        np.abs(entry.syn_est - entry.syn3_true),
                        weights=weight_entry,
                    )
                ),
                "s1_spearman": weighted_rank_corr(
                    triple.s1.to_numpy(),
                    triple.syn3_true.to_numpy(),
                    weight_triple,
                ),
                "within_slice_top30_recall": float(
                    triple.loc[strong & triple.indices.isin(keep), "weight"].sum()
                    / triple.loc[strong, "weight"].sum()
                ),
            }
        )
    return rows


def recall(
    selected: set[tuple[int, ...]],
    truth: pd.DataFrame,
    split: str,
    threshold: float,
    source: str | None = None,
) -> tuple[float, int]:
    subset = truth[(truth.split == split) & (truth.syn3_true > threshold)]
    if source is not None:
        subset = subset[subset.source == source]
    denominator = subset.weight.sum()
    if denominator == 0:
        return np.nan, 0
    value = float(
        subset.loc[subset.indices.isin(selected), "weight"].sum()
        / denominator
    )
    return value, len(subset)


def top_set(score: pd.Series, keep: float) -> set[tuple[int, ...]]:
    n_keep = max(1, round(len(score) * keep))
    return set(score.nlargest(n_keep).index)


def protected_set(
    primary: pd.Series,
    rescue: pd.Series,
    keep: float,
    lane_fraction: float,
) -> set[tuple[int, ...]]:
    n_keep = max(1, round(len(primary) * keep))
    n_rescue = round(n_keep * lane_fraction)
    selected = set(rescue.nlargest(n_rescue).index) if n_rescue else set()
    for key in primary.sort_values(ascending=False).index:
        if len(selected) >= n_keep:
            break
        selected.add(key)
    return selected


def main() -> None:
    truth_entries, truth_triples = load_correct_truth()
    sources = [
        (
            "uniform_k0",
            OUT / "residual_ridge_estimates_all.csv.gz",
            "uniform_k0",
        ),
        (
            "cached_poly2",
            OUT / "cached_poly2_estimates.csv.gz",
            "cached_poly2",
        ),
        (
            "standalone_arith",
            R83 / "outputs" / "structured_estimates_all.csv.gz",
            "arith",
        ),
        (
            "residual_arith",
            OUT / "residual_ridge_estimates_all.csv.gz",
            "k0_residual_arith",
        ),
        (
            "residual_poly2",
            OUT / "residual_ridge_estimates_all_poly2.csv.gz",
            "k0_residual_poly2",
        ),
    ]
    all_scores: dict[str, pd.DataFrame] = {}
    source_rows: list[dict[str, object]] = []
    for name, path, model in sources:
        per_conf, scores = estimator_tables(path, model)
        all_scores[name] = scores
        source_rows.extend(
            stratified_metrics(
                per_conf, scores, truth_entries, truth_triples, name
            )
        )
    pd.DataFrame(source_rows).to_csv(
        OUT / "source_stratified_metrics.csv", index=False
    )

    common = all_scores["uniform_k0"].set_index("indices").rename(
        columns={"s1": "uniform_k0_s1", "s2": "uniform_k0_s2"}
    )
    for name, scores in all_scores.items():
        if name == "uniform_k0":
            continue
        common = common.join(
            scores.set_index("indices").rename(
                columns={"s1": f"{name}_s1", "s2": f"{name}_s2"}
            ),
            validate="one_to_one",
        )
    legacy = pd.read_csv(R79 / "outputs" / "triple_score_kgrid.csv")
    legacy["indices"] = legacy.indices.map(parse_key)
    legacy = legacy.set_index("indices")
    for k in (10, 25, 50):
        common[f"uniform_k{k}_s1"] = legacy[str(k)]

    for name in ("cached_poly2", "standalone_arith", "residual_arith", "residual_poly2"):
        common[f"rank_{name}"] = common[f"{name}_s1"].rank(pct=True)
    common["rankmax_residual_poly2"] = np.maximum(
        common.rank_residual_arith, common.rank_cached_poly2
    )
    common["rankmean_residual_poly2"] = (
        common.rank_residual_arith + common.rank_cached_poly2
    ) / 2.0
    common["rankmax_residual_s2"] = np.maximum(
        common.rank_residual_arith,
        common.uniform_k0_s2.rank(pct=True),
    )

    direct = [
        "uniform_k0_s1",
        "uniform_k10_s1",
        "uniform_k25_s1",
        "uniform_k50_s1",
        "cached_poly2_s1",
        "standalone_arith_s1",
        "residual_poly2_s1",
        "residual_arith_s1",
        "rankmax_residual_poly2",
        "rankmean_residual_poly2",
        "rankmax_residual_s2",
    ]
    curve_rows: list[dict[str, object]] = []
    for method in direct:
        for keep in (0.05, 0.10, 0.20, 0.30):
            selected = top_set(common[method], keep)
            for threshold in (0.10, 0.15, 0.20):
                for source, source_label in (
                    (None, "all"),
                    ("remainder1800", "random"),
                    ("exact397", "targeted"),
                ):
                    value, n_strong = recall(
                        selected, truth_triples, "test", threshold, source
                    )
                    curve_rows.append(
                        {
                            "method": method,
                            "keep": keep,
                            "threshold": threshold,
                            "source": source_label,
                            "test_recall": value,
                            "n_test_strong": n_strong,
                            "tune_selected_lane": np.nan,
                        }
                    )

    lane_specs = [
        ("k0_s1_plus_s2", "uniform_k0_s1", "uniform_k0_s2"),
        (
            "residual_arith_plus_s2",
            "residual_arith_s1",
            "uniform_k0_s2",
        ),
        (
            "residual_arith_plus_poly2",
            "residual_arith_s1",
            "cached_poly2_s1",
        ),
    ]
    lane_rows = []
    for method, primary_name, rescue_name in lane_specs:
        primary = common[primary_name]
        rescue = common[rescue_name]
        for keep in (0.05, 0.10, 0.20, 0.30):
            candidates = []
            for lane in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
                selected = protected_set(primary, rescue, keep, lane)
                tune_recall, n_tune = recall(
                    selected, truth_triples, "tune", 0.10
                )
                candidates.append((tune_recall, -lane, selected, n_tune))
            best_tune, negative_lane, selected, n_tune = max(candidates)
            lane = -negative_lane
            lane_rows.append(
                {
                    "method": method,
                    "keep": keep,
                    "chosen_lane": lane,
                    "tune_recall_gt_010": best_tune,
                    "n_tune_strong": n_tune,
                }
            )
            for threshold in (0.10, 0.15, 0.20):
                for source, source_label in (
                    (None, "all"),
                    ("remainder1800", "random"),
                    ("exact397", "targeted"),
                ):
                    value, n_strong = recall(
                        selected, truth_triples, "test", threshold, source
                    )
                    curve_rows.append(
                        {
                            "method": method,
                            "keep": keep,
                            "threshold": threshold,
                            "source": source_label,
                            "test_recall": value,
                            "n_test_strong": n_strong,
                            "tune_selected_lane": lane,
                        }
                    )
    pd.DataFrame(lane_rows).to_csv(
        OUT / "protected_lane_selection.csv", index=False
    )
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUT / "global_recall_curve.csv", index=False)

    headline = curve[
        (curve.keep == 0.30)
        & (curve.threshold == 0.10)
        & (curve.source == "all")
    ].sort_values("test_recall", ascending=False)
    payload = {
        "truth_test_counts": {
            str(threshold): int(
                (
                    (truth_triples.split == "test")
                    & (truth_triples.syn3_true > threshold)
                ).sum()
            )
            for threshold in (0.10, 0.15, 0.20)
        },
        "top30_recall_gt_010": dict(
            zip(headline.method, headline.test_recall)
        ),
    }
    (OUT / "search_headline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(source_rows).to_string(index=False))
    print("\n全空间 Top-30% / syn3>0.10：")
    print(
        headline[
            ["method", "test_recall", "tune_selected_lane"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
