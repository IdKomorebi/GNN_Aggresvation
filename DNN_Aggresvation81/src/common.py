# -*- coding: utf-8 -*-
"""81号筛选与评价的公共函数。"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
N_ALL = 13244
KGRID = (0, 1, 5, 10, 25, 50)
STRONG = 0.10
STRENGTH_BANDS = (
    (0.10, 0.12, "b10_12"),
    (0.12, 0.15, "b12_15"),
    (0.15, 0.20, "b15_20"),
    (0.20, 1.01, "b_gt20"),
)


def load_scores() -> pd.DataFrame:
    data = pd.read_parquet(OUT / "score_cube.parquet")
    data["indices"] = data["indices"].map(tuple)
    return data


def load_truth() -> pd.DataFrame:
    truth = pd.read_csv(
        ROOT.parent / "DNN_Aggresvation79" / "outputs" / "truth_design_split.csv"
    )
    truth["indices"] = truth.indices.map(ast.literal_eval)
    truth["strong"] = truth.strong.astype(bool)
    return truth


def score_series(scores: pd.DataFrame, k_value: int, column: str) -> pd.Series:
    part = scores[scores.K == k_value].set_index("indices")
    result = part[column].astype(float)
    assert len(result) == N_ALL and result.index.is_unique
    return result


def top_n(series: pd.Series, n_keep: int, eligible: Iterable | None = None) -> set:
    if eligible is not None:
        series = series.loc[list(eligible)]
    n_keep = min(int(n_keep), len(series))
    return set(series.nlargest(n_keep).index)


def fixed_force_fill(
    s1: pd.Series,
    s2: pd.Series,
    total_fraction: float,
    rescue_fraction: float,
    eligible: Iterable | None = None,
) -> set:
    """固定总预算：S2先占救援位，其余按S1补满。"""
    universe = s1.index if eligible is None else pd.Index(list(eligible))
    total_n = min(int(round(N_ALL * total_fraction)), len(universe))
    rescue_n = min(int(round(N_ALL * rescue_fraction)), total_n)
    must = top_n(s2, rescue_n, universe)
    remainder = s1.loc[universe].drop(index=list(must), errors="ignore")
    return must | top_n(remainder, total_n - len(must))


def fixed_rank_fusion(
    s1: pd.Series,
    s2: pd.Series,
    total_fraction: float,
    alpha: float,
    eligible: Iterable | None = None,
) -> set:
    """固定总预算：按百分位融合，alpha是S2权重。"""
    universe = s1.index if eligible is None else pd.Index(list(eligible))
    frame = pd.DataFrame({"s1": s1.loc[universe], "s2": s2.loc[universe]})
    rank1 = frame.s1.rank(pct=True, method="average")
    rank2 = frame.s2.rank(pct=True, method="average")
    fused = (1.0 - alpha) * rank1 + alpha * rank2
    return top_n(fused, int(round(N_ALL * total_fraction)))


def force_fill_score(
    base_score: pd.Series,
    rescue_score: pd.Series,
    total_fraction: float,
    rescue_fraction: float,
    eligible: Iterable,
) -> set:
    return fixed_force_fill(
        base_score,
        rescue_score,
        total_fraction,
        rescue_fraction,
        eligible=eligible,
    )


def best_checkpoint_score(
    scores: pd.DataFrame,
    checkpoints: Iterable[int],
    eligible: Iterable,
) -> pd.Series:
    eligible = pd.Index(list(eligible))
    pieces = []
    for k_value in sorted(set(checkpoints)):
        series = score_series(scores, k_value, "s1_syn3").loc[eligible]
        pieces.append(series.rank(pct=True, method="average").rename(str(k_value)))
    return pd.concat(pieces, axis=1).max(axis=1)


def _scope_mask(truth: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=truth.index)
    return truth.split.eq(scope)


def evaluate_keep(truth: pd.DataFrame, keep: set) -> dict[str, float]:
    selected = truth.indices.isin(keep)
    result: dict[str, float] = {
        "n_keep": len(keep),
        "budget": len(keep) / N_ALL,
    }
    for scope in ("all", "tune", "test"):
        scope_mask = _scope_mask(truth, scope)
        data = truth[scope_mask]
        sel = selected[scope_mask]
        strong = data.strong.to_numpy()
        weight = data.weight.to_numpy(dtype=float)
        sel_arr = sel.to_numpy(dtype=bool)
        strong_weight = float(weight[strong].sum())
        selected_weight = float(weight[sel_arr].sum())
        hit_weight = float(weight[strong & sel_arr].sum())
        result[f"{scope}_recall"] = hit_weight / strong_weight
        result[f"{scope}_precision"] = (
            hit_weight / selected_weight if selected_weight else np.nan
        )
        result[f"{scope}_raw_hit"] = int((strong & sel_arr).sum())
        result[f"{scope}_raw_strong"] = int(strong.sum())
        result[f"{scope}_raw_recall"] = (
            float((strong & sel_arr).sum() / strong.sum()) if strong.sum() else np.nan
        )

        for lo, hi, label in STRENGTH_BANDS:
            band = (data.syn3_true.to_numpy() >= lo) & (
                data.syn3_true.to_numpy() < hi
            )
            denom = float(weight[band].sum())
            result[f"{scope}_{label}_recall"] = (
                float(weight[band & sel_arr].sum() / denom) if denom else np.nan
            )
            result[f"{scope}_{label}_raw_hit"] = int((band & sel_arr).sum())
            result[f"{scope}_{label}_raw_n"] = int(band.sum())
    return result


def pareto_frontier(
    data: pd.DataFrame, cost_col: str, value_col: str
) -> pd.DataFrame:
    ordered = data.sort_values([cost_col, value_col], ascending=[True, False])
    accepted = []
    best = -np.inf
    for row in ordered.itertuples(index=False):
        value = getattr(row, value_col)
        if value > best + 1e-12:
            accepted.append(row._asdict())
            best = value
    return pd.DataFrame(accepted)

