# -*- coding: utf-8 -*-
"""分析K0三元父集合保真、同目标S1与极强协同召回，并选择K网格胜者。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R68 = REPO / "DNN_Aggresvation68"
R69 = REPO / "DNN_Aggresvation69"
R75 = REPO / "DNN_Aggresvation75"
R77 = REPO / "DNN_Aggresvation77"
R79 = REPO / "DNN_Aggresvation79"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from runlog import log  # noqa: E402

N_ALL = 13244
RETENTION = 0.30
NEW_SCHEMES = {
    "triple_only",
    "triple90_pair10",
    "triple70_pair20_uniform10",
    "triple50_pair40_uniform10",
    "triple40_pair30_uniform30",
    "local234_20_60_20",
    "local234_uniform10",
    "warm_local234",
    "warm_triple70",
    "triple70_group8",
    "triple70_group32",
    "local234_group8",
    "local234_group32",
}


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_corr(left, right, weights) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    weights = np.asarray(weights, dtype=float)
    left_centered = left - weighted_mean(left, weights)
    right_centered = right - weighted_mean(right, weights)
    denominator = np.sqrt(
        np.sum(weights * left_centered**2)
        * np.sum(weights * right_centered**2)
    )
    if denominator == 0:
        return np.nan
    return float(
        np.sum(weights * left_centered * right_centered) / denominator
    )


def weighted_spearman(left, right, weights) -> float:
    return weighted_corr(rankdata(left), rankdata(right), weights)


def load_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    data = prepare_data(cfg)
    name_to_idx = {name: idx for idx, name in enumerate(data["general"])}

    exact = pd.read_csv(R68 / "outputs/triples_certified.csv")
    exact["indices"] = [
        tuple(
            sorted(
                (
                    name_to_idx[row.fi],
                    name_to_idx[row.fj],
                    name_to_idx[row.fk],
                )
            )
        )
        for row in exact.itertuples()
    ]
    exact = exact.rename(
        columns={"v_ijk_true": "parent_true", "best_pair": "pair_true"}
    )
    exact = exact[
        ["indices", "conf", "parent_true", "pair_true", "syn3_true"]
    ]
    exact["source"] = "exact397"
    exact["weight"] = 1.0

    random = pd.read_csv(R77 / "outputs/h2_unbiased_pool.csv")
    random = random[random.group == "triple_rand_ext"].copy()
    random["indices"] = random.ix.map(ast.literal_eval)
    random = random.rename(columns={"syn2_true_max": "pair_true"})
    random["parent_true"] = random.pair_true + random.syn3_true
    random = random[
        ["indices", "conf", "parent_true", "pair_true", "syn3_true"]
    ]
    random["source"] = "remainder1800"
    random["weight"] = (N_ALL - 397) / 1800
    entries = pd.concat([exact, random], ignore_index=True)

    design = pd.read_csv(R79 / "outputs/truth_design_split.csv")
    design["indices"] = design.indices.map(ast.literal_eval)
    entries = entries.merge(
        design[["indices", "split"]],
        on="indices",
        how="left",
        validate="many_to_one",
    )
    triples = (
        entries.groupby(["indices", "source", "weight", "split"], as_index=False)
        .agg(syn3_true=("syn3_true", "max"))
    )
    triples["strong"] = triples.syn3_true > 0.10
    return entries, triples


def scheme_seed(path: Path) -> tuple[str, int]:
    stem = path.stem
    if stem.endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    body = stem.removeprefix("k0_")
    scheme, seed = body.rsplit("_seed", 1)
    return scheme, int(seed)


def load_estimate(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["indices"] = list(zip(frame.i, frame.j, frame.k))
    return frame


def make_entries(
    triple_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
) -> pd.DataFrame:
    pair_map = {
        (int(row.i), int(row.j), row.conf): float(row.est)
        for row in pair_frame[pair_frame["size"] == 2].itertuples()
    }
    triples = triple_frame[triple_frame["size"] == 3].copy()
    best_pair = []
    for row in triples.itertuples():
        best_pair.append(
            max(
                pair_map[(min(row.i, row.j), max(row.i, row.j), row.conf)],
                pair_map[(min(row.i, row.k), max(row.i, row.k), row.conf)],
                pair_map[(min(row.j, row.k), max(row.j, row.k), row.conf)],
            )
        )
    triples["best_pair_est"] = best_pair
    triples["syn3_est"] = triples.est - triples.best_pair_est
    triples["indices"] = list(zip(triples.i, triples.j, triples.k))
    return triples[
        ["indices", "conf", "est", "best_pair_est", "syn3_est"]
    ]


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=frame.index)
    return frame.split.eq(scope)


def evaluate(
    scheme: str,
    seed: int,
    mode: str,
    estimated_entries: pd.DataFrame,
    truth_entries: pd.DataFrame,
    truth_triples: pd.DataFrame,
    checkpoint_path: Path,
) -> dict[str, object]:
    joined = truth_entries.merge(
        estimated_entries,
        on=["indices", "conf"],
        how="left",
        validate="one_to_one",
    )
    assert not joined[["est", "best_pair_est", "syn3_est"]].isna().any().any()
    score_rows = estimated_entries.loc[
        estimated_entries.groupby("indices").syn3_est.idxmax(),
        ["indices", "conf", "syn3_est"],
    ].rename(columns={"conf": "s1_conf", "syn3_est": "score"})
    joined = joined.merge(
        score_rows[["indices", "s1_conf"]],
        on="indices",
        how="left",
        validate="many_to_one",
    )
    triple_design = truth_triples.merge(
        score_rows, on="indices", how="left", validate="one_to_one"
    )

    all_scores = (
        estimated_entries.groupby("indices").syn3_est.max().sort_values(ascending=False)
    )
    keep = set(all_scores.index[: round(N_ALL * RETENTION)])
    result: dict[str, object] = {
        "scheme": scheme,
        "seed": seed,
        "mode": mode,
    }
    for scope in ("all", "tune", "test"):
        entry_part = joined[scope_mask(joined, scope)]
        entry_weight = entry_part.weight.to_numpy()
        result[f"{scope}_parent_mae"] = weighted_mean(
            np.abs(entry_part.est - entry_part.parent_true), entry_weight
        )
        result[f"{scope}_parent_bias"] = weighted_mean(
            entry_part.est - entry_part.parent_true, entry_weight
        )
        result[f"{scope}_pair_mae"] = weighted_mean(
            np.abs(entry_part.best_pair_est - entry_part.pair_true),
            entry_weight,
        )
        result[f"{scope}_syn_mae"] = weighted_mean(
            np.abs(entry_part.syn3_est - entry_part.syn3_true), entry_weight
        )

        triple_part = triple_design[scope_mask(triple_design, scope)]
        weight = triple_part.weight.to_numpy()
        selected = triple_part["indices"].isin(keep).to_numpy()
        strong = triple_part.strong.to_numpy()
        hit_weight = float(weight[selected & strong].sum())
        strong_weight = float(weight[strong].sum())
        selected_weight = float(weight[selected].sum())
        recall = hit_weight / strong_weight
        precision = hit_weight / selected_weight
        result[f"{scope}_s1_spearman"] = weighted_spearman(
            triple_part.syn3_true, triple_part.score, weight
        )
        result[f"{scope}_top30_recall"] = recall
        result[f"{scope}_top30_precision"] = precision
        result[f"{scope}_top30_f1"] = 2 * recall * precision / (recall + precision)

    true_max_idx = joined.groupby("indices").syn3_true.idxmax()
    true_max = joined.loc[true_max_idx].copy()
    extreme = true_max[true_max.syn3_true > 0.20]
    result.update(
        {
            "extreme_n": int(len(extreme)),
            "extreme_parent_error_mean": float(
                (extreme.est - extreme.parent_true).mean()
            ),
            "extreme_pair_error_mean": float(
                (extreme.best_pair_est - extreme.pair_true).mean()
            ),
            "extreme_syn_est_mean": float(extreme.syn3_est.mean()),
            "extreme_negative_count": int((extreme.syn3_est < 0).sum()),
            "extreme_top30_hit": int(extreme["indices"].isin(keep).sum()),
            "extreme_conf_match": int((extreme.conf == extreme.s1_conf).sum()),
            "extreme_monotonicity_violation": int(
                (extreme.est < extreme.best_pair_est).sum()
            ),
        }
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result["warm_start"] = bool(checkpoint.get("warm_start", False))
    result["epochs_run"] = int(checkpoint.get("epochs_run", -1))
    result["val_loss"] = float(checkpoint.get("val_loss", np.nan))
    result["r2_full"] = float(checkpoint.get("r2_full", np.nan))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preliminary", "confirm"),
        default="preliminary",
    )
    args = parser.parse_args()
    log("ANALYZE-K0", "START", note="评价三元父集合、S1和极强协同")
    truth_entries, truth_triples = load_truth()
    paths = sorted(OUT.glob("k0_*_seed*.parquet"))
    assert paths, "没有K0输出"
    frames = {
        scheme_seed(path): load_estimate(path)
        for path in paths
    }
    rows = []
    for (scheme, seed), frame in frames.items():
        checkpoint_path = (
            OUT / f"oracle_{scheme}_seed{seed}.pt"
            if (OUT / f"oracle_{scheme}_seed{seed}.pt").exists()
            else R75 / "outputs" / f"oracle_{scheme}_seed{seed}.pt"
        )
        entries = make_entries(frame, frame)
        rows.append(
            evaluate(
                scheme,
                seed,
                "single",
                entries,
                truth_entries,
                truth_triples,
                checkpoint_path,
            )
        )
        if scheme in NEW_SCHEMES and ("uniform", seed) in frames:
            hybrid = make_entries(frame, frames[("uniform", seed)])
            rows.append(
                evaluate(
                    scheme,
                    seed,
                    "hybrid_uniform_pairs",
                    hybrid,
                    truth_entries,
                    truth_triples,
                    checkpoint_path,
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "k0_metrics_by_seed.csv", index=False)
    aggregate = (
        metrics.groupby(["scheme", "mode"], as_index=False)
        .agg(
            n_seed=("seed", "nunique"),
            tune_parent_mae=("tune_parent_mae", "mean"),
            test_parent_mae=("test_parent_mae", "mean"),
            tune_s1_spearman=("tune_s1_spearman", "mean"),
            test_s1_spearman=("test_s1_spearman", "mean"),
            tune_top30_recall=("tune_top30_recall", "mean"),
            test_top30_recall=("test_top30_recall", "mean"),
            tune_top30_precision=("tune_top30_precision", "mean"),
            test_top30_precision=("test_top30_precision", "mean"),
            tune_top30_f1=("tune_top30_f1", "mean"),
            test_top30_f1=("test_top30_f1", "mean"),
            extreme_parent_error=("extreme_parent_error_mean", "mean"),
            extreme_pair_error=("extreme_pair_error_mean", "mean"),
            extreme_syn_est=("extreme_syn_est_mean", "mean"),
            extreme_top30_hit=("extreme_top30_hit", "mean"),
            extreme_conf_match=("extreme_conf_match", "mean"),
            extreme_monotonicity_violation=(
                "extreme_monotonicity_violation",
                "mean",
            ),
            r2_full=("r2_full", "mean"),
            epochs_run=("epochs_run", "mean"),
        )
    )
    aggregate.to_csv(OUT / "k0_scheme_summary.csv", index=False)

    # 选型必须让所有版本站在同一条起跑线上：只看 seed0。
    # seed1 仅用于冻结胜者后的确认，不能把两seed均值与新方案单seed混排。
    seed0_metrics = metrics[metrics.seed == 0]
    new_single = seed0_metrics[
        seed0_metrics.scheme.isin(NEW_SCHEMES)
        & seed0_metrics["mode"].eq("single")
    ]
    best_parent = new_single.sort_values(
        ["tune_parent_mae", "test_parent_mae"]
    ).iloc[0]
    new_all = seed0_metrics[seed0_metrics.scheme.isin(NEW_SCHEMES)]
    best_s1 = new_all.sort_values(
        ["tune_top30_f1", "tune_s1_spearman"],
        ascending=False,
    ).iloc[0]
    if args.stage == "preliminary":
        selection = {
            "selection_uses_test": False,
            "selection_seed": 0,
            "best_parent_scheme": best_parent.scheme,
            "best_parent_mode": best_parent["mode"],
            "best_s1_scheme": best_s1.scheme,
            "best_s1_mode": best_s1["mode"],
            "parent_selection_metric": "seed0 tune_parent_mae minimum",
            "s1_selection_metric": "seed0 tune_top30_f1 maximum",
        }
        (OUT / "k0_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        selection = json.loads(
            (OUT / "k0_selection.json").read_text(encoding="utf-8")
        )
        selection["confirmation_seed"] = 1
        selection["confirmation_does_not_reselect"] = True
        (OUT / "k0_confirmation.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        best_parent = aggregate[
            aggregate.scheme.eq(selection["best_parent_scheme"])
            & aggregate["mode"].eq(selection["best_parent_mode"])
        ].iloc[0]
        best_s1 = aggregate[
            aggregate.scheme.eq(selection["best_s1_scheme"])
            & aggregate["mode"].eq(selection["best_s1_mode"])
        ].iloc[0]
    log(
        "ANALYZE-K0",
        "DECISION",
        note=(
            f"parent胜者={best_parent.scheme}/{best_parent['mode']}；"
            f"S1胜者={best_s1.scheme}/{best_s1['mode']}"
        ),
    )
    print(aggregate.sort_values(["mode", "tune_top30_f1"], ascending=[True, False]).to_string(index=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
