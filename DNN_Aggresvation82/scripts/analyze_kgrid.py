# -*- coding: utf-8 -*-
"""比较 uniform 与三元组专用 oracle 在 K 步微调后的绝对值和 S1 保真。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R79 = REPO / "DNN_Aggresvation79"
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_k0 import (  # noqa: E402
    N_ALL,
    load_truth,
    scope_mask,
    weighted_mean,
    weighted_spearman,
)
from runlog import log  # noqa: E402

KGRID = (0, 1, 5, 10, 25, 50)
RETENTION = 0.30


def load_grid(scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = R79 if scheme == "uniform" else ROOT
    low_paths = sorted(
        (base / "outputs").glob(
            f"kgrid_low_{scheme}_seed0_shard*.csv.gz"
        )
    )
    triple_paths = sorted(
        (base / "outputs").glob(
            f"kgrid_triple_{scheme}_seed0_shard*.csv.gz"
        )
    )
    if len(low_paths) != 1 or not triple_paths:
        raise RuntimeError(
            f"{scheme} K-grid不完整：low={len(low_paths)}, triple={len(triple_paths)}"
        )
    low = pd.concat([pd.read_csv(path) for path in low_paths], ignore_index=True)
    triple = pd.concat(
        [pd.read_csv(path) for path in triple_paths], ignore_index=True
    )
    assert len(low) == 990 * len(KGRID) * 12
    assert len(triple) == N_ALL * len(KGRID) * 12
    return low, triple


def attach_children(
    triple: pd.DataFrame, low: pd.DataFrame
) -> pd.DataFrame:
    pairs = low[low["size"] == 2]
    pair_map = {
        (int(row.i), int(row.j), int(row.K), row.conf): float(row.est)
        for row in pairs.itertuples()
    }
    best_pair = []
    for row in triple.itertuples():
        best_pair.append(
            max(
                pair_map[(min(row.i, row.j), max(row.i, row.j), row.K, row.conf)],
                pair_map[(min(row.i, row.k), max(row.i, row.k), row.K, row.conf)],
                pair_map[(min(row.j, row.k), max(row.j, row.k), row.K, row.conf)],
            )
        )
    result = triple[["i", "j", "k", "K", "conf", "est"]].copy()
    result["best_pair_est"] = best_pair
    result["syn3_est"] = result.est - result.best_pair_est
    result["indices"] = list(zip(result.i, result.j, result.k))
    return result


def evaluate_mode(
    scheme: str,
    mode: str,
    entries: pd.DataFrame,
    truth_entries: pd.DataFrame,
    truth_triples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = truth_entries.merge(
        entries,
        on=["indices", "conf"],
        how="left",
        validate="one_to_many",
    )
    assert not joined[["est", "best_pair_est", "syn3_est"]].isna().any().any()
    rows: list[dict[str, object]] = []
    strength_rows: list[dict[str, object]] = []
    bins = [-np.inf, 0.10, 0.12, 0.15, 0.20, np.inf]
    labels = ["<=0.10", "0.10-0.12", "0.12-0.15", "0.15-0.20", ">0.20"]

    for k_value in KGRID:
        part = joined[joined.K == k_value].copy()
        score_rows = part.loc[
            part.groupby("indices").syn3_est.idxmax(),
            ["indices", "conf", "syn3_est"],
        ].rename(columns={"conf": "s1_conf", "syn3_est": "score"})
        triple = truth_triples.merge(
            score_rows, on="indices", how="left", validate="one_to_one"
        )
        ranking = (
            entries[entries.K == k_value]
            .groupby("indices")
            .syn3_est.max()
            .sort_values(ascending=False)
        )
        keep = set(ranking.index[: round(N_ALL * RETENTION)])

        row: dict[str, object] = {
            "scheme": scheme,
            "mode": mode,
            "K": k_value,
        }
        for scope in ("all", "tune", "test"):
            entry_scope = part[scope_mask(part, scope)]
            weight = entry_scope.weight.to_numpy()
            row[f"{scope}_parent_mae"] = weighted_mean(
                np.abs(entry_scope.est - entry_scope.parent_true), weight
            )
            row[f"{scope}_parent_bias"] = weighted_mean(
                entry_scope.est - entry_scope.parent_true, weight
            )
            row[f"{scope}_pair_mae"] = weighted_mean(
                np.abs(entry_scope.best_pair_est - entry_scope.pair_true), weight
            )
            row[f"{scope}_syn_mae"] = weighted_mean(
                np.abs(entry_scope.syn3_est - entry_scope.syn3_true), weight
            )

            triple_scope = triple[scope_mask(triple, scope)]
            triple_weight = triple_scope.weight.to_numpy()
            selected = triple_scope["indices"].isin(keep).to_numpy()
            strong = triple_scope.strong.to_numpy()
            hit_weight = float(triple_weight[selected & strong].sum())
            positive_weight = float(triple_weight[strong].sum())
            selected_weight = float(triple_weight[selected].sum())
            recall = hit_weight / positive_weight
            precision = hit_weight / selected_weight
            row[f"{scope}_s1_spearman"] = weighted_spearman(
                triple_scope.syn3_true, triple_scope.score, triple_weight
            )
            row[f"{scope}_top30_recall"] = recall
            row[f"{scope}_top30_precision"] = precision
            row[f"{scope}_top30_f1"] = (
                2 * recall * precision / (recall + precision)
            )

        true_max = part.loc[part.groupby("indices").syn3_true.idxmax()].copy()
        extreme = true_max[true_max.syn3_true > 0.20].copy()
        row.update(
            {
                "extreme_n": len(extreme),
                "extreme_parent_true_mean": extreme.parent_true.mean(),
                "extreme_parent_est_mean": extreme.est.mean(),
                "extreme_parent_error_mean": (
                    extreme.est - extreme.parent_true
                ).mean(),
                "extreme_pair_true_mean": extreme.pair_true.mean(),
                "extreme_pair_est_mean": extreme.best_pair_est.mean(),
                "extreme_pair_error_mean": (
                    extreme.best_pair_est - extreme.pair_true
                ).mean(),
                "extreme_syn_true_mean": extreme.syn3_true.mean(),
                "extreme_syn_est_mean": extreme.syn3_est.mean(),
                "extreme_negative_count": int((extreme.syn3_est < 0).sum()),
                "extreme_top30_hit": int(extreme["indices"].isin(keep).sum()),
                "extreme_conf_match": int(
                    (
                        extreme.conf
                        == extreme["indices"].map(
                            score_rows.set_index("indices").s1_conf
                        )
                    ).sum()
                ),
                "extreme_monotonicity_violation": int(
                    (extreme.est < extreme.best_pair_est).sum()
                ),
            }
        )
        rows.append(row)

        strength = triple.merge(
            true_max[
                [
                    "indices",
                    "parent_true",
                    "pair_true",
                    "est",
                    "best_pair_est",
                    "syn3_est",
                ]
            ],
            on="indices",
            how="left",
            validate="one_to_one",
        )
        strength["strength_bin"] = pd.cut(
            strength.syn3_true, bins=bins, labels=labels
        )
        strength["hit"] = strength["indices"].isin(keep)
        for label, group in strength.groupby("strength_bin", observed=True):
            strength_rows.append(
                {
                    "scheme": scheme,
                    "mode": mode,
                    "K": k_value,
                    "strength_bin": str(label),
                    "n_sample": len(group),
                    "estimated_count": group.weight.sum(),
                    "recall_top30": weighted_mean(
                        group.hit.astype(float), group.weight
                    ),
                    "parent_error_mean": weighted_mean(
                        group.est - group.parent_true, group.weight
                    ),
                    "pair_error_mean": weighted_mean(
                        group.best_pair_est - group.pair_true, group.weight
                    ),
                    "syn_est_mean": weighted_mean(
                        group.syn3_est, group.weight
                    ),
                    "syn_true_mean": weighted_mean(
                        group.syn3_true, group.weight
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(strength_rows)


def main() -> None:
    log("ANALYZE-KGRID", "START", note="比较uniform与两个K0胜者")
    selection = json.loads(
        (OUT / "k0_selection.json").read_text(encoding="utf-8")
    )
    winners = sorted(
        {
            selection["best_parent_scheme"],
            selection["best_s1_scheme"],
        }
    )
    truth_entries, truth_triples = load_truth()
    uniform_low, uniform_triple = load_grid("uniform")
    uniform_entries = attach_children(uniform_triple, uniform_low)

    metric_frames = []
    strength_frames = []
    metrics, strength = evaluate_mode(
        "uniform",
        "single",
        uniform_entries,
        truth_entries,
        truth_triples,
    )
    metric_frames.append(metrics)
    strength_frames.append(strength)

    for scheme in winners:
        low, triple = load_grid(scheme)
        for mode, child_low in (
            ("single", low),
            ("hybrid_uniform_pairs", uniform_low),
        ):
            entries = attach_children(triple, child_low)
            metrics, strength = evaluate_mode(
                scheme,
                mode,
                entries,
                truth_entries,
                truth_triples,
            )
            metric_frames.append(metrics)
            strength_frames.append(strength)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_strength = pd.concat(strength_frames, ignore_index=True)
    all_metrics.to_csv(OUT / "kgrid_comparison.csv", index=False)
    all_strength.to_csv(OUT / "kgrid_recall_by_strength.csv", index=False)
    useful = [
        "scheme",
        "mode",
        "K",
        "test_parent_mae",
        "test_s1_spearman",
        "test_top30_recall",
        "test_top30_f1",
        "extreme_parent_error_mean",
        "extreme_pair_error_mean",
        "extreme_syn_est_mean",
        "extreme_negative_count",
        "extreme_top30_hit",
    ]
    print(all_metrics[useful].to_string(index=False))
    log(
        "ANALYZE-KGRID",
        "DONE",
        note=f"完成{len(all_metrics)}个方案×K比较",
    )


if __name__ == "__main__":
    main()
