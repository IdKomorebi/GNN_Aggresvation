# -*- coding: utf-8 -*-
"""诊断强三阶协同为何会被 S1 漏掉，以及 S2 为何能救回。

核心分解（对同一 triple、同一 confidential 目标）：

    syn3_est - syn3_true
      = (v3_est - v3_true) - (best_pair_est - best_pair_true)
      = parent_error - pair_error

这样可以直接区分三元父集合低估和二元子集合高估。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R68 = REPO / "DNN_Aggresvation68"
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
R79 = REPO / "DNN_Aggresvation79"
OUT = ROOT / "outputs"

sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    N_ALL,
    best_checkpoint_score,
    fixed_force_fill,
    load_scores,
    score_series,
    top_n,
)

KGRID = (0, 1, 5, 10, 25, 50)


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column], weights=frame["weight"]))


def load_truth_entries() -> pd.DataFrame:
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
        columns={
            "v_ijk_true": "parent_true",
            "best_pair": "pair_true",
        }
    )
    exact = exact[
        ["indices", "conf", "parent_true", "pair_true", "syn3_true"]
    ].copy()
    exact["source"] = "exact397"
    exact["weight"] = 1.0

    random = pd.read_csv(R77 / "outputs/h2_unbiased_pool.csv")
    random = random[random["group"] == "triple_rand_ext"].copy()
    random["indices"] = random["ix"].map(ast.literal_eval)
    random = random.rename(columns={"syn2_true_max": "pair_true"})
    random["parent_true"] = random["pair_true"] + random["syn3_true"]
    random = random[
        ["indices", "conf", "parent_true", "pair_true", "syn3_true"]
    ].copy()
    random["source"] = "remainder1800"
    random["weight"] = (13244 - 397) / 1800

    truth = pd.concat([exact, random], ignore_index=True)
    assert truth.groupby(["indices", "conf"]).size().eq(1).all()
    return truth


def load_estimates() -> pd.DataFrame:
    estimates = pd.read_csv(R79 / "outputs/syn3_entries_kgrid.csv.gz")
    estimates["indices"] = list(zip(estimates.i, estimates.j, estimates.k))
    return estimates[
        ["indices", "conf", "K", "est", "best_pair_est", "syn3_est"]
    ].copy()


def triple_scores(estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (indices, k_value), group in estimates.groupby(["indices", "K"]):
        s1_row = group.loc[group["syn3_est"].idxmax()]
        s2_row = group.loc[group["est"].idxmax()]
        rows.append(
            {
                "indices": indices,
                "K": int(k_value),
                "s1": float(s1_row.syn3_est),
                "s1_conf": s1_row.conf,
                "s2_any": float(s2_row.est),
                "s2_conf": s2_row.conf,
            }
        )
    result = pd.DataFrame(rows)
    result["s1_rank"] = result.groupby("K")["s1"].rank(
        method="min", ascending=False
    )
    result["s2_rank"] = result.groupby("K")["s2_any"].rank(
        method="min", ascending=False
    )
    result["s1_top30"] = result["s1_rank"] <= round(13244 * 0.30)
    result["s2_top20"] = result["s2_rank"] <= round(13244 * 0.20)
    return result


def selected_protocol_stages(extreme_indices: set[tuple]) -> tuple[pd.DataFrame, dict]:
    """重建81号正式协议，检查S2究竟在哪一层救回了哪些极强组合。"""
    scores = load_scores()
    s1_k10 = score_series(scores, 10, "s1_syn3")
    s2_k10 = score_series(scores, 10, "s2_parent_any")
    pure_stage1 = top_n(s1_k10, round(N_ALL * 0.60))
    mixed_stage1 = fixed_force_fill(s1_k10, s2_k10, 0.60, 0.20)
    rescued = mixed_stage1 - pure_stage1

    best_s1_baseline = best_checkpoint_score(
        scores, [0, 10, 25], pure_stage1
    )
    baseline_final = top_n(best_s1_baseline, round(N_ALL * 0.30))

    best_s1_mixed = best_checkpoint_score(scores, [0, 10, 25], mixed_stage1)
    protected = top_n(
        s2_k10,
        round(N_ALL * 0.03),
        eligible=rescued,
    )
    remainder = best_s1_mixed.drop(index=list(protected), errors="ignore")
    selected_final = protected | top_n(
        remainder, round(N_ALL * 0.30) - len(protected)
    )

    rows = []
    for indices in sorted(extreme_indices):
        rows.append(
            {
                "indices": indices,
                "pure_s1_k10_top60": indices in pure_stage1,
                "s1_s2_k10_top60": indices in mixed_stage1,
                "newly_rescued_by_s2": indices in rescued,
                "protected_at_final": indices in protected,
                "d79_final": indices in baseline_final,
                "d81_final": indices in selected_final,
            }
        )
    stages = pd.DataFrame(rows)
    counts = {
        column: int(stages[column].sum())
        for column in stages.columns
        if column != "indices"
    }
    counts["d81_minus_d79_new_hits"] = int(
        (stages.d81_final & ~stages.d79_final).sum()
    )
    counts["d79_hits_lost_by_d81"] = int(
        (stages.d79_final & ~stages.d81_final).sum()
    )
    return stages, counts


def make_diagnostics() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]
]:
    truth = load_truth_entries()
    estimates = load_estimates()
    scores = triple_scores(estimates)

    # 每个三元组以真值最大的 confidential 作为“真正协同目标”。
    true_idx = truth.groupby("indices")["syn3_true"].idxmax()
    true_max = truth.loc[true_idx].copy()
    true_max = true_max.rename(columns={"conf": "true_conf"})
    strongest = true_max[true_max["syn3_true"] > 0.20].copy()
    strong = true_max[true_max["syn3_true"] > 0.10].copy()
    assert len(strongest) == 15, len(strongest)

    same_conf = estimates.merge(
        true_max[
            [
                "indices",
                "true_conf",
                "parent_true",
                "pair_true",
                "syn3_true",
                "source",
                "weight",
            ]
        ],
        left_on=["indices", "conf"],
        right_on=["indices", "true_conf"],
        how="inner",
        validate="many_to_one",
    )
    same_conf["parent_error"] = same_conf["est"] - same_conf["parent_true"]
    same_conf["pair_error"] = (
        same_conf["best_pair_est"] - same_conf["pair_true"]
    )
    same_conf["syn_error"] = same_conf["syn3_est"] - same_conf["syn3_true"]
    same_conf["error_identity_residual"] = (
        same_conf["syn_error"]
        - (same_conf["parent_error"] - same_conf["pair_error"])
    )
    assert same_conf["error_identity_residual"].abs().max() < 1e-10

    detail = (
        same_conf.merge(scores, on=["indices", "K"], validate="one_to_one")
        .sort_values(["syn3_true", "indices", "K"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    detail["s1_conf_matches_true"] = detail["s1_conf"] == detail["true_conf"]
    detail["s2_conf_matches_true"] = detail["s2_conf"] == detail["true_conf"]
    detail["true_strength"] = pd.cut(
        detail["syn3_true"],
        bins=[-np.inf, 0.10, 0.15, 0.20, np.inf],
        labels=["<=0.10", "0.10-0.15", "0.15-0.20", ">0.20"],
        right=True,
    )

    summary_rows: list[dict[str, object]] = []
    for label, indices in {
        "strong_gt_010": set(strong["indices"]),
        "extreme_gt_020": set(strongest["indices"]),
    }.items():
        subset = detail[detail["indices"].isin(indices)]
        for k_value, group in subset.groupby("K"):
            summary_rows.append(
                {
                    "cohort": label,
                    "K": int(k_value),
                    "n_triples": int(group["indices"].nunique()),
                    "parent_true_mean": weighted_mean(group, "parent_true"),
                    "pair_true_mean": weighted_mean(group, "pair_true"),
                    "syn3_true_mean": weighted_mean(group, "syn3_true"),
                    "parent_est_mean": weighted_mean(group, "est"),
                    "pair_est_mean": weighted_mean(group, "best_pair_est"),
                    "syn3_same_conf_est_mean": weighted_mean(
                        group, "syn3_est"
                    ),
                    "s1_est_mean": weighted_mean(group, "s1"),
                    "parent_error_mean": weighted_mean(group, "parent_error"),
                    "pair_error_mean": weighted_mean(group, "pair_error"),
                    "syn_error_mean": weighted_mean(group, "syn_error"),
                    "negative_same_conf_rate": weighted_mean(
                        group.assign(flag=(group.syn3_est < 0).astype(float)),
                        "flag",
                    ),
                    "s1_top30_rate": weighted_mean(
                        group.assign(flag=group.s1_top30.astype(float)), "flag"
                    ),
                    "s2_top20_rate": weighted_mean(
                        group.assign(flag=group.s2_top20.astype(float)), "flag"
                    ),
                    "s1_conf_match_rate": weighted_mean(
                        group.assign(
                            flag=group.s1_conf_matches_true.astype(float)
                        ),
                        "flag",
                    ),
                    "s2_conf_match_rate": weighted_mean(
                        group.assign(
                            flag=group.s2_conf_matches_true.astype(float)
                        ),
                        "flag",
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)

    extreme_k10 = detail[
        detail["indices"].isin(set(strongest["indices"])) & (detail["K"] == 10)
    ].copy()
    stage_detail, stage_counts = selected_protocol_stages(
        set(strongest["indices"])
    )
    decomposition = {
        "n_extreme": int(len(extreme_k10)),
        "k10_negative_same_conf_count": int((extreme_k10.syn3_est < 0).sum()),
        "k10_parent_underestimated_count": int(
            (extreme_k10.parent_error < 0).sum()
        ),
        "k10_pair_overestimated_count": int(
            (extreme_k10.pair_error > 0).sum()
        ),
        "k10_parent_more_underfit_than_pair_count": int(
            (extreme_k10.parent_error < extreme_k10.pair_error).sum()
        ),
        "k10_s1_top30_count": int(extreme_k10.s1_top30.sum()),
        "k10_s2_top20_count": int(extreme_k10.s2_top20.sum()),
        "k10_s1_conf_match_count": int(extreme_k10.s1_conf_matches_true.sum()),
        "k10_s2_conf_match_count": int(extreme_k10.s2_conf_matches_true.sum()),
        "k10_s2_top20_but_s1_not_top30_count": int(
            (extreme_k10.s2_top20 & ~extreme_k10.s1_top30).sum()
        ),
        "k10_s2_rescue_conf_match_count": int(
            (
                extreme_k10.s2_top20
                & ~extreme_k10.s1_top30
                & extreme_k10.s2_conf_matches_true
            ).sum()
        ),
        "selected_protocol_extreme_stages": stage_counts,
    }
    return detail, summary, stage_detail, decomposition


def main() -> None:
    detail, summary, stage_detail, decomposition = make_diagnostics()
    detail.to_csv(OUT / "s1_s2_error_decomposition.csv", index=False)
    summary.to_csv(OUT / "s1_s2_error_summary.csv", index=False)
    stage_detail.to_csv(
        OUT / "s1_s2_extreme_protocol_stages.csv", index=False
    )

    # 报告图使用15个极强三元组的原始样本均值；不把抽样权重和原始个数混在一起。
    extreme = detail[detail["syn3_true"] > 0.20].copy()
    raw_grid = (
        extreme.groupby("K", as_index=False)
        .agg(
            parent_true=("parent_true", "mean"),
            parent_est=("est", "mean"),
            pair_true=("pair_true", "mean"),
            pair_est=("best_pair_est", "mean"),
            syn3_true=("syn3_true", "mean"),
            syn3_est_same_conf=("syn3_est", "mean"),
            s1=("s1", "mean"),
            s2_any=("s2_any", "mean"),
            negative_same_conf_count=("syn3_est", lambda s: int((s < 0).sum())),
            s1_top30_count=("s1_top30", "sum"),
            s2_top20_count=("s2_top20", "sum"),
        )
        .sort_values("K")
    )
    raw_grid.to_csv(OUT / "s1_s2_extreme_kgrid_raw.csv", index=False)

    strength = detail[(detail["K"] == 10) & (detail["syn3_true"] > 0.10)].copy()
    strength["band"] = pd.cut(
        strength.syn3_true,
        bins=[0.10, 0.12, 0.15, 0.20, np.inf],
        labels=["0.10–0.12", "0.12–0.15", "0.15–0.20", ">0.20"],
        include_lowest=True,
    )
    strength_raw = (
        strength.groupby("band", observed=True, as_index=False)
        .agg(
            n=("indices", "nunique"),
            parent_error=("parent_error", "mean"),
            pair_error=("pair_error", "mean"),
            syn_error=("syn_error", "mean"),
            s1_top30_rate=("s1_top30", "mean"),
        )
    )
    strength_raw.to_csv(OUT / "s1_s2_strength_k10_raw.csv", index=False)

    (OUT / "s1_s2_error_summary.json").write_text(
        json.dumps(decomposition, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(decomposition, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
