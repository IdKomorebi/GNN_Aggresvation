# -*- coding: utf-8 -*-
"""测试多个 K 排名是否能形成低成本“救援通道”。

目的不是再堆一个黑箱模型，而是回答一个简单问题：
K0、K5、K10 各自看见的强协同若有互补，在固定候选预算下取“多榜平均/任一榜靠前”
能否比只看最后一个 K 更高召回？
"""
from __future__ import annotations

import ast
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from runlog import log  # noqa: E402

N_ALL = 13244
RETENTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def weighted_recall(truth: pd.DataFrame, keep: set[tuple], scope: str) -> float:
    use = truth.strong if scope == "all" else (truth.strong & (truth.split == scope))
    data = truth[use]
    hit = data["indices"].map(lambda key: key in keep).astype(float).to_numpy()
    weight = data.weight.to_numpy()
    return float(np.sum(hit * weight) / np.sum(weight))


def make_rank_features(score: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=score.index)
    for k_value in score.columns:
        # 1 表示该 K 下排名最前，0 表示最后。
        result[k_value] = score[k_value].rank(pct=True, method="average")
    return result


def evaluate_variant(
    rows: list[dict],
    truth: pd.DataFrame,
    ranking_score: pd.Series,
    *,
    label: str,
    max_k: int,
    family: str,
) -> None:
    ranking = ranking_score.sort_values(ascending=False)
    for retention in RETENTIONS:
        keep = set(ranking.index[: int(round(N_ALL * retention))])
        rows.append(
            {
                "variant": label,
                "family": family,
                "max_k": max_k,
                "retention": retention,
                "tune_recall": weighted_recall(truth, keep, "tune"),
                "test_recall": weighted_recall(truth, keep, "test"),
                "all_recall": weighted_recall(truth, keep, "all"),
            }
        )


def main() -> None:
    log("RESCUE", "START", note="测试 K0/K5/K10 多榜互补与调参集线性组合")
    score = pd.read_csv(OUT / "triple_score_kgrid.csv", index_col=0)
    score.index = score.index.map(ast.literal_eval)
    score.columns = [int(column) for column in score.columns]
    truth = pd.read_csv(OUT / "truth_design_split.csv")
    truth["indices"] = truth.indices.map(ast.literal_eval)
    truth["strong"] = truth.strong.astype(bool)
    ranks = make_rank_features(score)

    rows: list[dict] = []
    for k_value in (0, 1, 5, 10, 25, 50):
        evaluate_variant(
            rows,
            truth,
            ranks[k_value],
            label=f"single_K{k_value}",
            max_k=k_value,
            family="single",
        )

    for ks in (
        (0, 5),
        (0, 10),
        (0, 5, 10),
        (0, 10, 25),
        (0, 5, 10, 25),
    ):
        selected = ranks[list(ks)]
        evaluate_variant(
            rows,
            truth,
            selected.mean(axis=1),
            label="mean_rank_" + "_".join(map(str, ks)),
            max_k=max(ks),
            family="mean_rank",
        )
        evaluate_variant(
            rows,
            truth,
            selected.max(axis=1),
            label="best_rank_" + "_".join(map(str, ks)),
            max_k=max(ks),
            family="rescue_any",
        )
        evaluate_variant(
            rows,
            truth,
            selected.mean(axis=1) + 0.5 * selected.std(axis=1),
            label="mean_plus_uncertainty_" + "_".join(map(str, ks)),
            max_k=max(ks),
            family="uncertainty_rescue",
        )

    # 可解释的非负线性权重，只在 tune 选择；权重步长 0.1，避免过度精调。
    base_ks = (0, 5, 10)
    grid = []
    for w0 in np.arange(0, 1.01, 0.1):
        for w5 in np.arange(0, 1.01 - w0, 0.1):
            w10 = 1.0 - w0 - w5
            grid.append((round(w0, 1), round(w5, 1), round(w10, 1)))
    learned_rows = []
    for retention in RETENTIONS:
        candidates = []
        for weights in grid:
            combined = sum(weight * ranks[k] for weight, k in zip(weights, base_ks))
            keep = set(combined.nlargest(int(round(N_ALL * retention))).index)
            candidates.append(
                {
                    "weights": weights,
                    "score": combined,
                    "tune_recall": weighted_recall(truth, keep, "tune"),
                }
            )
        chosen = max(candidates, key=lambda row: (row["tune_recall"], -sum(np.array(row["weights"]) > 0)))
        keep = set(chosen["score"].nlargest(int(round(N_ALL * retention))).index)
        learned_rows.append(
            {
                "variant": "tune_weighted_rank_0_5_10",
                "family": "tune_weighted_rank",
                "max_k": 10,
                "retention": retention,
                "weights": json.dumps(chosen["weights"]),
                "tune_recall": chosen["tune_recall"],
                "test_recall": weighted_recall(truth, keep, "test"),
                "all_recall": weighted_recall(truth, keep, "all"),
            }
        )
    result = pd.concat([pd.DataFrame(rows), pd.DataFrame(learned_rows)], ignore_index=True)
    result.to_csv(OUT / "rescue_rank_metrics.csv", index=False)

    # 每个候选预算只允许 tune 决策一次，再展示隔离 test。
    selected_rows = []
    for retention, group in result.groupby("retention"):
        # 计算公平：只在 max K<=10 的低成本方案中选；K25/K50 留作上限对照。
        eligible = group[group.max_k <= 10]
        chosen = eligible.sort_values(
            ["tune_recall", "max_k"], ascending=[False, True]
        ).iloc[0]
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT / "rescue_rank_selected.csv", index=False)
    log(
        "RESCUE",
        "DECISION",
        note="；".join(
            f"top{row.retention:.0%}={row.variant}, test recall {row.test_recall:.1%}"
            for row in selected.itertuples()
        ),
    )
    print(
        selected[
            ["retention", "variant", "max_k", "weights", "tune_recall", "test_recall", "all_recall"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
