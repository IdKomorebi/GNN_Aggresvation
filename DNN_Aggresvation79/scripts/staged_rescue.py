# -*- coding: utf-8 -*-
"""模拟“先宽进、再精排”的真正分层协议，并计入保存状态后的连续训练成本。"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from runlog import log  # noqa: E402

N_ALL = 13244
FINAL_KEEP = int(round(N_ALL * 0.30))
FINAL_K = 25


def recall(truth: pd.DataFrame, keep: set, scope: str) -> float:
    use = truth.strong if scope == "all" else truth.strong & (truth.split == scope)
    data = truth[use]
    hit = data.indices.map(lambda key: key in keep).astype(float)
    return float((hit * data.weight).sum() / data.weight.sum())


def main() -> None:
    score = pd.read_csv(OUT / "triple_score_kgrid.csv", index_col=0)
    score.index = score.index.map(ast.literal_eval)
    score.columns = [int(column) for column in score.columns]
    truth = pd.read_csv(OUT / "truth_design_split.csv")
    truth["indices"] = truth.indices.map(ast.literal_eval)
    truth["strong"] = truth.strong.astype(bool)
    rows = []

    for first_k in (0, 5, 10):
        for first_retention in (0.40, 0.50, 0.60, 0.80, 1.00):
            first_n = int(round(N_ALL * first_retention))
            active = score[first_k].nlargest(first_n).index
            active_score = score.loc[active]

            variants = {"single_K25": active_score[25]}
            checkpoint_ks = [k for k in (0, 5, 10, 25) if k >= first_k or k == 0]
            ranks = active_score[checkpoint_ks].rank(pct=True)
            variants["best_checkpoint_rank"] = ranks.max(axis=1)

            for variant, final_score in variants.items():
                keep = set(final_score.nlargest(FINAL_KEEP).index)
                continuation = first_k + first_retention * (FINAL_K - first_k)
                restart = first_k + first_retention * FINAL_K
                rows.append(
                    {
                        "first_k": first_k,
                        "first_retention": first_retention,
                        "final_k": FINAL_K,
                        "final_retention": 0.30,
                        "variant": variant,
                        "avg_updates_continuation": continuation,
                        "avg_updates_restart": restart,
                        "tune_recall": recall(truth, keep, "tune"),
                        "test_recall": recall(truth, keep, "test"),
                        "all_recall": recall(truth, keep, "all"),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "staged_rescue_top30.csv", index=False)
    frontier = result.sort_values(
        ["avg_updates_continuation", "tune_recall"], ascending=[True, False]
    )
    accepted = []
    best = -1.0
    for row in frontier.itertuples():
        if row.tune_recall > best + 1e-12:
            accepted.append(row._asdict())
            best = row.tune_recall
    pd.DataFrame(accepted).to_csv(OUT / "staged_rescue_frontier.csv", index=False)
    log(
        "STAGED-RESCUE",
        "DONE",
        note="完成 K0/K5/K10 宽进 → K25 Top30 精排及多checkpoint救援模拟",
    )
    print(
        result.sort_values("all_recall", ascending=False)[
            [
                "first_k",
                "first_retention",
                "variant",
                "avg_updates_continuation",
                "tune_recall",
                "test_recall",
                "all_recall",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
