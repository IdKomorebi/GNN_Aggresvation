# -*- coding: utf-8 -*-
"""模拟“先微调再S1+S2宽进，然后继续微调精排”的端到端分层协议。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    N_ALL,
    best_checkpoint_score,
    evaluate_keep,
    fixed_force_fill,
    fixed_rank_fusion,
    load_scores,
    load_truth,
    score_series,
    top_n,
)
from runlog import Timer, log  # noqa: E402

FIRST_KS = (0, 1, 5, 10, 25)
FIRST_RETENTIONS = (0.30, 0.40, 0.50, 0.60, 0.80)
FINAL_KS = (10, 25, 50)
FINAL_RETENTIONS = (0.10, 0.20, 0.30)
FIRST_RESCUES = (0.01, 0.02, 0.05, 0.10)
FIRST_ALPHAS = (0.10, 0.20, 0.30, 0.40, 0.50)
FINAL_RESCUES = (0.02, 0.05)
FINAL_ALPHAS = (0.10, 0.20, 0.30, 0.40, 0.50)


def stage1_variants(s1, s2_any, s2_aligned, retention):
    yield "s1_only", "none", 0.0, top_n(s1, round(N_ALL * retention))
    for rescue in FIRST_RESCUES:
        if rescue <= retention:
            yield (
                "force_fill",
                "any",
                rescue,
                fixed_force_fill(s1, s2_any, retention, rescue),
            )
            yield (
                "force_fill",
                "aligned",
                rescue,
                fixed_force_fill(s1, s2_aligned, retention, rescue),
            )
    for alpha in FIRST_ALPHAS:
        yield (
            "rank_fusion",
            "any",
            alpha,
            fixed_rank_fusion(s1, s2_any, retention, alpha),
        )
        yield (
            "rank_fusion",
            "aligned",
            alpha,
            fixed_rank_fusion(s1, s2_aligned, retention, alpha),
        )


def main() -> None:
    log("STAGED", "START", note="两层S1+S2端到端参数扫描")
    with Timer() as timer:
        scores = load_scores()
        truth = load_truth()
        rows = []
        for first_k in FIRST_KS:
            s1_first = score_series(scores, first_k, "s1_syn3")
            s2_any_first = score_series(scores, first_k, "s2_parent_any")
            s2_aligned_first = score_series(scores, first_k, "s2_parent_aligned")
            for retention in FIRST_RETENTIONS:
                for (
                    first_family,
                    first_s2,
                    first_parameter,
                    active,
                ) in stage1_variants(
                    s1_first, s2_any_first, s2_aligned_first, retention
                ):
                    for final_k in FINAL_KS:
                        if final_k <= first_k:
                            continue
                        s1_final = score_series(scores, final_k, "s1_syn3")
                        s2_any_final = score_series(
                            scores, final_k, "s2_parent_any"
                        )
                        s2_aligned_final = score_series(
                            scores, final_k, "s2_parent_aligned"
                        )
                        checkpoints = [0, first_k, final_k]
                        best_s1 = best_checkpoint_score(
                            scores, checkpoints, active
                        )
                        for final_retention in FINAL_RETENTIONS:
                            if final_retention > retention:
                                continue
                            final_n = round(N_ALL * final_retention)
                            finals = [
                                (
                                    "s1_final",
                                    "none",
                                    0.0,
                                    top_n(s1_final, final_n, active),
                                ),
                                (
                                    "best_s1",
                                    "none",
                                    0.0,
                                    top_n(best_s1, final_n),
                                ),
                            ]
                            for rescue in FINAL_RESCUES:
                                if rescue <= final_retention:
                                    finals.extend(
                                        [
                                            (
                                                "best_s1_force",
                                                "any",
                                                rescue,
                                                fixed_force_fill(
                                                    best_s1,
                                                    s2_any_final,
                                                    final_retention,
                                                    rescue,
                                                    eligible=active,
                                                ),
                                            ),
                                            (
                                                "best_s1_force",
                                                "aligned",
                                                rescue,
                                                fixed_force_fill(
                                                    best_s1,
                                                    s2_aligned_final,
                                                    final_retention,
                                                    rescue,
                                                    eligible=active,
                                                ),
                                            ),
                                        ]
                                    )
                            for alpha in FINAL_ALPHAS:
                                finals.extend(
                                    [
                                        (
                                            "best_s1_fusion",
                                            "any",
                                            alpha,
                                            fixed_rank_fusion(
                                                best_s1,
                                                s2_any_final,
                                                final_retention,
                                                alpha,
                                                eligible=active,
                                            ),
                                        ),
                                        (
                                            "best_s1_fusion",
                                            "aligned",
                                            alpha,
                                            fixed_rank_fusion(
                                                best_s1,
                                                s2_aligned_final,
                                                final_retention,
                                                alpha,
                                                eligible=active,
                                            ),
                                        ),
                                    ]
                                )
                            continuation = first_k + retention * (
                                final_k - first_k
                            )
                            restart = first_k + retention * final_k
                            for (
                                final_family,
                                final_s2,
                                final_parameter,
                                keep,
                            ) in finals:
                                rows.append(
                                    {
                                        "first_k": first_k,
                                        "first_retention": retention,
                                        "first_family": first_family,
                                        "first_s2": first_s2,
                                        "first_parameter": first_parameter,
                                        "final_k": final_k,
                                        "final_retention": final_retention,
                                        "final_family": final_family,
                                        "final_s2": final_s2,
                                        "final_parameter": final_parameter,
                                        "avg_updates_continuation": continuation,
                                        "avg_updates_restart": restart,
                                        **evaluate_keep(truth, keep),
                                    }
                                )

        result = pd.DataFrame(rows)
        result.to_csv(ROOT / "outputs" / "staged_sweep.csv", index=False)

    log(
        "STAGED",
        "DONE",
        note="完成先微调、S1+S2宽进、继续微调精排的全扫描",
        elapsed_s=timer.elapsed,
        rows=len(result),
    )
    print(f"完成 staged_sweep.csv：{len(result)}个协议")


if __name__ == "__main__":
    main()
