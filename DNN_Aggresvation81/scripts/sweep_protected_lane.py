# -*- coding: utf-8 -*-
"""测试“第一层S2救回候选的少量保护通道”。

动机：K10的S1+S2 Top60能显著提高覆盖，但这些候选到K25按S1缩到Top30时
可能再次被删除。这里固定总Top30预算，只给“相对纯S1新换入”的候选保留少量名额。
"""
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

FIRST_KS = (5, 10)
FIRST_RETENTIONS = (0.40, 0.50, 0.60, 0.80)
FINAL_KS = (25, 50)
FINAL_RETENTIONS = (0.20, 0.30)
PROTECT_FRACTIONS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)


def selector_variants(s1, s2_any, s2_aligned, retention):
    for alpha in (0.20, 0.30, 0.40, 0.50):
        # 同一目标对齐的融合，是单层扫描里最稳定的宽进方案。
        fused_score = (
            (1 - alpha) * s1.rank(pct=True)
            + alpha * s2_aligned.rank(pct=True)
        )
        active = top_n(fused_score, round(N_ALL * retention))
        yield "rank_fusion", "aligned", alpha, active, fused_score
    for rescue in (0.01, 0.02, 0.05, 0.10, 0.20):
        for label, s2 in (("aligned", s2_aligned), ("any", s2_any)):
            active = fixed_force_fill(s1, s2, retention, rescue)
            yield "force_fill", label, rescue, active, s2


def main() -> None:
    log("PROTECT", "START", note="扫描S2救回候选的保护名额")
    with Timer() as timer:
        scores = load_scores()
        truth = load_truth()
        rows = []
        for first_k in FIRST_KS:
            s1_first = score_series(scores, first_k, "s1_syn3")
            s2_any_first = score_series(scores, first_k, "s2_parent_any")
            s2_aligned_first = score_series(
                scores, first_k, "s2_parent_aligned"
            )
            for first_retention in FIRST_RETENTIONS:
                baseline_active = top_n(
                    s1_first, round(N_ALL * first_retention)
                )
                for (
                    first_family,
                    first_s2,
                    first_parameter,
                    active,
                    protect_score,
                ) in selector_variants(
                    s1_first,
                    s2_any_first,
                    s2_aligned_first,
                    first_retention,
                ):
                    rescued = active - baseline_active
                    for final_k in FINAL_KS:
                        best_s1 = best_checkpoint_score(
                            scores, [0, first_k, final_k], active
                        )
                        for final_retention in FINAL_RETENTIONS:
                            final_n = round(N_ALL * final_retention)
                            if final_n > len(active):
                                continue
                            for protect_fraction in PROTECT_FRACTIONS:
                                protect_n = min(
                                    round(N_ALL * protect_fraction),
                                    len(rescued),
                                    final_n,
                                )
                                protected = top_n(
                                    protect_score,
                                    protect_n,
                                    eligible=rescued,
                                )
                                remaining = best_s1.drop(
                                    index=list(protected), errors="ignore"
                                )
                                keep = protected | top_n(
                                    remaining, final_n - len(protected)
                                )
                                rows.append(
                                    {
                                        "first_k": first_k,
                                        "first_retention": first_retention,
                                        "first_family": first_family,
                                        "first_s2": first_s2,
                                        "first_parameter": first_parameter,
                                        "n_rescued_stage1": len(rescued),
                                        "final_k": final_k,
                                        "final_retention": final_retention,
                                        "protect_fraction": protect_fraction,
                                        "n_protected": len(protected),
                                        "avg_updates_continuation": first_k
                                        + first_retention
                                        * (final_k - first_k),
                                        **evaluate_keep(truth, keep),
                                    }
                                )
        result = pd.DataFrame(rows)
        result.to_csv(
            ROOT / "outputs" / "protected_lane_sweep.csv", index=False
        )
    log(
        "PROTECT",
        "DONE",
        note="保护通道扫描完成",
        elapsed_s=timer.elapsed,
        rows=len(result),
    )
    print(f"完成 protected_lane_sweep.csv：{len(result)}个协议")


if __name__ == "__main__":
    main()

