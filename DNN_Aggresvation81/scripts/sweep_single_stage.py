# -*- coding: utf-8 -*-
"""全体先微调到同一个K，再在固定候选预算下全面比较S1+S2。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    KGRID,
    N_ALL,
    evaluate_keep,
    fixed_force_fill,
    fixed_rank_fusion,
    load_scores,
    load_truth,
    score_series,
    top_n,
)
from runlog import Timer, log  # noqa: E402

BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
RESCUES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def add_row(rows, truth, keep, **meta):
    rows.append({**meta, **evaluate_keep(truth, keep)})


def main() -> None:
    log("SINGLE", "START", note="同K固定预算全参数扫描")
    with Timer() as timer:
        scores = load_scores()
        truth = load_truth()
        rows = []
        for k_value in KGRID:
            s1 = score_series(scores, k_value, "s1_syn3")
            s2_any = score_series(scores, k_value, "s2_parent_any")
            s2_aligned = score_series(scores, k_value, "s2_parent_aligned")
            for budget in BUDGETS:
                keep = top_n(s1, round(N_ALL * budget))
                add_row(
                    rows,
                    truth,
                    keep,
                    K=k_value,
                    family="s1_only",
                    s2_variant="none",
                    target_budget=budget,
                    parameter=0.0,
                )
                for rescue in RESCUES:
                    if rescue > budget:
                        continue
                    for label, s2 in (
                        ("any", s2_any),
                        ("aligned", s2_aligned),
                    ):
                        keep = fixed_force_fill(s1, s2, budget, rescue)
                        add_row(
                            rows,
                            truth,
                            keep,
                            K=k_value,
                            family="force_fill",
                            s2_variant=label,
                            target_budget=budget,
                            parameter=rescue,
                        )
                for alpha in ALPHAS:
                    for label, s2 in (
                        ("any", s2_any),
                        ("aligned", s2_aligned),
                    ):
                        keep = fixed_rank_fusion(s1, s2, budget, alpha)
                        add_row(
                            rows,
                            truth,
                            keep,
                            K=k_value,
                            family="rank_fusion",
                            s2_variant=label,
                            target_budget=budget,
                            parameter=alpha,
                        )

                # 复现Claude的“额外加预算”版本，和固定预算结果分开。
                for rescue in RESCUES:
                    base = top_n(s1, round(N_ALL * budget))
                    extra = top_n(s2_any, round(N_ALL * rescue))
                    add_row(
                        rows,
                        truth,
                        base | extra,
                        K=k_value,
                        family="union_add",
                        s2_variant="any",
                        target_budget=budget,
                        parameter=rescue,
                    )

        result = pd.DataFrame(rows)
        result.to_csv(ROOT / "outputs" / "single_stage_sweep.csv", index=False)

    log(
        "SINGLE",
        "DONE",
        note="完成K×预算×S2参数扫描",
        elapsed_s=timer.elapsed,
        rows=len(result),
    )
    print(f"完成 single_stage_sweep.csv：{len(result)}个协议")


if __name__ == "__main__":
    main()

