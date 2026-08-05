# -*- coding: utf-8 -*-
"""81号输出一致性和关键结论验证。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
N_ALL = 13244


def main() -> None:
    report = {"status": "ready", "checks": {}}
    cube = pd.read_parquet(OUT / "score_cube.parquet")
    assert len(cube) == N_ALL * 6
    assert cube.groupby(["i", "j", "k"]).ngroups == N_ALL
    assert set(cube.K.unique()) == {0, 1, 5, 10, 25, 50}
    report["checks"]["score_cube"] = {
        "rows": len(cube),
        "triples": N_ALL,
        "kgrid": sorted(cube.K.unique().tolist()),
    }

    single = pd.read_csv(OUT / "single_stage_sweep.csv")
    staged = pd.read_csv(OUT / "staged_sweep.csv")
    protected = pd.read_csv(OUT / "protected_lane_sweep.csv")
    assert len(single) == 2316
    assert len(staged) == 54720
    assert len(protected) == 3584
    for frame in (single, staged, protected):
        metric_columns = [
            column
            for column in frame.columns
            if column.endswith("_recall") or column.endswith("_precision")
        ]
        assert frame[metric_columns].min().min() >= 0
        assert frame[metric_columns].max().max() <= 1
    report["checks"]["sweep_rows"] = {
        "single": len(single),
        "staged": len(staged),
        "protected": len(protected),
    }

    summary = pd.read_csv(OUT / "selected_protocols.csv")
    assert set(summary.protocol) == {
        "D79_S1_baseline",
        "D81_tune_F1_selected",
        "D81_tune_recall_only",
    }
    base = summary[summary.protocol == "D79_S1_baseline"].iloc[0]
    selected = summary[summary.protocol == "D81_tune_F1_selected"].iloc[0]
    # 相同成本、相同最终候选数；参数选择不能使用test。
    assert base.avg_updates_continuation == selected.avg_updates_continuation == 19
    assert selected.tune_f1 > base.tune_f1
    assert selected.test_recall >= base.test_recall
    assert selected.test_precision >= base.test_precision
    assert selected.all_b_gt20_raw_hit > base.all_b_gt20_raw_hit
    report["checks"]["selected_vs_baseline"] = {
        "cost_updates": float(selected.avg_updates_continuation),
        "test_recall_baseline": float(base.test_recall),
        "test_recall_selected": float(selected.test_recall),
        "test_precision_baseline": float(base.test_precision),
        "test_precision_selected": float(selected.test_precision),
        "extreme_raw_baseline": int(base.all_b_gt20_raw_hit),
        "extreme_raw_selected": int(selected.all_b_gt20_raw_hit),
    }

    boot = json.loads((OUT / "bootstrap_summary.json").read_text())
    assert boot["n_bootstrap"] == 5000
    report["checks"]["bootstrap"] = boot
    figures = sorted((ROOT / "figures").glob("*.png"))
    assert len(figures) == 4 and all(path.stat().st_size > 20_000 for path in figures)
    report["checks"]["figures"] = [path.name for path in figures]

    (OUT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

