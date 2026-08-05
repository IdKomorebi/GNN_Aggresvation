# -*- coding: utf-8 -*-
"""对 S1/S2 强协同诊断做独立的关键数字与交付物检查。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    detail = pd.read_csv(OUT / "s1_s2_error_decomposition.csv")
    stages = pd.read_csv(OUT / "s1_s2_extreme_protocol_stages.csv")
    grid = pd.read_csv(OUT / "s1_s2_extreme_kgrid_raw.csv")

    extreme = detail[detail.syn3_true > 0.20].copy()
    assert extreme.groupby("K")["indices"].nunique().eq(15).all()
    assert not extreme.duplicated(["indices", "K"]).any()
    residual = (
        extreme.syn_error
        - (extreme.parent_error - extreme.pair_error)
    ).abs()
    assert residual.max() < 1e-10

    k10 = extreme[extreme.K == 10]
    checks = {
        "n_extreme": int(k10["indices"].nunique()),
        "k10_parent_underestimated": int((k10.parent_error < 0).sum()),
        "k10_pair_overestimated": int((k10.pair_error > 0).sum()),
        "k10_parent_more_underfit_than_pair": int(
            (k10.parent_error < k10.pair_error).sum()
        ),
        "k10_s1_conf_matches_true": int(k10.s1_conf_matches_true.sum()),
        "k10_s2_conf_matches_true": int(k10.s2_conf_matches_true.sum()),
        "pure_s1_k10_top60": int(stages.pure_s1_k10_top60.sum()),
        "s1_s2_k10_top60": int(stages.s1_s2_k10_top60.sum()),
        "newly_rescued_by_s2": int(stages.newly_rescued_by_s2.sum()),
        "d79_final": int(stages.d79_final.sum()),
        "d81_final": int(stages.d81_final.sum()),
        "d81_new_hits": int((stages.d81_final & ~stages.d79_final).sum()),
        "d81_lost_baseline_hits": int(
            (stages.d79_final & ~stages.d81_final).sum()
        ),
    }
    expected = {
        "n_extreme": 15,
        "k10_parent_underestimated": 15,
        "k10_pair_overestimated": 1,
        "k10_parent_more_underfit_than_pair": 15,
        "k10_s1_conf_matches_true": 4,
        "k10_s2_conf_matches_true": 0,
        "pure_s1_k10_top60": 8,
        "s1_s2_k10_top60": 12,
        "newly_rescued_by_s2": 4,
        "d79_final": 7,
        "d81_final": 11,
        "d81_new_hits": 4,
        "d81_lost_baseline_hits": 0,
    }
    assert checks == expected, (checks, expected)

    row10 = grid.loc[grid.K == 10].iloc[0]
    means = {
        "parent_true": float(k10.parent_true.mean()),
        "parent_est": float(k10.est.mean()),
        "pair_true": float(k10.pair_true.mean()),
        "pair_est": float(k10.best_pair_est.mean()),
        "syn3_true": float(k10.syn3_true.mean()),
        "syn3_est_same_conf": float(k10.syn3_est.mean()),
        "s2_any": float(k10.s2_any.mean()),
    }
    for column, value in means.items():
        assert np.isclose(row10[column], value, atol=1e-12), column

    artifact = json.loads(
        (ROOT / "artifact_s1_s2_diagnostic.json").read_text(encoding="utf-8")
    )
    assert artifact["surface"] == "report"
    assert len(artifact["manifest"]["charts"]) == 2
    assert len(artifact["manifest"]["tables"]) == 1
    html = ROOT / "report_s1_s2_diagnostic.html"
    assert html.exists() and html.stat().st_size > 100_000

    report = {
        "status": "share_with_caveats",
        "calculation_checks": checks,
        "k10_raw_means": means,
        "max_error_identity_residual": float(residual.max()),
        "artifact": {
            "charts": 2,
            "tables": 1,
            "html_bytes": html.stat().st_size,
            "portable_verification": "structural_only",
        },
        "required_caveats": [
            "极强样本只有15个，其中14个来自历史top候选池。",
            "S2-any在K10时与真实协同目标0/15对齐，属于跨目标代理信号。",
            "当前环境无Chromium，HTML只完成结构验证，未完成浏览器交互验证。",
        ],
    }
    (OUT / "s1_s2_diagnostic_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
