#!/usr/bin/env python3
"""Collect DNN45 ablation results into comparison CSV files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUNING_DIR = PROJECT_ROOT / "outputs" / "tuning"
OUT_DIR = PROJECT_ROOT / "outputs" / "_comparison"


def latest_run(exp_dir: Path) -> Path | None:
    runs = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    target_rows = []

    for exp_dir in sorted(TUNING_DIR.glob("ab*")):
        run_dir = latest_run(exp_dir)
        if run_dir is None:
            continue

        summary_path = run_dir / "results" / "summary.json"
        r2_path = run_dir / "training" / "per_target_r2.csv"
        config_path = run_dir / "config_used.yaml"
        if not summary_path.exists() or not r2_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        r2_df = pd.read_csv(r2_path)
        mean_r2 = float(r2_df["test_r2"].mean())
        price_mask = r2_df["confidential_field"].str.contains(
            "price|lmp|loss", case=False, regex=True
        )
        price_mean_r2 = float(r2_df.loc[price_mask, "test_r2"].mean())
        non_price_mean_r2 = float(r2_df.loc[~price_mask, "test_r2"].mean())

        summary_rows.append(
            {
                "experiment": exp_dir.name,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "mean_r2": mean_r2,
                "price_like_mean_r2": price_mean_r2,
                "non_price_mean_r2": non_price_mean_r2,
                "best_test_loss": summary.get("best_test_loss"),
                "final_test_mse_targets": summary.get("final_test_mse_targets"),
                "architecture": summary.get("architecture"),
                "config": str(config_path.relative_to(PROJECT_ROOT)),
            }
        )

        for row in r2_df.to_dict("records"):
            target_rows.append(
                {
                    "experiment": exp_dir.name,
                    "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                    **row,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_r2", ascending=False)
    target_df = pd.DataFrame(target_rows)

    summary_df.to_csv(OUT_DIR / "ablation_summary_multi.csv", index=False)
    target_df.to_csv(OUT_DIR / "ablation_per_target_multi.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'ablation_summary_multi.csv'}")
    print(f"Saved: {OUT_DIR / 'ablation_per_target_multi.csv'}")


if __name__ == "__main__":
    main()
