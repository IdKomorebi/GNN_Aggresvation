"""
汇总DNN_Aggresvation6多实验结果。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_TARGETS = [
    "congestion_price_da",
    "congestion_price_rt",
    "total_losses",
    "marginal_loss_price_da",
    "da_as_total_mw_thirty_minutes_reserve",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DNN_Aggresvation6 runs")
    parser.add_argument("--outputs", default="DNN_Aggresvation6/outputs")
    args = parser.parse_args()

    output_root = Path(args.outputs).resolve()
    rows = []
    for summary_path in sorted(output_root.glob("run_*/results/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        row = {
            "experiment": summary.get("experiment"),
            "run_dir": summary.get("run_dir"),
            "architecture": summary.get("architecture"),
            "best_test_loss_targets": summary.get("best_test_loss"),
            "final_test_mse_targets": summary.get("final_test_mse_targets"),
            "final_test_mse_all": summary.get("final_test_mse_all"),
            "final_test_mse_excluded": summary.get("final_test_mse_excluded"),
            "excluded_targets": ",".join(summary.get("excluded_target_names", [])),
        }
        per_target = summary.get("per_target_r2", {})
        for target in KEY_TARGETS:
            row[f"r2_{target}"] = per_target.get(target)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["experiment", "run_dir"])
    output_path = output_root / "comparison_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
