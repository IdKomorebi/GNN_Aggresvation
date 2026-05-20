from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import prepare_data, train_fixed_gnn
from src.utils import ensure_dir, load_yaml, save_json, set_seed, torch_device


def _latest_run() -> Path:
    output_root = PROJECT_ROOT / "outputs"
    runs = sorted([p for p in output_root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found under {output_root}")
    return runs[-1]


def _patch_dataset_path(cfg: dict) -> dict:
    path = Path(str(cfg["dataset"]["processed_csv"]))
    resolved = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if resolved.exists():
        return cfg

    fallback = REPO_ROOT / "data" / "Processed" / "pjm_rto_hourly_2025_cleaned.csv"
    if fallback.exists():
        cfg = dict(cfg)
        cfg["dataset"] = dict(cfg["dataset"])
        cfg["dataset"]["processed_csv"] = str(fallback)
        return cfg

    raise FileNotFoundError(f"Configured dataset path does not exist: {resolved}")


def _make_folds(anchor_labels: pd.DataFrame, fold_size: int) -> list[list[str]]:
    ordered = anchor_labels.sort_values("q_label", ascending=False)["field"].astype(str).tolist()
    folds: list[list[str]] = []
    for start in range(0, len(ordered), fold_size):
        folds.append(ordered[start : start + fold_size])
    return folds


def run_validation(run_dir: Path, fold_size: int = 2) -> Path:
    cfg = _patch_dataset_path(load_yaml(run_dir / "config_used.yaml"))
    seed = int(cfg.get("runtime", {}).get("random_state", 42))
    set_seed(seed)
    device = torch_device(str(cfg.get("runtime", {}).get("device", "cpu")))

    data = prepare_data(cfg)
    adjacency = np.load(run_dir / "graph" / "adjacency.npy")
    anchor_labels = pd.read_csv(run_dir / "labels" / "general_anchor_labels.csv")
    anchor_labels["field"] = anchor_labels["field"].astype(str)
    if "use_in_gnn" in anchor_labels.columns:
        anchor_labels = anchor_labels[anchor_labels["use_in_gnn"].astype(bool)].copy()
    anchor_labels = anchor_labels.sort_values("q_label", ascending=False).reset_index(drop=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_dir = ensure_dir(run_dir / "validation" / f"heldout_anchor_{stamp}")
    folds = _make_folds(anchor_labels, fold_size=max(1, int(fold_size)))

    heldout_rows = []
    distribution_rows = []
    for fold_idx, heldout_fields in enumerate(folds, 1):
        set_seed(seed + fold_idx)
        train_labels = anchor_labels[~anchor_labels["field"].isin(heldout_fields)].copy()
        fold_dir = ensure_dir(validation_dir / f"fold_{fold_idx:02d}")
        scores = train_fixed_gnn(data, adjacency, train_labels, cfg, fold_dir, device)
        score_by_field = scores.set_index("field")

        for _, row in anchor_labels[anchor_labels["field"].isin(heldout_fields)].iterrows():
            field = str(row["field"])
            score = float(score_by_field.loc[field, "score"])
            q_label = float(row["q_label"])
            heldout_rows.append(
                {
                    "fold": fold_idx,
                    "field": field,
                    "q_label": q_label,
                    "heldout_score": score,
                    "error": score - q_label,
                    "abs_error": abs(score - q_label),
                }
            )

        general_scores = scores[scores["field_type"] == "general"].copy()
        train_anchor_set = set(train_labels["field"].astype(str))
        heldout_set = set(heldout_fields)
        non_anchor_scores = general_scores[
            ~general_scores["field"].astype(str).isin(train_anchor_set | heldout_set)
        ]["score"]
        heldout_scores = general_scores[general_scores["field"].astype(str).isin(heldout_set)]["score"]
        train_anchor_scores = general_scores[general_scores["field"].astype(str).isin(train_anchor_set)]["score"]

        distribution_rows.append(
            {
                "fold": fold_idx,
                "heldout_fields": ", ".join(heldout_fields),
                "train_anchor_count": int(len(train_anchor_scores)),
                "heldout_anchor_count": int(len(heldout_scores)),
                "non_anchor_general_count": int(len(non_anchor_scores)),
                "train_anchor_mean_score": float(train_anchor_scores.mean()),
                "heldout_anchor_mean_score": float(heldout_scores.mean()),
                "non_anchor_general_mean_score": float(non_anchor_scores.mean()),
                "non_anchor_general_median_score": float(non_anchor_scores.median()),
                "non_anchor_general_p90_score": float(non_anchor_scores.quantile(0.9)),
                "non_anchor_general_above_0_8": int((non_anchor_scores >= 0.8).sum()),
            }
        )

    heldout_df = pd.DataFrame(heldout_rows)
    dist_df = pd.DataFrame(distribution_rows)
    heldout_df.to_csv(validation_dir / "heldout_anchor_predictions.csv", index=False, encoding="utf-8-sig")
    dist_df.to_csv(validation_dir / "fold_score_distributions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_run": str(run_dir),
        "fold_size": int(fold_size),
        "folds": len(folds),
        "heldout_count": int(len(heldout_df)),
        "heldout_mae": float(heldout_df["abs_error"].mean()),
        "heldout_rmse": float(np.sqrt(np.mean(np.square(heldout_df["error"])))),
        "heldout_mean_error": float(heldout_df["error"].mean()),
        "heldout_overprediction_rate": float((heldout_df["error"] > 0).mean()),
        "mean_non_anchor_general_score": float(dist_df["non_anchor_general_mean_score"].mean()),
        "mean_non_anchor_general_p90_score": float(dist_df["non_anchor_general_p90_score"].mean()),
        "mean_non_anchor_general_above_0_8": float(dist_df["non_anchor_general_above_0_8"].mean()),
    }
    save_json(validation_dir / "summary.json", summary)
    print(f"Validation directory: {validation_dir}")
    print(pd.Series(summary).to_string())
    return validation_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out General anchor validation for DNN_Aggresvation2.")
    parser.add_argument("--run-dir", default=None, help="Existing DNN_Aggresvation2 run directory. Defaults to latest.")
    parser.add_argument("--fold-size", type=int, default=2, help="Number of General anchors held out per fold.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else _latest_run()
    run_validation(run_dir, fold_size=args.fold_size)


if __name__ == "__main__":
    main()
