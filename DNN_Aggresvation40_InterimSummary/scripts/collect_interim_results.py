#!/usr/bin/env python3
"""Collect DNN40 interim results and draw the three-experiment comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_multi() -> dict[str, float]:
    base = OUT / "graph_multi_window1_fixed_info"
    summary_path = _latest(list(base.glob("*/results/summary.json")))
    if summary_path is None:
        return {}
    return {
        k: float(v)
        for k, v in _load_json(summary_path).get("per_target_r2", {}).items()
    }


def _load_single() -> dict[str, float]:
    base = OUT / "graph_single_window1_fixed_info"
    values: dict[str, float] = {}
    for target_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = _latest(list(target_dir.glob("*/results/summary.json")))
        if summary_path is None:
            continue
        per_target = _load_json(summary_path).get("per_target_r2", {})
        for name, value in per_target.items():
            values[name] = float(value)
    return values


def _load_probe() -> dict[str, float]:
    base = OUT / "dnn_probe_general_only"
    csv_path = _latest(list(base.glob("*/target_probe_results.csv")))
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path)
    return {
        str(row["confidential_field"]): float(row["probe_r2"])
        for _, row in df.iterrows()
    }


def _make_frame() -> pd.DataFrame:
    multi = _load_multi()
    single = _load_single()
    probe = _load_probe()
    targets = sorted(set(multi) | set(single) | set(probe))
    rows = []
    for target in targets:
        rows.append(
            {
                "confidential_field": target,
                "graph_multi_fixed_info_r2": multi.get(target, np.nan),
                "graph_single_fixed_info_r2": single.get(target, np.nan),
                "dnn_probe_general_only_r2": probe.get(target, np.nan),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["graph_single_minus_probe"] = (
            df["graph_single_fixed_info_r2"] - df["dnn_probe_general_only_r2"]
        )
        df["graph_multi_minus_probe"] = (
            df["graph_multi_fixed_info_r2"] - df["dnn_probe_general_only_r2"]
        )
        df = df.sort_values("graph_single_fixed_info_r2", ascending=False)
    return df


def _plot(df: pd.DataFrame) -> None:
    if df.empty:
        raise SystemExit("No results found.")

    plot_df = df.sort_values("graph_single_fixed_info_r2", ascending=True)
    labels = plot_df["confidential_field"].tolist()
    y = np.arange(len(labels))
    height = 0.24

    fig_h = max(6.0, 0.42 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    colors = {
        "graph_multi_fixed_info_r2": "#3B6EA8",
        "graph_single_fixed_info_r2": "#2A9D8F",
        "dnn_probe_general_only_r2": "#E76F51",
    }
    specs = [
        ("graph_multi_fixed_info_r2", "Graph multi, fixed info"),
        ("graph_single_fixed_info_r2", "Graph single, fixed info"),
        ("dnn_probe_general_only_r2", "Fully-connected DNN probe"),
    ]
    for offset, (col, label) in zip([-height, 0.0, height], specs):
        ax.barh(y + offset, plot_df[col], height=height, label=label, color=colors[col])

    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test R2")
    ax.set_title("DNN40 interim comparison: correlation-graph GNN vs fully-connected DNN")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    ax.set_xlim(left=min(-0.1, float(np.nanmin(plot_df[[
        "graph_multi_fixed_info_r2",
        "graph_single_fixed_info_r2",
        "dnn_probe_general_only_r2",
    ]].to_numpy())) - 0.05), right=1.02)
    fig.tight_layout()

    fig.savefig(OUT / "interim_three_experiment_r2.png", dpi=220)
    fig.savefig(OUT / "interim_three_experiment_r2.pdf")
    plt.close(fig)


def main() -> None:
    df = _make_frame()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "interim_three_experiment_comparison.csv", index=False)
    _plot(df)

    summary = {
        "n_targets": int(len(df)),
        "mean_graph_multi_fixed_info_r2": float(df["graph_multi_fixed_info_r2"].mean()),
        "mean_graph_single_fixed_info_r2": float(df["graph_single_fixed_info_r2"].mean()),
        "mean_dnn_probe_general_only_r2": float(df["dnn_probe_general_only_r2"].mean()),
        "mean_graph_single_minus_probe": float(df["graph_single_minus_probe"].mean()),
        "mean_graph_multi_minus_probe": float(df["graph_multi_minus_probe"].mean()),
    }
    with open(OUT / "interim_three_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
