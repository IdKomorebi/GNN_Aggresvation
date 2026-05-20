from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    build_fixed_graph,
    compute_anchor_labels,
    maybe_plot,
    prepare_data,
    select_general_anchors,
    train_fixed_gnn,
)
from src.utils import ensure_dir, load_yaml, make_run_dir, save_json, save_yaml, set_seed, torch_device


def _is_completed_pretrain_run(path: Path) -> bool:
    return (
        (path / "relationships" / "metric_tensor.npy").exists()
        and (path / "relationships" / "metrics.json").exists()
        and (path / "pretrain" / "global_metric_weights.csv").exists()
    )


def _latest_completed_pretrain_run() -> Path:
    runs = sorted([p for p in (PROJECT_ROOT / "outputs").glob("run_*") if p.is_dir()], reverse=True)
    for run in runs:
        if _is_completed_pretrain_run(run):
            return run
    raise FileNotFoundError("No completed run with reusable relationship tensor and global metric weights was found.")


def run_from_pretrained(config_path: str | Path, source_run: str | Path | None = None) -> Path:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("runtime", {}).get("random_state", 42)))
    device = torch_device(str(cfg.get("runtime", {}).get("device", "cpu")))

    source = Path(source_run).resolve() if source_run else _latest_completed_pretrain_run()
    if not _is_completed_pretrain_run(source):
        raise FileNotFoundError(f"Source run is missing reusable artifacts: {source}")

    run_dir = make_run_dir(cfg["dataset"].get("output_root", "outputs"))
    save_yaml(run_dir / "config_used.yaml", cfg)
    save_json(run_dir / "reuse_source_run.json", {"source_run": str(source)})
    print(f"Run directory: {run_dir}")
    print(f"Reusing pretraining artifacts from: {source}")

    data = prepare_data(cfg)
    save_json(
        run_dir / "field_split.json",
        {
            "columns": data.columns,
            "confidential": data.confidential,
            "general": data.general,
            "general_count": len(data.general),
            "confidential_count": len(data.confidential),
        },
    )

    rel_dir = ensure_dir(run_dir / "relationships")
    metric_tensor = np.load(source / "relationships" / "metric_tensor.npy")
    shutil.copy2(source / "relationships" / "metric_tensor.npy", rel_dir / "metric_tensor.npy")
    shutil.copy2(source / "relationships" / "metrics.json", rel_dir / "metrics.json")
    with (source / "relationships" / "metrics.json").open("r", encoding="utf-8") as f:
        metrics_payload = json.load(f)
    metrics = list(metrics_payload["metrics"])

    pretrain_dir = ensure_dir(run_dir / "pretrain")
    shutil.copy2(source / "pretrain" / "global_metric_weights.csv", pretrain_dir / "global_metric_weights.csv")
    source_weights_json = source / "pretrain" / "correlation_weights.json"
    if source_weights_json.exists():
        shutil.copy2(source_weights_json, pretrain_dir / "correlation_weights.json")
    global_alpha = pd.read_csv(source / "pretrain" / "global_metric_weights.csv")["global_alpha"].to_numpy(dtype=np.float32)

    graph_dir = ensure_dir(run_dir / "graph")
    adjacency = build_fixed_graph(data, metric_tensor, metrics, global_alpha, cfg, graph_dir)

    labels_dir = ensure_dir(run_dir / "labels")
    anchors = select_general_anchors(data, metric_tensor, global_alpha, cfg, labels_dir)
    save_json(
        labels_dir / "selected_general_anchors.json",
        {
            "high_anchors": anchors.loc[anchors["anchor_role"] == "high", "field"].astype(str).tolist(),
            "low_anchors": anchors.loc[anchors["anchor_role"] == "low", "field"].astype(str).tolist(),
            "anchors": anchors["field"].astype(str).tolist(),
        },
    )
    anchor_labels = compute_anchor_labels(data, anchors, cfg, device, labels_dir, probe_cache_dirs=[source / "labels"])

    gnn_dir = ensure_dir(run_dir / "gnn")
    train_fixed_gnn(data, adjacency, anchor_labels, cfg, gnn_dir, device)
    maybe_plot(run_dir)
    print(f"Finished: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse pretrained relationship weights and rerun anchor/GNN stages.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "data2025_v2_fixed_edge.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--source-run",
        default=None,
        help="Existing run with relationships/metric_tensor.npy and pretrain/global_metric_weights.csv.",
    )
    args = parser.parse_args()
    run_from_pretrained(args.config, args.source_run)


if __name__ == "__main__":
    main()
