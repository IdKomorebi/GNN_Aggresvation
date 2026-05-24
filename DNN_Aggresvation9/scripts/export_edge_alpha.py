#!/usr/bin/env python3
"""
导出DNN_Aggresvation7已训练run中的逐边相关系数权重。

该脚本不重跑训练，只读取每个run的 model.pt、metric_tensor.npy、
edge_mask.npy 和 metrics.json，然后生成 edge_alpha 诊断文件。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_run(run_dir: Path, temperature: float | None = None) -> None:
    model_path = run_dir / "training" / "model.pt"
    metric_path = run_dir / "relationships" / "metric_tensor.npy"
    edge_mask_path = run_dir / "graph" / "edge_mask.npy"
    meta_path = run_dir / "relationships" / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    if not edge_mask_path.exists():
        raise FileNotFoundError(edge_mask_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    state = torch.load(model_path, map_location="cpu")
    if "edge_beta" not in state:
        print(f"[skip] {run_dir.name}: checkpoint has no edge_beta")
        return

    meta = _load_json(meta_path)
    metrics = meta["metrics"]
    field_names = meta["columns"]
    metric_tensor = np.load(metric_path)
    edge_mask = np.load(edge_mask_path)

    if temperature is None:
        config_path = run_dir / "config_used.yaml"
        temperature = 1.0
        if config_path.exists():
            text = config_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip().startswith("edge_alpha_temperature:"):
                    temperature = float(line.split(":", 1)[1].strip())
                    break

    edge_beta = state["edge_beta"].float()
    edge_alpha = F.softmax(edge_beta / max(float(temperature), 1e-3), dim=-1).numpy()
    edge_weight = np.sum(metric_tensor * edge_alpha, axis=-1) * edge_mask

    out_dir = run_dir / "edge_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_nodes = edge_mask.shape[0]
    valid_edges = edge_mask.astype(bool) & ~np.eye(n_nodes, dtype=bool)
    rows = []
    for target_idx in range(n_nodes):
        for source_idx in range(n_nodes):
            if not valid_edges[target_idx, source_idx]:
                continue
            alpha_values = edge_alpha[target_idx, source_idx]
            dominant_idx = int(np.argmax(alpha_values))
            row = {
                "target_idx": target_idx,
                "target_field": field_names[target_idx],
                "source_idx": source_idx,
                "source_field": field_names[source_idx],
                "edge_weight": float(edge_weight[target_idx, source_idx]),
                "dominant_metric": metrics[dominant_idx],
                "dominant_alpha": float(alpha_values[dominant_idx]),
            }
            for metric_idx, metric_name in enumerate(metrics):
                row[f"alpha_{metric_name}"] = float(alpha_values[metric_idx])
                row[f"corr_{metric_name}"] = float(metric_tensor[target_idx, source_idx, metric_idx])
            rows.append(row)

    pd.DataFrame(rows).sort_values(
        ["target_field", "edge_weight"], ascending=[True, False]
    ).to_csv(out_dir / "edge_alpha_by_edge.csv", index=False)

    valid_alpha = edge_alpha[valid_edges]
    summary_rows = []
    for metric_idx, metric_name in enumerate(metrics):
        values = valid_alpha[:, metric_idx]
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "edge_alpha_summary.csv", index=False)

    prior_rows = []
    for key, value in state.items():
        if key.startswith("att_prior_scales."):
            prior_rows.append(
                {
                    "layer": int(key.rsplit(".", 1)[1]),
                    "raw_value": float(value.item()),
                    "softplus_scale": float(F.softplus(value).item()),
                }
            )
    if prior_rows:
        pd.DataFrame(prior_rows).sort_values("layer").to_csv(
            out_dir / "attention_prior_scales.csv", index=False
        )

    np.save(out_dir / "edge_alpha_tensor.npy", edge_alpha)
    np.save(out_dir / "edge_weight_matrix.npy", edge_weight)
    for metric_idx, metric_name in enumerate(metrics):
        pd.DataFrame(
            edge_alpha[:, :, metric_idx],
            index=field_names,
            columns=field_names,
        ).to_csv(out_dir / f"edge_alpha_matrix_{metric_name}.csv")

    print(f"[ok] {run_dir.name}: wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="要导出的run目录；默认导出outputs下所有run_*目录",
    )
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    run_dirs = args.run_dirs
    if not run_dirs:
        run_dirs = sorted((PROJECT_ROOT / "outputs").glob("run_*"))

    for run_dir in run_dirs:
        export_run(run_dir.resolve(), temperature=args.temperature)


if __name__ == "__main__":
    main()
