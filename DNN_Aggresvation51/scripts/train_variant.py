#!/usr/bin/env python3
"""DNN51 聚合消融——精简训练脚本（只测准确率）。

验证当前「分裂聚合」（G-G GCN+先验 / G→C 门控 GAT+先验）的设计必要性：
把整张图换成单一聚合方式，看 12 个 confidential 的平均 test R² 相比 0.86xx 如何。

model.unified_aggregation ∈ {none, gcn_noprior, gcn_prior, gat_noprior, gat_prior}
输出：outputs/<exp_name>/{model.pt, per_target_r2.csv, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT))

from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import train_model


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["dataset"]["csv_path"] = str(_resolve(cfg["dataset"]["csv_path"]))
    exp = str(cfg.get("experiment", {}).get("name", Path(args.config).stem))
    device = torch.device(str(cfg.get("runtime", {}).get("device", "cuda:0")))

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)

    data_info = prepare_data(cfg)
    n_nodes = data_info["n_nodes"]

    # 复用相关性缓存（全图字段一致，5 个变体共享）
    cache = _resolve(cfg["correlation"]["cache_dir"]) / "metric_tensor.npy"
    mt = np.load(cache)
    assert mt.shape[0] == n_nodes, f"cache 节点数 {mt.shape[0]} != {n_nodes}"

    g = cfg.get("graph", {}); m = cfg.get("model", {})
    use_bipartite = bool(m.get("bipartite", False))
    edge_mask = build_edge_mask(
        mt, top_k=int(g.get("top_k", 10)), threshold=float(g.get("threshold", 0.08)),
        symmetrize=bool(g.get("symmetrize", True)),
        selection=str(g.get("selection", "threshold_or_topk")),
        min_k=None if g.get("min_k") is None else int(g["min_k"]),
        max_k=None if g.get("max_k") is None else int(g["max_k"]),
        n_general=data_info["n_general"], bipartite=use_bipartite,
    )
    win = int(m.get("window_size", 1))
    unified = str(m.get("unified_aggregation", "none"))

    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=n_nodes,
        n_general=data_info["n_general"], confidential_indices=data_info["confidential_indices"],
        hidden_dim=int(m.get("hidden_dim", 32)), num_layers=int(m.get("num_layers", 2)),
        dropout=float(m.get("dropout", 0.05)), input_dim=win,
        architecture=str(m.get("architecture", "hybrid_bilinear_gated")),
        attention_dropout=float(m.get("attention_dropout", 0.05)),
        attention_dim=int(m.get("attention_dim", m.get("hidden_dim", 32))),
        attention_temperature=float(m.get("attention_temperature", 1.0)),
        edge_alpha_temperature=float(m.get("edge_alpha_temperature", 1.0)),
        alpha_init_std=float(m.get("alpha_init_std", 0.0)),
        prior_scale_init=float(m.get("prior_scale_init", 1.0)),
        gate_bias_init=float(m.get("gate_bias_init", -1.0)),
        prior_log_eps=float(m.get("prior_log_eps", 1e-4)),
        target_specific_heads=bool(m.get("target_specific_heads", True)),
        input_encoder=str(m.get("input_encoder", "linear")),
        input_encoder_dropout=float(m.get("input_encoder_dropout", 0.0)),
        bipartite=use_bipartite,
        unified_aggregation=unified,
    ).to(device)

    print(f"[{exp}] unified_aggregation={unified}  bipartite={use_bipartite}  "
          f"layers={m.get('num_layers')}  hidden={m.get('hidden_dim')}")

    res = train_model(model=model, train_data=data_info["train_data"],
                      test_data=data_info["test_data"], data_info=data_info,
                      cfg=cfg, device=device)

    out_dir = ROOT / "outputs" / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    r2 = res["per_target_r2"]
    mean_r2 = float(np.mean(list(r2.values())))
    pd.DataFrame(
        [{"confidential_field": k, "test_r2": v} for k, v in r2.items()]
    ).sort_values("test_r2", ascending=False).to_csv(out_dir / "per_target_r2.csv", index=False)

    summary = {
        "experiment": exp,
        "unified_aggregation": unified,
        "mean_r2_12conf": mean_r2,
        "per_target_r2": r2,
        "n_general": data_info["n_general"],
        "n_confidential": data_info["n_confidential"],
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{exp}] 完成  12-conf 平均 R² = {mean_r2:.4f}")


if __name__ == "__main__":
    main()
