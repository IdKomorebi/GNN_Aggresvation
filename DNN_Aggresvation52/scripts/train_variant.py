#!/usr/bin/env python3
"""DNN52 训练：聚合方式(6) × 输入编码强度，单次训练只测准确率。

设计空间：
  聚合 mode ∈ {gcn_noprior, gcn_static, gcn_dynamic, gat_noprior, gat_static, gat_dynamic}
            = {gcn,gat} × {无先验, 全局静态先验, 全局动态先验}
  编码 encoder ∈ {linear_noact(弱), linear(中), mlp(强), mlp_deep(最强)}

用法：train_variant.py --encoder mlp --mode gat_dynamic
输出：outputs/<encoder_tag>/<mode>/{model.pt, per_target_r2.csv, summary.json}
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

# 编码强度档：folder 名（按强度排序）+ encoder 标识
ENCODER_TAG = {
    "linear_noact": "enc1_linear_noact",
    "linear": "enc2_linear",
    "mlp": "enc3_mlp",
    "mlp_deep": "enc4_mlp_deep",
}


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(SCRIPTS.parent / "configs/base.yaml"))
    ap.add_argument("--encoder", required=True, choices=list(ENCODER_TAG))
    ap.add_argument("--mode", required=True,
                    choices=["gcn_noprior", "gcn_static", "gcn_dynamic",
                             "gat_noprior", "gat_static", "gat_dynamic"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["dataset"]["csv_path"] = str(_resolve(cfg["dataset"]["csv_path"]))
    cfg.setdefault("model", {})
    cfg["model"]["input_encoder"] = args.encoder
    cfg["model"]["unified_aggregation"] = args.mode
    device = torch.device(str(cfg.get("runtime", {}).get("device", "cuda:0")))

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)

    data_info = prepare_data(cfg)
    n_nodes = data_info["n_nodes"]
    mt = np.load(_resolve(cfg["correlation"]["cache_dir"]) / "metric_tensor.npy")
    assert mt.shape[0] == n_nodes

    g = cfg.get("graph", {}); m = cfg["model"]
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
        input_encoder=args.encoder,
        input_encoder_dropout=float(m.get("input_encoder_dropout", 0.0)),
        bipartite=use_bipartite,
        unified_aggregation=args.mode,
    ).to(device)

    print(f"[{ENCODER_TAG[args.encoder]}/{args.mode}] encoder={args.encoder} mode={args.mode}")

    res = train_model(model=model, train_data=data_info["train_data"],
                      test_data=data_info["test_data"], data_info=data_info,
                      cfg=cfg, device=device)

    out_dir = ROOT / "outputs" / ENCODER_TAG[args.encoder] / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    r2 = res["per_target_r2"]
    mean_r2 = float(np.mean(list(r2.values())))
    pd.DataFrame(
        [{"confidential_field": k, "test_r2": v} for k, v in r2.items()]
    ).sort_values("test_r2", ascending=False).to_csv(out_dir / "per_target_r2.csv", index=False)

    summary = {
        "encoder": args.encoder, "encoder_tag": ENCODER_TAG[args.encoder],
        "mode": args.mode, "mean_r2_12conf": mean_r2,
        "per_target_r2": r2,
        "final_alpha": {mname: float(v) for mname, v in zip(
            cfg["correlation"]["metrics"], res["final_alpha"])},
        "n_general": data_info["n_general"], "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{ENCODER_TAG[args.encoder]}/{args.mode}] 完成  12-conf 平均 R² = {mean_r2:.4f}")


if __name__ == "__main__":
    main()
