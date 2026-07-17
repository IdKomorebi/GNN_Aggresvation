#!/usr/bin/env python3
"""DNN55：分裂设计 general→general 用 GCN，general→confidential 用经典 or transformer GAT。

固定：unified_aggregation=none（分裂设计）、G-G = GCN+静态先验（不变）、mlp 编码、seed 42。
变量：
  --conf_variant {transformer, classic}   confidential 侧注意力形式
  --arch {gated, no_gate}                  先验混合：gated=动态门控 / no_gate=静态

输出：outputs/<conf_variant>_<arch>/{model.pt, per_target_r2.csv, summary.json}
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

ARCH = {"gated": "hybrid_bilinear_gated", "no_gate": "hybrid_bilinear_no_gate"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_variant", required=True, choices=["transformer", "classic"])
    ap.add_argument("--arch", required=True, choices=["gated", "no_gate"])
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    device = torch.device(str(cfg.get("runtime", {}).get("device", "cuda:0")))
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)

    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    g = cfg["graph"]; m = cfg["model"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"],
                                symmetrize=True, n_general=di["n_general"], bipartite=True)

    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"],
        n_general=di["n_general"], confidential_indices=di["confidential_indices"],
        hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"],
        input_dim=1, architecture=ARCH[args.arch], attention_dropout=m["attention_dropout"],
        attention_dim=m["attention_dim"], attention_temperature=m["attention_temperature"],
        edge_alpha_temperature=m["edge_alpha_temperature"], alpha_init_std=m["alpha_init_std"],
        prior_scale_init=m["prior_scale_init"], gate_bias_init=m["gate_bias_init"],
        prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
        input_encoder="mlp", bipartite=True,
        unified_aggregation="none",                 # 分裂设计：G-G GCN + G→C 注意力
        conf_gat_variant=args.conf_variant,
    ).to(device)

    tag = f"{args.conf_variant}_{args.arch}"
    print(f"[{tag}] G-G=GCN(+静态先验) | G→C={args.conf_variant} GAT | 先验={args.arch}")

    res = train_model(model=model, train_data=di["train_data"], test_data=di["test_data"],
                      data_info=di, cfg=cfg, device=device)

    out_dir = ROOT / "outputs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    r2 = res["per_target_r2"]; mean_r2 = float(np.mean(list(r2.values())))
    pd.DataFrame([{"confidential_field": k, "test_r2": v} for k, v in r2.items()]
                 ).sort_values("test_r2", ascending=False).to_csv(out_dir / "per_target_r2.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {"tag": tag, "conf_variant": args.conf_variant, "arch": args.arch,
         "mean_r2_12conf": mean_r2, "per_target_r2": r2, "seed": seed}, indent=2, ensure_ascii=False))
    print(f"[{tag}] 完成  12-conf 平均 R² = {mean_r2:.4f}")


if __name__ == "__main__":
    main()
