#!/usr/bin/env python3
"""K=0 robustness check for all three MLP and GNN oracle seeds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import GNNOracle, MLPOracle, build_priors


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    n_general, n_conf = data["n_general"], data["n_confidential"]
    test = data["test_data"]
    xte = torch.as_tensor(test[:, gi], dtype=torch.float32, device=DEV)
    yte = test[:, ci]
    name_to_local = {name: i for i, name in enumerate(data["general"])}
    subsets = json.loads((ROOT / "outputs/subsets.json").read_text())

    metric = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    edge = build_edge_mask(metric, top_k=cfg["graph"]["top_k"],
                           threshold=cfg["graph"]["threshold"], symmetrize=True,
                           n_general=n_general, bipartite=True)
    a_gg, prior_cg = build_priors(metric, edge, n_general)

    rows = []
    for arch in ["mlp", "gnn"]:
        for seed in [0, 1, 2]:
            checkpoint = torch.load(ROOT / f"outputs/oracle_{arch}_seed{seed}.pt",
                                    map_location=DEV, weights_only=False)
            if arch == "mlp":
                model = MLPOracle(n_general, n_conf)
            else:
                model = GNNOracle(a_gg, prior_cg, n_conf,
                                  hidden=checkpoint.get("gnn_hidden", 128),
                                  n_layers=checkpoint.get("gnn_layers", 3))
            model.load_state_dict(checkpoint["state"])
            model.to(DEV).eval()
            with torch.no_grad():
                for sid, meta in sorted(subsets.items()):
                    mask = torch.zeros(len(xte), n_general, device=DEV)
                    mask[:, [name_to_local[f] for f in meta["fields"]]] = 1.0
                    values = r2(model(xte, mask).cpu().numpy(), yte)
                    for j, conf in enumerate(data["confidential"]):
                        rows.append({"sid": sid, "source": meta["source"],
                                     "group": meta["group"], "size": meta["size"],
                                     "arch": arch, "seed": seed, "conf": conf,
                                     "r2": float(values[j])})
            print(f"K0 complete {arch} seed={seed}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "outputs/k0_multiseed.csv", index=False)


if __name__ == "__main__":
    main()
