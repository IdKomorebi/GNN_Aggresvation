#!/usr/bin/env python3
"""Train one dedicated GCN attacker for a D67/D68 subset.

The architecture and split exactly follow D60/D67/D68. D60 random-subset GCN
truth is reused rather than rerun. Output is local to D69 so prior artifacts are
never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 400
PATIENCE = 120


def per_conf_r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def train_model(model, xtr, ytr, xte, yte):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best = float("inf")
    best_state = None
    patience = 0
    n = len(xtr)
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, device=DEV)
        for start in range(0, n, 128):
            ix = order[start:start + 128]
            opt.zero_grad()
            loss = ((model(xtr[ix]) - ytr[ix]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(((model(xte).cpu().numpy() - yte) ** 2).mean())
        if val < best:
            best = val
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    subsets = json.loads((ROOT / "outputs/subsets.json").read_text())
    meta = subsets[args.subset_id]
    if meta["source"] == "d60_random":
        raise ValueError("D60 GCN truth already exists and must be reused")

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
    metric = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    name_to_local = {name: i for i, name in enumerate(data["general"])}
    selected_local = [name_to_local[name] for name in meta["fields"]]
    selected_global = general_idx[selected_local]
    n_general = len(selected_global)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    selected_nodes = np.concatenate([selected_global, conf_idx])
    metric_sub = metric[np.ix_(selected_nodes, selected_nodes)]
    conf_new = list(range(n_general, n_general + len(conf_idx)))
    graph_cfg = cfg["graph"]
    model_cfg = cfg["model"]
    edge_mask = build_edge_mask(
        metric_sub,
        top_k=min(int(graph_cfg["top_k"]), max(1, n_general)),
        threshold=float(graph_cfg["threshold"]),
        symmetrize=True,
        n_general=n_general,
        bipartite=True,
    )

    train = data["train_data"][:, selected_nodes]
    test = data["test_data"][:, selected_nodes]
    xtr_np, ytr_np = _build_windowed_samples(train, conf_new, 1)
    xte_np, yte_np = _build_windowed_samples(test, conf_new, 1)
    xtr = torch.as_tensor(xtr_np, dtype=torch.float32, device=DEV)
    ytr = torch.as_tensor(ytr_np, dtype=torch.float32, device=DEV)
    xte = torch.as_tensor(xte_np, dtype=torch.float32, device=DEV)

    model = InferenceDrivenGNN(
        metric_tensor=metric_sub,
        edge_mask=edge_mask,
        n_nodes=n_general + len(conf_idx),
        n_general=n_general,
        confidential_indices=conf_new,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        input_dim=1,
        architecture=model_cfg["architecture"],
        attention_dropout=model_cfg["attention_dropout"],
        attention_dim=model_cfg["attention_dim"],
        attention_temperature=model_cfg["attention_temperature"],
        edge_alpha_temperature=model_cfg["edge_alpha_temperature"],
        alpha_init_std=model_cfg["alpha_init_std"],
        prior_scale_init=model_cfg["prior_scale_init"],
        gate_bias_init=model_cfg["gate_bias_init"],
        prior_log_eps=model_cfg["prior_log_eps"],
        target_specific_heads=True,
        input_encoder="mlp",
        bipartite=True,
        unified_aggregation="gcn_dynamic",
    ).to(DEV)
    model = train_model(model, xtr, ytr, xte, yte_np)
    with torch.no_grad():
        pred = model(xte).cpu().numpy()
    r2 = per_conf_r2(pred, yte_np)

    result = {
        "subset_id": args.subset_id,
        "source": meta["source"],
        "group": meta["group"],
        "size": meta["size"],
        "fields": meta["fields"],
        "struct": "gcn",
        "seed": args.seed,
        "mean_r2": float(r2.mean()),
        "per_conf_r2": {
            data["confidential"][j]: float(r2[j]) for j in range(len(r2))
        },
    }
    out = ROOT / "outputs/retrain_gcn"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.subset_id}_gcn_seed{args.seed}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"[{args.subset_id}/gcn/seed{args.seed}/size={meta['size']}] mean R2={r2.mean():.4f}")


if __name__ == "__main__":
    main()
