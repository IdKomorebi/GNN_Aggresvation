#!/usr/bin/env python3
"""Dedicated-DNN certificate with inner validation and untouched outer test."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
OUT = ROOT / "outputs"
CERT = OUT / "clean_retrain"
CERT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402


DEFAULT_EPOCHS = 800
PATIENCE = 120
BATCH = 128
VAL_FRAC = 0.15
VAL_SPLIT_SEED = 20260722


class DNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dest = CERT / f"{args.candidate}_seed{args.seed}.json"
    if dest.exists() and not args.force:
        print(f"skip existing {dest}")
        return

    candidates = json.loads((OUT / "candidates.json").read_text())["candidates"]
    if args.candidate not in candidates:
        raise KeyError(args.candidate)
    meta = candidates[args.candidate]

    cfg = yaml.safe_load((R69 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    name2local = {n: i for i, n in enumerate(data["general"])}
    selected_global = gi[[name2local[f] for f in meta["shared"]]]
    development = data["train_data"]
    untouched_test = data["test_data"]

    split_rng = np.random.RandomState(VAL_SPLIT_SEED)
    perm = split_rng.permutation(len(development))
    n_val = int(round(len(development) * VAL_FRAC))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(development[train_idx][:, selected_global], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(development[train_idx][:, ci], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(development[val_idx][:, selected_global], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(development[val_idx][:, ci], dtype=torch.float32, device=device)
    x_test = torch.as_tensor(untouched_test[:, selected_global], dtype=torch.float32, device=device)
    y_test_np = untouched_test[:, ci]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = DNN(len(selected_global), len(ci)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val, best_state, best_epoch, stale = float("inf"), None, -1, 0
    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=train_gen).to(device)
        for start in range(0, len(order), BATCH):
            ix = order[start:start + BATCH]
            opt.zero_grad()
            loss = ((model(x_train[ix]) - y_train[ix]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(((model(x_val) - y_val) ** 2).mean().item())
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    # The untouched outer test is first accessed here, after all model selection is complete.
    with torch.no_grad():
        test_pred = model(x_test).cpu().numpy()
    r2 = per_conf_r2(test_pred, y_test_np)
    result = {
        "candidate": args.candidate,
        "source": meta["source"],
        "k": meta["k"],
        "n_shared": len(meta["shared"]),
        "seed": args.seed,
        "outer_split_seed": 42,
        "outer_development_n": int(len(development)),
        "inner_train_n": int(len(train_idx)),
        "inner_validation_n": int(len(val_idx)),
        "untouched_test_n": int(len(untouched_test)),
        "validation_split_seed": VAL_SPLIT_SEED,
        "max_epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch + 1),
        "best_validation_mse": float(best_val),
        "test_access_stage": "after_retrain_model_selection",
        "candidate_selection_reused_outer_test": True,
        "worst_test_r2": float(r2.max()),
        "mean_test_r2": float(r2.mean()),
        "per_conf_test_r2": {n: float(v) for n, v in zip(data["confidential"], r2)},
        "protected": meta["protected"],
        "shared": meta["shared"],
    }
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["candidate", "seed", "best_epoch", "worst_test_r2", "mean_test_r2"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
