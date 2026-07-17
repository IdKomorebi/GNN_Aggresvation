#!/usr/bin/env python3
"""Evaluate and equally fine-tune one universal oracle architecture over a shard.

Both MLPOracle and GNNOracle receive the same fixed subset mask, the same batches,
the same optimizer, and the same K grid. The output keeps per-confidential R2 so
direct sensitivity and pair/triple synergy can be evaluated separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import GNNOracle, MLPOracle, build_priors


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = [0, 10, 50, 200]
BATCH = 256
LR = 1e-3


def per_conf_r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def build_model(arch: str, checkpoint: dict, cfg: dict, n_general: int, n_conf: int):
    if arch == "mlp":
        model = MLPOracle(n_general, n_conf)
    else:
        metric = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
        graph_cfg = cfg["graph"]
        edge_mask = build_edge_mask(
            metric,
            top_k=graph_cfg["top_k"],
            threshold=graph_cfg["threshold"],
            symmetrize=True,
            n_general=n_general,
            bipartite=True,
        )
        a_gg, prior_cg = build_priors(metric, edge_mask, n_general)
        model = GNNOracle(
            a_gg,
            prior_cg,
            n_conf,
            hidden=checkpoint.get("gnn_hidden", 128),
            n_layers=checkpoint.get("gnn_layers", 3),
        )
    model.load_state_dict(checkpoint["state"])
    return model.to(DEV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "gnn"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general, n_conf = data["n_general"], data["n_confidential"]
    name_to_local = {name: i for i, name in enumerate(data["general"])}
    train, test = data["train_data"], data["test_data"]
    xtr = torch.as_tensor(train[:, general_idx], dtype=torch.float32, device=DEV)
    ytr = torch.as_tensor(train[:, conf_idx], dtype=torch.float32, device=DEV)
    xte = torch.as_tensor(test[:, general_idx], dtype=torch.float32, device=DEV)
    yte = test[:, conf_idx]

    ckpt_path = ROOT / f"outputs/oracle_{args.arch}_seed{args.seed}.pt"
    checkpoint = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    subsets = json.loads((ROOT / "outputs/subsets.json").read_text())
    sids = [sid for i, sid in enumerate(sorted(subsets))
            if i % args.nshard == args.shard]
    if args.limit > 0:
        sids = sids[:args.limit]
    out_dir = ROOT / f"outputs/est/{args.arch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    started = time.time()
    for sid in sids:
        out_path = out_dir / f"{sid}.json"
        if out_path.exists() and not args.overwrite:
            continue
        meta = subsets[sid]
        selected = [name_to_local[name] for name in meta["fields"]]
        mask = torch.zeros(1, n_general, device=DEV)
        mask[0, selected] = 1.0
        mask_tr = mask.expand(len(xtr), -1)
        mask_te = mask.expand(len(xte), -1)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build_model(args.arch, checkpoint, cfg, n_general, n_conf)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        record: dict[int, dict] = {}

        def snap() -> dict:
            model.eval()
            with torch.no_grad():
                r2 = per_conf_r2(model(xte, mask_te).cpu().numpy(), yte)
            return {
                "mean_r2": float(r2.mean()),
                "per_conf_r2": {
                    data["confidential"][j]: float(r2[j]) for j in range(n_conf)
                },
            }

        record[0] = {**snap(), "time": 0.0}
        rng = np.random.RandomState(args.seed)
        step = 0
        start_train = time.time()
        checkpoints = set(KGRID)
        while step < max(KGRID):
            order = rng.permutation(len(xtr))
            for begin in range(0, len(order), BATCH):
                ix = torch.as_tensor(order[begin:begin + BATCH], device=DEV)
                model.train()
                optimizer.zero_grad()
                loss = ((model(xtr[ix], mask_tr[ix]) - ytr[ix]) ** 2).mean()
                loss.backward()
                optimizer.step()
                step += 1
                if step in checkpoints:
                    record[step] = {**snap(), "time": time.time() - start_train}
                if step >= max(KGRID):
                    break

        output = {
            "subset_id": sid,
            "source": meta["source"],
            "group": meta["group"],
            "size": meta["size"],
            "fields": meta["fields"],
            "arch": args.arch,
            "oracle_seed": args.seed,
            "lr": LR,
            "batch": BATCH,
            "by_k": record,
        }
        out_path.write_text(json.dumps(output, ensure_ascii=False) + "\n")
        done += 1
        if done % 20 == 0:
            print(f"{args.arch} shard={args.shard}: new={done}/{len(sids)} elapsed={time.time()-started:.0f}s", flush=True)
    print(f"{args.arch} shard={args.shard}/{args.nshard} complete new={done} elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
