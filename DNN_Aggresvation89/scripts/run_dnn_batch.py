#!/usr/bin/env python3
"""批量训练专用 MLP；训练内 validation 早停，最终只在 audit 半集评估。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402

SPLIT_SEED = 880725
EPOCHS = 400
PATIENCE = 120


class DNN(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


def per_conf_r2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    ss = ((truth - pred) ** 2).sum(0)
    st = ((truth - truth.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1.0 - ss / st, 0.0, None)


def fit_one(
    indices: list[int],
    seed: int,
    device: torch.device,
    train_data: np.ndarray,
    audit_data: np.ndarray,
    general_idx: np.ndarray,
    conf_idx: np.ndarray,
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = train_data[:, general_idx[indices]]
    y = train_data[:, conf_idx]
    xa = audit_data[:, general_idx[indices]]
    ya = audit_data[:, conf_idx]

    perm = np.random.RandomState(20260725).permutation(len(x))
    nv = max(round(0.15 * len(x)), 1)
    vi, ti = perm[:nv], perm[nv:]
    xt = torch.as_tensor(x[ti], dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[ti], dtype=torch.float32, device=device)
    xv = torch.as_tensor(x[vi], dtype=torch.float32, device=device)
    yv = torch.as_tensor(y[vi], dtype=torch.float32, device=device)
    x_a = torch.as_tensor(xa, dtype=torch.float32, device=device)

    model = DNN(len(indices), len(conf_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best, state, bad, best_epoch = float("inf"), None, 0, 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for k in range(0, len(xt), 128):
            ix = order[k : k + 128]
            opt.zero_grad()
            loss = ((model(xt[ix]) - yt[ix]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(((model(xv) - yv) ** 2).mean())
        if val < best:
            best, state, bad, best_epoch = val, deepcopy(model.state_dict()), 0, epoch + 1
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(x_a).cpu().numpy()
    return per_conf_r2(pred, ya), best_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    train_data = data["train_data"]
    held = data["test_data"]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(held))
    audit = held[perm[len(perm) // 2 :]]
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    conf_names = list(data["confidential"])

    subsets = json.loads((ROOT / "outputs/subsets.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "outputs/retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_items = list(subsets.items())
    all_items = [
        item
        for pos, item in enumerate(all_items)
        if pos % args.num_shards == args.shard_index
    ]
    pending = [
        (sid, spec)
        for sid, spec in all_items
        if not (out_dir / f"{sid}_seed{args.seed}.json").exists()
    ]
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"device={device}; train={len(train_data)} audit={len(audit)}; "
        f"shard={args.shard_index}/{args.num_shards}; pending={len(pending)}",
        flush=True,
    )

    t0 = time.perf_counter()
    for pos, (sid, spec) in enumerate(pending, 1):
        started = time.perf_counter()
        r2, epoch = fit_one(
            spec["indices"],
            args.seed,
            device,
            train_data,
            audit,
            gi,
            ci,
        )
        payload = {
            "subset_id": sid,
            "indices": spec["indices"],
            "size": spec["size"],
            "seed": args.seed,
            "best_epoch": epoch,
            "evaluation_split": "audit",
            "per_conf_r2": {name: float(r2[i]) for i, name in enumerate(conf_names)},
        }
        (out_dir / f"{sid}_seed{args.seed}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[{pos}/{len(pending)}] {sid} size={spec['size']} "
            f"epoch={epoch} meanR2={r2.mean():.4f} "
            f"{time.perf_counter()-started:.1f}s",
            flush=True,
        )
    print(f"全部完成，总用时 {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
