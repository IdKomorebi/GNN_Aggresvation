# -*- coding: utf-8 -*-
"""参数量匹配：普通宽 MLP vs 显式二阶特征 MLP。"""
from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R82 = REPO / "DNN_Aggresvation82"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R82 / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from src.oracle import MLPOracle  # noqa: E402
from samplers import local234_uniform10  # noqa: E402
from feature_oracle import Poly2Oracle, poly2_stats  # noqa: E402
from runlog import log  # noqa: E402
from train_tail_oracle import load_data, per_conf_r2  # noqa: E402

BATCH = 256
LR = 1e-3
WD = 5e-4
EPOCHS = 400
PATIENCE = 60
N_VAL_MASKS = 8


def build_model(
    kind: str,
    n_general: int,
    n_conf: int,
    stats: dict[str, np.ndarray],
) -> torch.nn.Module:
    if kind == "poly2":
        return Poly2Oracle(n_general, n_conf, **stats)
    # hidden=432 约 42 万参数，略多于 poly2 的约 41 万，容量对照偏向 raw MLP。
    return MLPOracle(n_general, n_conf, hidden=432)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("poly2", "rawwide"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    log("FEATURE-TRAIN", "START", f"{args.kind}/seed{args.seed}")
    data = load_data()
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    stats = poly2_stats(data["train_data"][:, general_idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(
        data["train_data"][:, general_idx], dtype=torch.float32, device=device
    )
    y_train = torch.as_tensor(
        data["train_data"][:, conf_idx], dtype=torch.float32, device=device
    )
    x_test = torch.as_tensor(
        data["test_data"][:, general_idx], dtype=torch.float32, device=device
    )
    y_test = data["test_data"][:, conf_idx]

    permutation = np.random.RandomState(args.seed).permutation(len(x_train))
    n_val = max(1, int(len(permutation) * 0.15))
    val_idx = torch.as_tensor(permutation[:n_val], device=device)
    train_idx = torch.as_tensor(permutation[n_val:], device=device)
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(args.kind, n_general, n_conf, stats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    val_masks = [
        torch.as_tensor(
            local234_uniform10(len(val_idx), n_general, val_rng),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(N_VAL_MASKS)
    ]

    best = float("inf")
    best_state = None
    stale = 0
    for epoch in range(EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for begin in range(0, len(order), BATCH):
            idx = order[begin : begin + BATCH]
            masks = torch.as_tensor(
                local234_uniform10(len(idx), n_general, mask_rng),
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            loss = ((model(x_train[idx], masks) - y_train[idx]) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(
                np.mean(
                    [
                        ((model(x_val, mask) - y_val) ** 2).mean().item()
                        for mask in val_masks
                    ]
                )
            )
        if val_loss < best - 1e-6:
            best = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mask = torch.ones(len(x_test), n_general, device=device)
        full_r2 = float(
            per_conf_r2(model(x_test, mask).cpu().numpy(), y_test).mean()
        )
    output = ROOT / "outputs" / f"oracle_{args.kind}_seed{args.seed}.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "kind": args.kind,
            "seed": args.seed,
            "stats": stats,
            "val_loss": best,
            "r2_full": full_r2,
            "epochs_run": epoch + 1,
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
        },
        output,
    )
    elapsed = time.time() - started
    log(
        "FEATURE-TRAIN",
        "DONE",
        f"{args.kind}/seed{args.seed} 完成",
        elapsed_s=elapsed,
    )
    print(
        f"{args.kind}: epoch={epoch + 1} val={best:.5f} "
        f"fullR2={full_r2:.4f} params={sum(p.numel() for p in model.parameters())} "
        f"[{elapsed:.1f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
