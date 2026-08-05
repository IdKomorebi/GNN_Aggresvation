# -*- coding: utf-8 -*-
"""训练精确三元组偏置 MLP oracle；除掩码分布外对齐75号。"""
from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R75 = REPO / "DNN_Aggresvation75"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from src.oracle import MLPOracle  # noqa: E402
from runlog import log  # noqa: E402
from samplers import (  # noqa: E402
    SAMPLERS,
    expected_size_mass,
    local234_uniform10,
    triple70_pair20_uniform10,
)

BATCH = 256
LR = 1e-3
WD = 5e-4
N_VAL_MASK = 8


def per_conf_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None)


def load_data() -> dict:
    cfg = yaml.safe_load((R75 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    torch.manual_seed(42)
    np.random.seed(42)
    return prepare_data(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=sorted(SAMPLERS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    warm = args.scheme.startswith("warm_")
    epochs = 120 if warm else 400
    patience = 30 if warm else 60
    log(
        "TRAIN",
        "START",
        note=f"{args.scheme}/seed{args.seed}；{'uniform暖启动' if warm else '从头训练'}",
        scheme=args.scheme,
        seed=args.seed,
    )

    data = load_data()
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
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

    n_samples = len(x_train)
    n_val = max(int(n_samples * 0.15), 1)
    permutation = np.random.RandomState(args.seed).permutation(n_samples)
    val_idx = torch.as_tensor(permutation[:n_val], device=device)
    train_idx = torch.as_tensor(permutation[n_val:], device=device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOracle(n_general, n_conf)
    if warm:
        checkpoint = torch.load(
            R75 / "outputs" / f"oracle_uniform_seed{args.seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sampler = SAMPLERS[args.scheme]
    # 分组只改变训练梯度的任务集中度；验证仍用逐样本独立掩码，
    # 避免因有效验证掩码数变少而改变 early-stop 噪声。
    if args.scheme.startswith("triple70_group"):
        validation_sampler = triple70_pair20_uniform10
    elif args.scheme.startswith("local234_group"):
        validation_sampler = local234_uniform10
    else:
        validation_sampler = sampler
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    val_masks = [
        torch.as_tensor(
            validation_sampler(len(val_idx), n_general, val_rng),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(N_VAL_MASK)
    ]

    best = float("inf")
    best_state = None
    stale = 0
    n_train = len(train_idx)
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(n_train, device=device)]
        for begin in range(0, n_train, BATCH):
            batch_idx = order[begin : begin + BATCH]
            masks = torch.as_tensor(
                sampler(len(batch_idx), n_general, mask_rng),
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            loss = ((model(x_train[batch_idx], masks) - y_train[batch_idx]) ** 2).mean()
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
            if stale >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full_mask = torch.ones(len(x_test), n_general, device=device)
        full_r2 = float(
            per_conf_r2(
                model(x_test, full_mask).cpu().numpy(),
                y_test,
            ).mean()
        )
    output = ROOT / "outputs" / f"oracle_{args.scheme}_seed{args.seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "scheme": args.scheme,
            "seed": args.seed,
            "warm_start": warm,
            "val_loss": best,
            "r2_full": full_r2,
            "epochs_run": epoch + 1,
            "mask_size_mass": expected_size_mass(args.scheme),
            "mask_group_size": (
                32
                if args.scheme.endswith("group32")
                else 8
                if args.scheme.endswith("group8")
                else 1
            ),
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
        },
        output,
    )
    elapsed = time.time() - started
    log(
        "TRAIN",
        "DONE",
        note=f"{args.scheme}/seed{args.seed}训练完成",
        elapsed_s=elapsed,
        scheme=args.scheme,
        seed=args.seed,
        epochs=epoch + 1,
        val_loss=round(best, 6),
    )
    print(
        f"[{args.scheme}/seed{args.seed}] epoch={epoch + 1} "
        f"val={best:.5f} fullR2={full_r2:.4f} [{elapsed:.1f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
