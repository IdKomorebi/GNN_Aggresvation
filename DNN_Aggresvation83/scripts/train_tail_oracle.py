# -*- coding: utf-8 -*-
"""用 mask×conf 任务级 CVaR 训练低阶通用 oracle。

D82 的 grouped-mask 仍对所有样本、mask、conf 求普通平均。本实验保持相同
local-2/3/4 掩码分布和 MLP，只改变聚合损失：

    ERM + lambda * (CVaR_q - ERM)

其中每个“任务单元”是同一 mask 下、同一 confidential 输出在 32 个样本上的
平均 MSE。这样难的 mask-conf 组合不会再被另外 95 个容易任务单元平均掉。
"""
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
R82 = REPO / "DNN_Aggresvation82"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R82 / "src"))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from src.oracle import MLPOracle  # noqa: E402
from samplers import local234_group32  # noqa: E402
from runlog import log  # noqa: E402

BATCH = 256
GROUP = 32
LR = 1e-3
WD = 5e-4
EPOCHS = 400
PATIENCE = 60
N_VAL_MASK_BATCHES = 8

# (CVaR 保留的最难任务比例 q, CVaR 在总损失中的权重 lambda)
VARIANTS = {
    "cvar50": (0.50, 1.00),
    "cvar25": (0.25, 1.00),
    "cvar10": (0.10, 1.00),
    "mix25_cvar25": (0.25, 0.25),
    "mix50_cvar25": (0.25, 0.50),
    "mix75_cvar25": (0.25, 0.75),
}


def load_data() -> dict:
    cfg = yaml.safe_load((R75 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    torch.manual_seed(42)
    np.random.seed(42)
    return prepare_data(cfg)


def task_losses(
    squared_error: torch.Tensor,
    group_size: int = GROUP,
) -> torch.Tensor:
    """把逐样本逐 conf 损失聚合为同 mask×conf 的任务损失。"""
    chunks = []
    for begin in range(0, len(squared_error), group_size):
        chunks.append(squared_error[begin : begin + group_size].mean(dim=0))
    return torch.stack(chunks, dim=0).flatten()


def robust_loss(
    squared_error: torch.Tensor,
    q: float,
    cvar_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tasks = task_losses(squared_error)
    mean_loss = tasks.mean()
    n_tail = max(1, int(np.ceil(q * tasks.numel())))
    tail_loss = torch.topk(tasks, k=n_tail, largest=True).values.mean()
    loss = mean_loss + cvar_weight * (tail_loss - mean_loss)
    return loss, mean_loss, tail_loss


def per_conf_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    q, cvar_weight = VARIANTS[args.variant]
    started = time.time()
    log(
        "TRAIN",
        "START",
        f"{args.variant}/seed{args.seed}；q={q}, lambda={cvar_weight}",
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
    model = MLPOracle(n_general, n_conf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    val_masks = [
        torch.as_tensor(
            local234_group32(len(val_idx), n_general, val_rng),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(N_VAL_MASK_BATCHES)
    ]

    best = float("inf")
    best_mean = float("inf")
    best_tail = float("inf")
    best_state = None
    stale = 0
    n_train = len(train_idx)
    for epoch in range(EPOCHS):
        model.train()
        order = train_idx[torch.randperm(n_train, device=device)]
        for begin in range(0, n_train, BATCH):
            batch_idx = order[begin : begin + BATCH]
            masks = torch.as_tensor(
                local234_group32(len(batch_idx), n_general, mask_rng),
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            error2 = (
                model(x_train[batch_idx], masks) - y_train[batch_idx]
            ) ** 2
            loss, _, _ = robust_loss(error2, q, cvar_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        val_values = []
        val_means = []
        val_tails = []
        with torch.no_grad():
            for masks in val_masks:
                error2 = (model(x_val, masks) - y_val) ** 2
                loss, mean_loss, tail_loss = robust_loss(error2, q, cvar_weight)
                val_values.append(loss.item())
                val_means.append(mean_loss.item())
                val_tails.append(tail_loss.item())
        val_objective = float(np.mean(val_values))
        if val_objective < best - 1e-6:
            best = val_objective
            best_mean = float(np.mean(val_means))
            best_tail = float(np.mean(val_tails))
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
        full_mask = torch.ones(len(x_test), n_general, device=device)
        full_r2 = float(
            per_conf_r2(model(x_test, full_mask).cpu().numpy(), y_test).mean()
        )
    output = ROOT / "outputs" / f"oracle_{args.variant}_seed{args.seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "variant": args.variant,
            "seed": args.seed,
            "q": q,
            "cvar_weight": cvar_weight,
            "val_loss": best,
            "val_mean": best_mean,
            "val_tail": best_tail,
            "r2_full": full_r2,
            "epochs_run": epoch + 1,
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
        },
        output,
    )
    elapsed = time.time() - started
    log(
        "TRAIN",
        "DONE",
        f"{args.variant}/seed{args.seed} 完成，epoch={epoch + 1}",
        elapsed_s=elapsed,
    )
    print(
        f"[{args.variant}/seed{args.seed}] epoch={epoch + 1} "
        f"robust={best:.5f} mean={best_mean:.5f} tail={best_tail:.5f} "
        f"fullR2={full_r2:.4f} [{elapsed:.1f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
