#!/usr/bin/env python3
"""Collect K25 task deltas and deterministic one-gradient task signatures."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    flat_gradient,
    load_base_model,
    load_data,
    load_design,
    parameter_vector,
    per_conf_r2,
    task_mask,
)

BATCH = 256
LR = 1e-3
WD = 5e-4
KGRID = (0, 1, 5, 25)


def deterministic_gradient(model, x, y, mask, indices):
    model.eval()
    model.zero_grad(set_to_none=True)
    batch_idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    batch_mask = mask.expand(len(batch_idx), -1)
    loss = ((model(x[batch_idx], batch_mask) - y[batch_idx]) ** 2).mean()
    loss.backward()
    result = flat_gradient(model).cpu()
    model.zero_grad(set_to_none=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshard", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "outputs/trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard{args.shard}of{args.nshard}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; skip")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This trajectory collector requires CUDA")
    device = torch.device("cuda")
    data, tensors = load_data(device)
    design = load_design()
    tasks = [
        task for pos, task in enumerate(design["tasks"])
        if pos % args.nshard == args.shard
    ]
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    base_model = load_base_model(device, n_general, n_conf)
    base_vec = parameter_vector(base_model).detach().clone()
    parameter_count = base_vec.numel()

    fixed_order = np.random.RandomState(9256).permutation(len(tensors["x_train"]))
    grad_idx_256 = fixed_order[:256]
    grad_idx_1024 = fixed_order[:1024]

    task_ids = []
    deltas = np.empty((len(tasks), parameter_count), dtype=np.float16)
    gradients_256 = np.empty_like(deltas)
    gradients_1024 = np.empty_like(deltas)
    predictions = np.empty((len(tasks), len(KGRID), n_conf), dtype=np.float32)
    elapsed = np.empty((len(tasks), len(KGRID)), dtype=np.float32)
    started = time.time()

    for task_pos, task in enumerate(tasks):
        mask = task_mask(task, n_general, device)
        mask_train = mask.expand(len(tensors["x_train"]), -1)
        mask_test = mask.expand(len(tensors["x_test"]), -1)

        # Stable task signature: same rows and dropout disabled for every task.
        model = load_base_model(device, n_general, n_conf)
        g256 = deterministic_gradient(
            model, tensors["x_train"], tensors["y_train"], mask, grad_idx_256
        )
        model = load_base_model(device, n_general, n_conf)
        g1024 = deterministic_gradient(
            model, tensors["x_train"], tensors["y_train"], mask, grad_idx_1024
        )

        # Teacher trajectory: identical initialization and mini-batch order per task.
        torch.manual_seed(0)
        np.random.seed(0)
        model = load_base_model(device, n_general, n_conf)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        checkpoint_to_pos = {k: i for i, k in enumerate(KGRID)}

        def snapshot(k_value, seconds):
            model.eval()
            with torch.no_grad():
                pred = model(tensors["x_test"], mask_test).cpu().numpy()
            predictions[task_pos, checkpoint_to_pos[k_value]] = per_conf_r2(
                pred, tensors["y_test_np"]
            )
            elapsed[task_pos, checkpoint_to_pos[k_value]] = seconds

        snapshot(0, 0.0)
        rng = np.random.RandomState(0)
        step = 0
        tune_started = time.perf_counter()
        while step < max(KGRID):
            order = rng.permutation(len(tensors["x_train"]))
            for begin in range(0, len(order), BATCH):
                idx = torch.as_tensor(
                    order[begin : begin + BATCH], dtype=torch.long, device=device
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    (
                        model(tensors["x_train"][idx], mask_train[idx])
                        - tensors["y_train"][idx]
                    )
                    ** 2
                ).mean()
                loss.backward()
                optimizer.step()
                step += 1
                if step in checkpoint_to_pos:
                    snapshot(step, time.perf_counter() - tune_started)
                if step >= max(KGRID):
                    break

        delta = parameter_vector(model).detach().cpu() - base_vec.cpu()
        task_ids.append(task["task_id"])
        deltas[task_pos] = delta.numpy().astype(np.float16)
        gradients_256[task_pos] = g256.numpy().astype(np.float16)
        gradients_1024[task_pos] = g1024.numpy().astype(np.float16)

        if (task_pos + 1) % 25 == 0:
            print(
                f"shard {args.shard}/{args.nshard}: {task_pos+1}/{len(tasks)} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    np.savez_compressed(
        out_path,
        task_ids=np.asarray(task_ids),
        delta_k25=deltas,
        gradient_256=gradients_256,
        gradient_1024=gradients_1024,
        predictions=predictions,
        elapsed=elapsed,
        kgrid=np.asarray(KGRID, dtype=np.int32),
    )
    meta = {
        "shard": args.shard,
        "nshard": args.nshard,
        "n_tasks": len(tasks),
        "parameter_count": parameter_count,
        "dtype_vectors": "float16",
        "gradient_protocol": "eval-mode deterministic same rows",
        "teacher": {"optimizer": "Adam", "lr": LR, "weight_decay": WD, "K": 25},
        "seconds": time.time() - started,
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
