#!/usr/bin/env python3
"""Wall-clock cost of K0, one-gradient subspace signatures, full FT, and LoRA."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    flat_gradient,
    load_base_model,
    load_data,
    load_design,
    task_mask,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data, tensors = load_data(device)
    design = load_design()
    tasks = {task["task_id"]: task for task in design["tasks"]}
    audit_ids = list(design["splits"]["audit_targets"])[:50]
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    subspace = torch.load(
        ROOT / "outputs/learned_subspace.pt", map_location=device, weights_only=False
    )["u"].to(device)
    order = np.random.RandomState(9256).permutation(len(tensors["x_train"]))
    rows = []

    # Warm up kernels and allocator.
    warm_model = load_base_model(device, n_general, n_conf)
    warm_mask = task_mask(tasks[audit_ids[0]], n_general, device)
    for _ in range(5):
        warm_model(tensors["x_test"], warm_mask.expand(len(tensors["x_test"]), -1))
    torch.cuda.synchronize()

    for task_id in audit_ids:
        task = tasks[task_id]
        mask = task_mask(task, n_general, device)
        model = load_base_model(device, n_general, n_conf)
        model.eval()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(tensors["x_test"], mask.expand(len(tensors["x_test"]), -1))
        torch.cuda.synchronize()
        rows.append({"task_id": task_id, "method": "K0_forward", "ms": 1000 * (time.perf_counter() - start)})

        for support in (256, 1024):
            idx = torch.as_tensor(order[:support], dtype=torch.long, device=device)
            model = load_base_model(device, n_general, n_conf)
            model.eval()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            loss = (
                (
                    model(tensors["x_train"][idx], mask.expand(support, -1))
                    - tensors["y_train"][idx]
                )
                ** 2
            ).mean()
            loss.backward()
            gradient = flat_gradient(model)
            for rank in (32, 128, 320):
                ur = subspace[:, :rank]
                coefficient = gradient @ ur
                _ = ur @ coefficient
            with torch.no_grad():
                model(tensors["x_test"], mask.expand(len(tensors["x_test"]), -1))
            torch.cuda.synchronize()
            rows.append({
                "task_id": task_id,
                "method": f"onegrad_{support}_including_r32_r128_r320_and_query",
                "ms": 1000 * (time.perf_counter() - start),
            })

    measured = pd.DataFrame(rows)
    measured.to_csv(ROOT / "outputs/cost_benchmark_raw.csv", index=False)
    summary = measured.groupby("method")["ms"].agg(["mean", "median", "std"]).reset_index()

    # The trajectory collector measured cumulative full-model FT plus checkpoints.
    shards = [np.load(path) for path in sorted(
        (ROOT / "outputs/trajectories").glob("shard*of3.npz")
    )]
    elapsed = np.concatenate([z["elapsed"] for z in shards])
    kgrid = list(map(int, shards[0]["kgrid"]))
    for k in (1, 5, 25):
        values = elapsed[:, kgrid.index(k)] * 1000
        summary.loc[len(summary)] = [
            f"full_K{k}_cumulative",
            float(values.mean()),
            float(np.median(values)),
            float(values.std()),
        ]

    lora_frames = []
    for path in sorted((ROOT / "outputs").glob("lora_lr*_shard*of*.csv.gz")):
        frame = pd.read_csv(path)
        if "lr" not in frame:
            tag = path.name.split("lora_lr", 1)[1].split("_shard", 1)[0]
            frame["lr"] = float(tag.replace("p", "."))
        lora_frames.append(frame)
    lora = pd.concat(lora_frames)
    # elapsed repeats per confidential; deduplicate checkpoints.
    lora = lora.drop_duplicates(["task_id", "rank", "K", "lr"])
    for keys, frame in lora.groupby(["rank", "K", "lr"]):
        rank, k, lr = keys
        values = frame["elapsed"].to_numpy() * 1000
        summary.loc[len(summary)] = [
            f"lora_r{rank}_K{k}_lr{lr:g}",
            float(values.mean()),
            float(np.median(values)),
            float(values.std()),
        ]
    summary.to_csv(ROOT / "outputs/cost_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
