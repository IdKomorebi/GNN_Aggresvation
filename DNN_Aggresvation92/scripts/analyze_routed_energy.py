#!/usr/bin/env python3
"""Does a size-routed PCA fix the global task-subspace failure?"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANKS = (4, 8, 16, 32, 64, 128)


def learn_u(delta, max_rank):
    gram = delta @ delta.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
    left = left[:, order]
    rank = min(max_rank, int((singular > singular[0] * 1e-7).sum()))
    u = (delta.T @ left[:, :rank]) / singular[None, :rank]
    u, _ = np.linalg.qr(u)
    return u[:, :rank].astype(np.float32)


def energy(delta, u):
    coefficient = delta @ u
    return np.sum(coefficient ** 2, axis=1) / np.maximum(
        np.sum(delta ** 2, axis=1), 1e-12
    )


def main():
    design = json.loads((ROOT / "outputs/design.json").read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in design["tasks"]}
    arrays = [
        np.load(path)
        for path in sorted((ROOT / "outputs/trajectories").glob("shard*of3.npz"))
    ]
    ids = np.concatenate([z["task_ids"] for z in arrays]).astype(str)
    delta = np.concatenate([z["delta_k25"] for z in arrays]).astype(np.float32)
    order = np.argsort(ids)
    ids, delta = ids[order], delta[order]
    row = {task_id: pos for pos, task_id in enumerate(ids)}
    base_ids = design["splits"]["basis_train"]
    eval_ids = sorted(
        set(design["splits"]["validation_targets"])
        | set(design["splits"]["audit_targets"])
    )

    def group(task_id):
        size = tasks[task_id]["size"]
        return "pair" if size == 2 else "triple" if size == 3 else "wide"

    global_delta = delta[[row[x] for x in base_ids]]
    global_u = learn_u(global_delta, max(RANKS))
    rows = []
    for group_name in ("pair", "triple", "wide"):
        train_ids = [x for x in base_ids if group(x) == group_name]
        held_ids = [x for x in eval_ids if group(x) == group_name]
        train_delta = delta[[row[x] for x in train_ids]]
        held_delta = delta[[row[x] for x in held_ids]]
        routed_u = learn_u(train_delta, max(RANKS))
        for rank in RANKS:
            global_rank = min(rank, global_u.shape[1])
            routed_rank = min(rank, routed_u.shape[1])
            global_fraction = energy(held_delta, global_u[:, :global_rank])
            routed_fraction = energy(held_delta, routed_u[:, :routed_rank])
            rows.append({
                "group": group_name,
                "requested_rank": rank,
                "global_effective_rank": global_rank,
                "routed_effective_rank": routed_rank,
                "n_train": len(train_ids),
                "n_heldout_targets": len(held_ids),
                "global_mean_energy": float(global_fraction.mean()),
                "routed_mean_energy": float(routed_fraction.mean()),
                "delta_routed_minus_global": float(
                    routed_fraction.mean() - global_fraction.mean()
                ),
            })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "outputs/routed_subspace_energy.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
