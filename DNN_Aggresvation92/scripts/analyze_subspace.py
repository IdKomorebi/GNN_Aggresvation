#!/usr/bin/env python3
"""Learn task-update subspaces and audit deployable one-gradient adapters."""
from __future__ import annotations

import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    load_base_model,
    load_data,
    load_design,
    parameter_vector,
    per_conf_r2,
    set_parameter_vector,
    task_mask,
)

RANKS = (4, 8, 16, 32, 64, 128, 192, 256, 320)
RIDGE_REL = (1e-6, 1e-4, 1e-2, 1.0, 100.0)


def safe_spearman(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def load_trajectories():
    arrays = []
    paths = sorted((ROOT / "outputs/trajectories").glob("shard*of3.npz"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected 3 trajectory shards, found {paths}")
    for path in paths:
        z = np.load(path)
        arrays.append({key: z[key] for key in z.files})
    ids = np.concatenate([a["task_ids"] for a in arrays]).astype(str)
    order = np.argsort(ids)
    result = {"task_ids": ids[order]}
    for key in ("delta_k25", "gradient_256", "gradient_1024", "predictions", "elapsed"):
        result[key] = np.concatenate([a[key] for a in arrays], axis=0)[order]
    result["kgrid"] = arrays[0]["kgrid"]
    return result


def fit_ridge(x_train, y_train, x_val, y_val):
    """Centered multi-output ridge; lambda selected by validation coefficient MSE."""
    x_mean = x_train.mean(0, keepdims=True)
    y_mean = y_train.mean(0, keepdims=True)
    x_scale = x_train.std(0, keepdims=True)
    x_scale[x_scale < 1e-8] = 1.0
    xs = (x_train - x_mean) / x_scale
    xv = (x_val - x_mean) / x_scale
    yc = y_train - y_mean
    gram = xs.T @ xs
    cross = xs.T @ yc
    base = max(float(np.trace(gram) / max(gram.shape[0], 1)), 1e-12)
    best = None
    eye = np.eye(gram.shape[0], dtype=np.float64)
    for relative in RIDGE_REL:
        weight = np.linalg.solve(gram + relative * base * eye, cross)
        val_prediction = xv @ weight + y_mean
        mse = float(np.mean((val_prediction - y_val) ** 2))
        if best is None or mse < best["mse"]:
            best = {
                "relative": relative,
                "mse": mse,
                "weight": weight,
                "x_mean": x_mean,
                "x_scale": x_scale,
                "y_mean": y_mean,
            }
    return best


def predict_ridge(fit, x):
    standardized = (x - fit["x_mean"]) / fit["x_scale"]
    return standardized @ fit["weight"] + fit["y_mean"]


def clip_coefficients(coefficients, train_coefficients):
    limit = float(np.quantile(np.linalg.norm(train_coefficients, axis=1), 0.99) * 1.5)
    norms = np.linalg.norm(coefficients, axis=1)
    scale = np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    return coefficients * scale[:, None]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for applying projected parameter vectors")
    started = time.time()
    device = torch.device("cuda")
    design = load_design()
    tasks = {task["task_id"]: task for task in design["tasks"]}
    trajectories = load_trajectories()
    ids = list(trajectories["task_ids"])
    id_to_row = {task_id: pos for pos, task_id in enumerate(ids)}
    split = design["splits"]
    train_rows = np.asarray([id_to_row[x] for x in split["basis_train"]])
    # A single/pair may be a parent in both validation and audit closures. Never use
    # such an exact task to tune the deployable mapper.
    pure_val_ids = sorted(set(split["validation_all"]) - set(split["audit_all"]))
    val_rows = np.asarray([id_to_row[x] for x in pure_val_ids])
    eval_ids = sorted(set(split["validation_all"]) | set(split["audit_all"]))
    eval_rows = np.asarray([id_to_row[x] for x in eval_ids])

    # The uncentered subspace passes through the oracle (delta=0), which is required
    # for a genuine adapter. A centered PCA would silently add a universal offset.
    delta_train = trajectories["delta_k25"][train_rows].astype(np.float32)
    print(f"SVD train matrix={delta_train.shape}", flush=True)
    gram = delta_train @ delta_train.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    singular = np.sqrt(eigenvalues)
    valid = singular > singular[0] * 1e-7
    max_rank = min(max(RANKS), int(valid.sum()))
    u = (delta_train.T @ left[:, :max_rank]) / singular[None, :max_rank]
    # Remove float32 numerical drift.
    u, _ = np.linalg.qr(u)
    u = u[:, :max_rank].astype(np.float32)

    spectrum = pd.DataFrame({
        "component": np.arange(1, len(singular) + 1),
        "singular_value": singular,
        "train_cumulative_energy": np.cumsum(eigenvalues) / np.maximum(eigenvalues.sum(), 1e-12),
    })
    spectrum.to_csv(ROOT / "outputs/subspace_spectrum.csv", index=False)

    data, tensors = load_data(device)
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    conf_names = list(data["confidential"])
    model = load_base_model(device, n_general, n_conf)
    base_vec = parameter_vector(model).detach().clone()

    # Truth is always the dedicated DNN retraining result, never K25 itself.
    truth_long = pd.read_csv(REPO / "DNN_Aggresvation69/outputs/truth_long.csv")
    truth_lookup = {
        (row.sid, row.conf): float(row.dnn)
        for row in truth_long.itertuples(index=False)
    }
    truth_by_task = {}
    for task_id, task in tasks.items():
        truth_by_task[task_id] = np.asarray(
            [truth_lookup[(task["sid69"], conf)] for conf in conf_names],
            dtype=np.float32,
        )

    masks = np.zeros((len(ids), n_general), dtype=np.float32)
    for row, task_id in enumerate(ids):
        masks[row, tasks[task_id]["indices"]] = 1.0

    estimate_rows = []

    def split_name(task_id):
        if task_id in split["audit_all"]:
            return "audit"
        if task_id in split["validation_all"]:
            return "validation"
        return "train"

    def role_name(task_id):
        role = tasks[task_id]["roles"]
        if "audit_target" in role or "validation_target" in role:
            return "target"
        return "parent"

    def append_estimate(task_id, method, rank, values):
        task = tasks[task_id]
        truth = truth_by_task[task_id]
        for conf_pos, conf in enumerate(conf_names):
            estimate_rows.append({
                "task_id": task_id,
                "sid69": task["sid69"],
                "split": split_name(task_id),
                "role": role_name(task_id),
                "size": task["size"],
                "group": task["group"],
                "method": method,
                "rank": rank,
                "conf": conf,
                "est": float(values[conf_pos]),
                "truth": float(truth[conf_pos]),
            })

    # Existing full-parameter optimization baselines were collected on the exact tasks.
    kgrid = list(map(int, trajectories["kgrid"]))
    for task_id, row in zip(eval_ids, eval_rows):
        for k_value in kgrid:
            pred = trajectories["predictions"][row, kgrid.index(k_value)]
            append_estimate(task_id, f"full_K{k_value}", 0, pred)

    diagnostic_rows = []
    hyper_rows = []
    train_gradient = {
        256: trajectories["gradient_256"][train_rows].astype(np.float32),
        1024: trajectories["gradient_1024"][train_rows].astype(np.float32),
    }
    val_gradient = {
        256: trajectories["gradient_256"][val_rows].astype(np.float32),
        1024: trajectories["gradient_1024"][val_rows].astype(np.float32),
    }
    eval_gradient = {
        256: trajectories["gradient_256"][eval_rows].astype(np.float32),
        1024: trajectories["gradient_1024"][eval_rows].astype(np.float32),
    }
    delta_val = trajectories["delta_k25"][val_rows].astype(np.float32)
    delta_eval = trajectories["delta_k25"][eval_rows].astype(np.float32)

    saved_fits = {}
    for rank in RANKS:
        if rank > max_rank:
            continue
        ur = u[:, :rank]
        a_train = delta_train @ ur
        a_val = delta_val @ ur
        a_eval = delta_eval @ ur
        train_recon = np.sum(a_train ** 2, axis=1) / np.maximum(
            np.sum(delta_train ** 2, axis=1), 1e-12
        )
        eval_recon = np.sum(a_eval ** 2, axis=1) / np.maximum(
            np.sum(delta_eval ** 2, axis=1), 1e-12
        )
        diagnostic_rows.extend([
            {
                "rank": rank,
                "split": "train",
                "metric": "delta_energy_fraction",
                "mean": float(train_recon.mean()),
                "median": float(np.median(train_recon)),
            },
            {
                "rank": rank,
                "split": "validation_audit",
                "metric": "delta_energy_fraction",
                "mean": float(eval_recon.mean()),
                "median": float(np.median(eval_recon)),
            },
        ])

        coefficient_sets = {"projected_teacher": a_eval}
        for support in (256, 1024):
            b_train = -(train_gradient[support] @ ur)
            b_val = -(val_gradient[support] @ ur)
            b_eval = -(eval_gradient[support] @ ur)

            # One scalar learned from training trajectories: the fairest projected-SGD baseline.
            eta = float(np.sum(b_train * a_train) / np.maximum(np.sum(b_train ** 2), 1e-12))
            scalar = clip_coefficients(eta * b_eval, a_train)
            coefficient_sets[f"scalar_grad{support}"] = scalar
            hyper_rows.append({
                "rank": rank, "support": support, "model": "scalar",
                "ridge_relative": np.nan, "val_coefficient_mse": float(
                    np.mean((eta * b_val - a_val) ** 2)
                ),
                "eta": eta,
            })

            feature_options = {
                f"gradmap_{support}": (b_train, b_val, b_eval),
                f"gradmask_{support}": (
                    np.concatenate([b_train, masks[train_rows]], axis=1),
                    np.concatenate([b_val, masks[val_rows]], axis=1),
                    np.concatenate([b_eval, masks[eval_rows]], axis=1),
                ),
            }
            for method, (x_train, x_val, x_eval) in feature_options.items():
                fit = fit_ridge(
                    x_train.astype(np.float64),
                    a_train.astype(np.float64),
                    x_val.astype(np.float64),
                    a_val.astype(np.float64),
                )
                predicted = predict_ridge(fit, x_eval.astype(np.float64)).astype(np.float32)
                coefficient_sets[method] = clip_coefficients(predicted, a_train)
                hyper_rows.append({
                    "rank": rank,
                    "support": support,
                    "model": method,
                    "ridge_relative": fit["relative"],
                    "val_coefficient_mse": fit["mse"],
                    "eta": np.nan,
                })
                saved_fits[(rank, method)] = fit

        # Mask-only control: detects whether a supposed gradient method is merely memorizing masks.
        mask_fit = fit_ridge(
            masks[train_rows].astype(np.float64),
            a_train.astype(np.float64),
            masks[val_rows].astype(np.float64),
            a_val.astype(np.float64),
        )
        coefficient_sets["maskmap"] = clip_coefficients(
            predict_ridge(mask_fit, masks[eval_rows].astype(np.float64)).astype(np.float32),
            a_train,
        )
        hyper_rows.append({
            "rank": rank, "support": 0, "model": "maskmap",
            "ridge_relative": mask_fit["relative"],
            "val_coefficient_mse": mask_fit["mse"],
            "eta": np.nan,
        })

        ur_gpu = torch.as_tensor(ur, dtype=torch.float32, device=device)
        for method, coefficients in coefficient_sets.items():
            coeff_gpu = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
            # Materialize in moderate batches to cap host/GPU memory.
            for begin in range(0, len(eval_ids), 32):
                end = min(begin + 32, len(eval_ids))
                delta_batch = coeff_gpu[begin:end] @ ur_gpu.T
                for local, task_id in enumerate(eval_ids[begin:end]):
                    set_parameter_vector(model, base_vec + delta_batch[local])
                    mask = task_mask(tasks[task_id], n_general, device)
                    model.eval()
                    with torch.no_grad():
                        prediction = model(
                            tensors["x_test"], mask.expand(len(tensors["x_test"]), -1)
                        ).cpu().numpy()
                    append_estimate(
                        task_id,
                        method,
                        rank,
                        per_conf_r2(prediction, tensors["y_test_np"]),
                    )
            print(f"rank={rank:3d} method={method} done", flush=True)

    pd.DataFrame(diagnostic_rows).to_csv(
        ROOT / "outputs/subspace_diagnostics.csv", index=False
    )
    pd.DataFrame(hyper_rows).to_csv(
        ROOT / "outputs/subspace_hyperparameters.csv", index=False
    )
    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(ROOT / "outputs/estimates.csv.gz", index=False, compression="gzip")

    # Direct v(S) fidelity, reported only on target tasks.
    direct_rows = []
    targets = estimates[estimates["role"] == "target"]
    for keys, frame in targets.groupby(["split", "method", "rank", "group"], dropna=False):
        split_value, method, rank, group = keys
        error = frame["est"].to_numpy() - frame["truth"].to_numpy()
        direct_rows.append({
            "split": split_value,
            "method": method,
            "rank": rank,
            "group": group,
            "n_values": len(frame),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "spearman": safe_spearman(frame["est"], frame["truth"]),
        })
    pd.DataFrame(direct_rows).to_csv(ROOT / "outputs/direct_metrics.csv", index=False)

    # Parent-child irreducible increments.
    estimate_lookup = {
        (row.task_id, row.method, int(row.rank), row.conf): float(row.est)
        for row in estimates.itertuples(index=False)
    }
    methods = sorted(set(zip(estimates["method"], estimates["rank"])))
    synergy_rows = []
    per_task_syn_rows = []

    def task_id_from_indices(indices):
        return "q_" + "_".join(f"{x:02d}" for x in sorted(indices))

    for split_value in ("validation", "audit"):
        target_ids = (
            split["validation_targets"] if split_value == "validation"
            else split["audit_targets"]
        )
        for task_id in target_ids:
            task = tasks[task_id]
            if task["size"] not in (2, 3):
                continue
            child_indices = task["indices"]
            if task["size"] == 2:
                parent_ids = [task_id_from_indices([x]) for x in child_indices]
            else:
                parent_ids = [
                    task_id_from_indices(parent)
                    for parent in combinations(child_indices, task["size"] - 1)
                ]
            truth_child = truth_by_task[task_id]
            truth_parent = np.stack([truth_by_task[x] for x in parent_ids])
            truth_syn = truth_child - truth_parent.max(axis=0)
            for method, rank in methods:
                try:
                    est_child = np.asarray([
                        estimate_lookup[(task_id, method, int(rank), conf)]
                        for conf in conf_names
                    ])
                    est_parent = np.stack([
                        [
                            estimate_lookup[(parent, method, int(rank), conf)]
                            for conf in conf_names
                        ]
                        for parent in parent_ids
                    ])
                except KeyError:
                    continue
                est_syn = est_child - est_parent.max(axis=0)
                for conf_pos, conf in enumerate(conf_names):
                    per_task_syn_rows.append({
                        "split": split_value,
                        "task_id": task_id,
                        "order": task["size"],
                        "group": task["group"],
                        "method": method,
                        "rank": int(rank),
                        "conf": conf,
                        "est_syn": float(est_syn[conf_pos]),
                        "truth_syn": float(truth_syn[conf_pos]),
                    })

    syn = pd.DataFrame(per_task_syn_rows)
    syn.to_csv(ROOT / "outputs/synergy_detail.csv.gz", index=False, compression="gzip")
    for keys, frame in syn.groupby(["split", "order", "method", "rank"]):
        split_value, order_value, method, rank = keys
        error = frame["est_syn"].to_numpy() - frame["truth_syn"].to_numpy()
        per_task = frame.groupby("task_id").agg(
            est_score=("est_syn", "max"), truth_score=("truth_syn", "max")
        ).reset_index()
        n_tasks = len(per_task)
        pool_n = max(1, int(math.ceil(0.30 * n_tasks)))
        predicted_pool = set(per_task.nlargest(pool_n, "est_score")["task_id"])
        for threshold in (0.05, 0.10, 0.20):
            strong = set(per_task.loc[per_task["truth_score"] > threshold, "task_id"])
            recall = (
                len(strong & predicted_pool) / len(strong) if strong else np.nan
            )
            synergy_rows.append({
                "split": split_value,
                "order": int(order_value),
                "method": method,
                "rank": int(rank),
                "n_tasks": n_tasks,
                "per_conf_mae": float(np.mean(np.abs(error))),
                "per_conf_bias": float(np.mean(error)),
                "per_conf_spearman": safe_spearman(frame["est_syn"], frame["truth_syn"]),
                "maxconf_spearman": safe_spearman(
                    per_task["est_score"], per_task["truth_score"]
                ),
                "pool_fraction": 0.30,
                "truth_threshold": threshold,
                "n_strong": len(strong),
                "strong_recall": recall,
            })
    pd.DataFrame(synergy_rows).to_csv(ROOT / "outputs/synergy_metrics.csv", index=False)

    torch.save(
        {
            "u": torch.from_numpy(u),
            "ranks": RANKS,
            "base_parameter_count": len(base_vec),
            "train_task_ids": split["basis_train"],
            "parameter_names": [name for name, _ in model.named_parameters()],
        },
        ROOT / "outputs/learned_subspace.pt",
    )
    summary = {
        "seconds": time.time() - started,
        "n_basis_train": len(train_rows),
        "n_validation_pure": len(val_rows),
        "n_evaluation_unique": len(eval_ids),
        "parameter_count": int(len(base_vec)),
        "max_rank": int(max_rank),
        "rank_energy": {
            str(rank): float(spectrum.loc[rank - 1, "train_cumulative_energy"])
            for rank in RANKS if rank <= len(spectrum)
        },
    }
    (ROOT / "outputs/subspace_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
