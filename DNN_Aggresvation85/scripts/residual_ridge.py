# -*- coding: utf-8 -*-
"""K0 oracle 预测 + 每集合一次闭式结构化残差修正。"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R75 = REPO / "DNN_Aggresvation75"
R83 = REPO / "DNN_Aggresvation83"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R83 / "src"))
sys.path.insert(0, str(R83 / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from src.oracle import MLPOracle  # noqa: E402
from truth import load_correct_truth  # noqa: E402
from runlog import log  # noqa: E402
from structured_solver import ALPHAS, make_features  # noqa: E402
from train_tail_oracle import load_data  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "d83_structured_residual", R83 / "scripts" / "structured_solver.py"
)
assert spec and spec.loader
d83 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d83)

MODELS = ("linear", "poly2", "arith")
MASK_BATCH = 16


def residual_solve(
    train_z: np.ndarray,
    train_raw: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_raw: np.ndarray,
    test_y: np.ndarray,
    offset_train: np.ndarray,
    offset_test: np.ndarray,
    key: tuple[int, ...],
    model: str,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design_train, ratio_specs = make_features(
        train_z, train_raw, key, model
    )
    design_test, _ = make_features(
        test_z, test_raw, key, model, ratio_specs
    )
    residual_train = train_y - offset_train

    fit_mean = design_train[fit_idx].mean(axis=0)
    fit_std = design_train[fit_idx].std(axis=0)
    fit_std[fit_std < 1e-8] = 1.0
    x_fit = (design_train[fit_idx] - fit_mean) / fit_std
    x_val = (design_train[val_idx] - fit_mean) / fit_std
    r_fit = residual_train[fit_idx]
    r_mean = r_fit.mean(axis=0)
    centered = r_fit - r_mean
    gram = x_fit.T @ x_fit
    cross = x_fit.T @ centered
    identity = np.eye(gram.shape[0])
    validation = []
    for alpha in ALPHAS:
        beta = np.linalg.solve(gram + alpha * identity, cross)
        correction = x_val @ beta + r_mean
        prediction = offset_train[val_idx] + correction
        validation.append(((prediction - train_y[val_idx]) ** 2).mean(axis=0))
    best = np.argmin(np.stack(validation, axis=0), axis=0)

    all_mean = design_train.mean(axis=0)
    all_std = design_train.std(axis=0)
    all_std[all_std < 1e-8] = 1.0
    x_all = (design_train - all_mean) / all_std
    x_test = (design_test - all_mean) / all_std
    r_mean = residual_train.mean(axis=0)
    centered = residual_train - r_mean
    gram = x_all.T @ x_all
    cross = x_all.T @ centered
    prediction = offset_test.copy()
    for alpha_index, alpha in enumerate(ALPHAS):
        conf = np.flatnonzero(best == alpha_index)
        if not len(conf):
            continue
        beta = np.linalg.solve(gram + alpha * identity, cross[:, conf])
        prediction[:, conf] += x_test @ beta + r_mean[conf]
    residual = ((prediction - test_y) ** 2).sum(axis=0)
    total = ((test_y - test_y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None), ALPHAS[best]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("truth", "all"), default="truth")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="逗号分隔：linear,poly2,arith",
    )
    args = parser.parse_args()
    selected_models = tuple(args.models.split(","))
    assert set(selected_models).issubset(MODELS)
    started = time.perf_counter()
    log(
        "RESIDUAL",
        "START",
        f"uniform K0 + structured residual ridge；scope={args.scope}, "
        f"models={selected_models}",
    )
    data = load_data()
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    conf_names = data["confidential"]
    train_z = data["train_data"][:, gi].astype(np.float64)
    test_z = data["test_data"][:, gi].astype(np.float64)
    train_y = data["train_data"][:, ci].astype(np.float64)
    test_y = data["test_data"][:, ci].astype(np.float64)
    means = data["mean"][data["general"]].to_numpy(dtype=np.float64)
    stds = data["std"][data["general"]].to_numpy(dtype=np.float64)
    train_raw = train_z * stds + means
    test_raw = test_z * stds + means

    truth_entries, truth_triples = load_correct_truth()
    triple_keys = (
        list(combinations(range(n_general), 3))
        if args.scope == "all"
        else sorted(set(truth_entries.indices))
    )
    keys = list(combinations(range(n_general), 2)) + triple_keys
    permutation = np.random.RandomState(20260724).permutation(len(train_z))
    n_val = round(len(permutation) * 0.15)
    val_idx = permutation[:n_val]
    fit_idx = permutation[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        R75 / "outputs" / "oracle_uniform_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    oracle = MLPOracle(n_general, n_conf)
    oracle.load_state_dict(checkpoint["state"])
    oracle = oracle.to(device).eval()
    x_train_t = torch.as_tensor(train_z, dtype=torch.float32, device=device)
    x_test_t = torch.as_tensor(test_z, dtype=torch.float32, device=device)

    rows = []
    alpha_rows = []
    k0_rows = []
    forward_seconds = 0.0
    solve_seconds = 0.0
    for begin in range(0, len(keys), MASK_BATCH):
        chunk = keys[begin : begin + MASK_BATCH]
        masks_np = np.zeros((len(chunk), n_general), dtype=np.float32)
        for row, key in enumerate(chunk):
            masks_np[row, list(key)] = 1.0
        masks = torch.as_tensor(masks_np, device=device)
        tick = time.perf_counter()
        with torch.no_grad():
            train_offset = (
                oracle(
                    x_train_t[None]
                    .expand(len(chunk), -1, -1)
                    .reshape(-1, n_general),
                    masks[:, None]
                    .expand(-1, len(x_train_t), -1)
                    .reshape(-1, n_general),
                )
                .reshape(len(chunk), len(x_train_t), n_conf)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            test_offset = (
                oracle(
                    x_test_t[None]
                    .expand(len(chunk), -1, -1)
                    .reshape(-1, n_general),
                    masks[:, None]
                    .expand(-1, len(x_test_t), -1)
                    .reshape(-1, n_general),
                )
                .reshape(len(chunk), len(x_test_t), n_conf)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        forward_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        for position, key in enumerate(chunk):
            for model in selected_models:
                r2, alpha = residual_solve(
                    train_z,
                    train_raw,
                    train_y,
                    test_z,
                    test_raw,
                    test_y,
                    train_offset[position],
                    test_offset[position],
                    key,
                    model,
                    fit_idx,
                    val_idx,
                )
                name = f"k0_residual_{model}"
                for conf, estimate, selected_alpha in zip(conf_names, r2, alpha):
                    rows.append(
                        {
                            "indices": repr(tuple(key)),
                            "size": len(key),
                            "model": name,
                            "conf": conf,
                            "est": float(estimate),
                        }
                    )
                    alpha_rows.append(
                        {
                            "indices": repr(tuple(key)),
                            "model": name,
                            "conf": conf,
                            "alpha": float(selected_alpha),
                        }
                    )
        solve_seconds += time.perf_counter() - tick
        for position, key in enumerate(chunk):
            residual = ((test_offset[position] - test_y) ** 2).sum(axis=0)
            total = ((test_y - test_y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
            k0_r2 = np.clip(1.0 - residual / total, 0.0, None)
            for conf, estimate in zip(conf_names, k0_r2):
                k0_rows.append(
                    {
                        "indices": repr(tuple(key)),
                        "size": len(key),
                        "model": "uniform_k0",
                        "conf": conf,
                        "est": float(estimate),
                    }
                )
        if (begin // MASK_BATCH) % 50 == 0:
            print(f"{min(begin + MASK_BATCH, len(keys))}/{len(keys)}", flush=True)

    frame = pd.concat([pd.DataFrame(k0_rows), pd.DataFrame(rows)], ignore_index=True)
    if args.scope == "truth":
        suffix = ""
    elif len(selected_models) == 1:
        suffix = f"_all_{selected_models[0]}"
    else:
        suffix = "_all"
    frame.to_csv(
        OUT / f"residual_ridge_estimates{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(alpha_rows).to_csv(
        OUT / f"residual_ridge_alphas{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    frame["indices"] = frame.indices.map(ast.literal_eval)
    evaluated_models = ["uniform_k0"] + [
        f"k0_residual_{model}" for model in selected_models
    ]
    summaries = pd.DataFrame(
        [
            d83.evaluate_model(frame, truth_entries, truth_triples, name)
            for name in evaluated_models
        ]
    )
    summaries.to_csv(OUT / f"residual_ridge_summary{suffix}.csv", index=False)
    benchmark = {
        "n_sets": len(keys),
        "n_models": len(selected_models),
        "forward_seconds": forward_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    (OUT / f"residual_ridge_benchmark{suffix}.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        "RESIDUAL",
        "DONE",
        f"{len(keys)}集合×{len(selected_models)}残差模型完成",
        **benchmark,
    )
    print(summaries.to_string(index=False))
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
