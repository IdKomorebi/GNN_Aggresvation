# -*- coding: utf-8 -*-
"""预计算充分统计量的 poly2 闭式攻击者。

对固定字典 Φ，任意集合只需抽取相应列的 Gram/cross 子矩阵：
  G = ΦᵀΦ, H = ΦᵀY
即可完成训练内 alpha 选择、全训练拟合和 test R² 计算。
扫描时不再逐集合遍历 4,589 行数据。
"""
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
R83 = REPO / "DNN_Aggresvation83"
OUT = ROOT / "outputs"
CACHE = OUT / "poly2_moments.npz"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R83 / "src"))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from truth import load_correct_truth  # noqa: E402
from runlog import log  # noqa: E402

ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])

spec = importlib.util.spec_from_file_location(
    "d83_structured", R83 / "scripts" / "structured_solver.py"
)
assert spec and spec.loader
d83 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d83)


def load_data() -> dict:
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    return prepare_data(cfg)


def dictionary(z: np.ndarray) -> np.ndarray:
    pairs = list(combinations(range(z.shape[1]), 2))
    products = np.column_stack(
        [z[:, left] * z[:, right] for left, right in pairs]
    )
    return np.column_stack([z, z**2, products]).astype(np.float64)


def moments(x: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        x.T @ x,
        x.T @ residual,
        np.sum(residual**2, axis=0),
    )


def build_cache() -> tuple[dict[str, np.ndarray], float]:
    started = time.perf_counter()
    data = load_data()
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    train_z = data["train_data"][:, gi].astype(np.float64)
    test_z = data["test_data"][:, gi].astype(np.float64)
    train_y = data["train_data"][:, ci].astype(np.float64)
    test_y = data["test_data"][:, ci].astype(np.float64)
    phi_train = dictionary(train_z)
    phi_test = dictionary(test_z)

    permutation = np.random.RandomState(20260724).permutation(len(train_z))
    n_val = round(len(permutation) * 0.15)
    val_idx = permutation[:n_val]
    fit_idx = permutation[n_val:]

    fit_mean = phi_train[fit_idx].mean(axis=0)
    fit_std = phi_train[fit_idx].std(axis=0)
    fit_std[fit_std < 1e-8] = 1.0
    x_fit = (phi_train[fit_idx] - fit_mean) / fit_std
    x_val = (phi_train[val_idx] - fit_mean) / fit_std
    y_fit_mean = train_y[fit_idx].mean(axis=0)
    g_fit, h_fit, _ = moments(
        x_fit, train_y[fit_idx] - y_fit_mean
    )
    g_val, h_val, rtr_val = moments(
        x_val, train_y[val_idx] - y_fit_mean
    )

    all_mean = phi_train.mean(axis=0)
    all_std = phi_train.std(axis=0)
    all_std[all_std < 1e-8] = 1.0
    x_all = (phi_train - all_mean) / all_std
    x_test = (phi_test - all_mean) / all_std
    y_all_mean = train_y.mean(axis=0)
    g_all, h_all, _ = moments(x_all, train_y - y_all_mean)
    g_test, h_test, rtr_test = moments(
        x_test, test_y - y_all_mean
    )
    target_total = ((test_y - test_y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    arrays = {
        "g_fit": g_fit,
        "h_fit": h_fit,
        "g_val": g_val,
        "h_val": h_val,
        "rtr_val": rtr_val,
        "g_all": g_all,
        "h_all": h_all,
        "g_test": g_test,
        "h_test": h_test,
        "rtr_test": rtr_test,
        "target_total": target_total,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **arrays)
    return arrays, time.perf_counter() - started


def feature_columns(
    key: tuple[int, ...],
    n_general: int,
    pair_position: dict[tuple[int, int], int],
) -> list[int]:
    return (
        list(key)
        + [n_general + index for index in key]
        + [
            2 * n_general + pair_position[tuple(sorted(pair))]
            for pair in combinations(key, 2)
        ]
    )


def query(
    columns: list[int],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    ix = np.ix_(columns, columns)
    gf = cache["g_fit"][ix]
    hf = cache["h_fit"][columns]
    gv = cache["g_val"][ix]
    hv = cache["h_val"][columns]
    identity = np.eye(len(columns))
    validation = []
    for alpha in ALPHAS:
        beta = np.linalg.solve(gf + alpha * identity, hf)
        sse = (
            cache["rtr_val"]
            - 2.0 * np.sum(beta * hv, axis=0)
            + np.sum(beta * (gv @ beta), axis=0)
        )
        validation.append(sse)
    best = np.argmin(np.stack(validation, axis=0), axis=0)

    ga = cache["g_all"][ix]
    ha = cache["h_all"][columns]
    gt = cache["g_test"][ix]
    ht = cache["h_test"][columns]
    sse = np.zeros(ha.shape[1], dtype=np.float64)
    for alpha_index, alpha in enumerate(ALPHAS):
        conf = np.flatnonzero(best == alpha_index)
        if not len(conf):
            continue
        beta = np.linalg.solve(ga + alpha * identity, ha[:, conf])
        sse[conf] = (
            cache["rtr_test"][conf]
            - 2.0 * np.sum(beta * ht[:, conf], axis=0)
            + np.sum(beta * (gt @ beta), axis=0)
        )
    return np.clip(1.0 - sse / cache["target_total"], 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    log("CACHE", "START", "poly2充分统计量构建与全空间扫描")
    if args.rebuild or not CACHE.exists():
        cache, build_seconds = build_cache()
    else:
        started = time.perf_counter()
        with np.load(CACHE) as payload:
            cache = {key: payload[key] for key in payload.files}
        build_seconds = 0.0
        load_seconds = time.perf_counter() - started
    data = load_data()
    n_general = int(data["n_general"])
    conf_names = data["confidential"]
    pairs = list(combinations(range(n_general), 2))
    pair_position = {pair: position for position, pair in enumerate(pairs)}
    keys = pairs + list(combinations(range(n_general), 3))

    scan_started = time.perf_counter()
    rows = []
    for key in keys:
        r2 = query(feature_columns(key, n_general, pair_position), cache)
        for conf, estimate in zip(conf_names, r2):
            rows.append(
                {
                    "indices": repr(tuple(key)),
                    "size": len(key),
                    "model": "cached_poly2",
                    "conf": conf,
                    "est": float(estimate),
                }
            )
    scan_seconds = time.perf_counter() - scan_started
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "cached_poly2_estimates.csv.gz", index=False, compression="gzip"
    )
    frame["indices"] = frame.indices.map(ast.literal_eval)
    truth_entries, truth_triples = load_correct_truth()
    summary = d83.evaluate_model(
        frame, truth_entries, truth_triples, "cached_poly2"
    )

    reference = pd.read_csv(
        R83 / "outputs" / "structured_estimates_all.csv.gz"
    )
    reference = reference[reference.model == "poly2"].copy()
    reference["indices"] = reference.indices.map(ast.literal_eval)
    check = frame.merge(
        reference[["indices", "conf", "est"]],
        on=["indices", "conf"],
        suffixes=("_cached", "_reference"),
        validate="one_to_one",
    )
    max_delta = float(
        np.max(np.abs(check.est_cached - check.est_reference))
    )
    benchmark = {
        "cache_build_seconds": build_seconds,
        "cache_load_seconds": locals().get("load_seconds", 0.0),
        "scan_seconds": scan_seconds,
        "n_sets": len(keys),
        "sets_per_second": len(keys) / scan_seconds,
        "max_abs_r2_delta_vs_d83": max_delta,
        **summary,
    }
    benchmark_name = (
        "cached_poly2_benchmark.json"
        if args.rebuild
        else "cached_poly2_benchmark_warm.json"
    )
    (OUT / benchmark_name).write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        "CACHE",
        "DONE",
        f"扫描{len(keys)}集合，{scan_seconds:.2f}s，maxΔ={max_delta:.2e}",
    )
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
