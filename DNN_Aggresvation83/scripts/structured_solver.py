# -*- coding: utf-8 -*-
"""二、三元集合的显式算术特征 + 闭式岭回归攻击者。

目的不是宣称线性模型能替代通用 oracle，而是检验两件事：
1. 当前最难协同是否主要来自 MLP 很难快速学到的乘法/除法关系；
2. 一个集合的适配能否由 25 次全网反传改成十几维矩阵的一次闭式求解。

只在训练集内部选择 ridge alpha，最终 R² 一律在隔离 test 上计算。
"""
from __future__ import annotations

import ast
import argparse
import importlib.util
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R68 = REPO / "DNN_Aggresvation68"
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
R82 = REPO / "DNN_Aggresvation82"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R82 / "src"))
sys.path.insert(0, str(ROOT / "src"))

from runlog import log  # noqa: E402
from truth import load_correct_truth  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "d82_analyze_structured", R82 / "scripts" / "analyze_k0.py"
)
assert spec and spec.loader
d82 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d82)

ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
MODELS = ("linear", "poly2", "arith")


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    spec: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """零附近分母置 0；裁剪阈值只能由训练数据产生。"""
    if spec is None:
        nonzero = np.abs(denominator[np.abs(denominator) > 1e-12])
        scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        eps = max(scale * 1e-3, 1e-8)
    else:
        eps, _, _ = spec
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = np.abs(denominator) > eps
    result[valid] = numerator[valid] / denominator[valid]
    if spec is None:
        finite = result[np.isfinite(result)]
        lo, hi = (
            tuple(np.quantile(finite, [0.005, 0.995]))
            if len(finite)
            else (-1.0, 1.0)
        )
        spec = (eps, float(lo), float(hi))
    _, lo, hi = spec
    result = np.clip(result, lo, hi)
    return (
        np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0),
        spec,
    )


def make_features(
    z: np.ndarray,
    raw: np.ndarray,
    key: tuple[int, ...],
    model: str,
    ratio_specs: list[tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    columns = [z[:, index] for index in key]
    learned_specs: list[tuple[float, float, float]] = []
    if model in {"poly2", "arith"}:
        columns.extend(z[:, index] ** 2 for index in key)
        columns.extend(
            z[:, left] * z[:, right] for left, right in combinations(key, 2)
        )
    if model == "arith":
        ratio_position = 0
        for left in key:
            for right in key:
                if left != right:
                    known = (
                        None
                        if ratio_specs is None
                        else ratio_specs[ratio_position]
                    )
                    ratio, learned = safe_ratio(
                        raw[:, left], raw[:, right], known
                    )
                    columns.append(ratio)
                    learned_specs.append(learned)
                    ratio_position += 1
    return np.column_stack(columns).astype(np.float64), learned_specs


def solve_one(
    train_z: np.ndarray,
    train_raw: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_raw: np.ndarray,
    test_y: np.ndarray,
    key: tuple[int, ...],
    model: str,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """训练内验证逐 conf 选 alpha，再在全部训练数据上重解。"""
    design_train, ratio_specs = make_features(
        train_z, train_raw, key, model
    )
    design_test, _ = make_features(
        test_z, test_raw, key, model, ratio_specs
    )

    fit_mean = design_train[fit_idx].mean(axis=0)
    fit_std = design_train[fit_idx].std(axis=0)
    fit_std[fit_std < 1e-8] = 1.0
    x_fit = (design_train[fit_idx] - fit_mean) / fit_std
    x_val = (design_train[val_idx] - fit_mean) / fit_std
    y_fit = train_y[fit_idx]
    y_mean = y_fit.mean(axis=0)
    yc_fit = y_fit - y_mean
    gram = x_fit.T @ x_fit
    cross = x_fit.T @ yc_fit
    identity = np.eye(gram.shape[0])
    val_mse = []
    for alpha in ALPHAS:
        beta = np.linalg.solve(gram + alpha * identity, cross)
        prediction = x_val @ beta + y_mean
        val_mse.append(((prediction - train_y[val_idx]) ** 2).mean(axis=0))
    best = np.argmin(np.stack(val_mse, axis=0), axis=0)

    all_mean = design_train.mean(axis=0)
    all_std = design_train.std(axis=0)
    all_std[all_std < 1e-8] = 1.0
    x_all = (design_train - all_mean) / all_std
    x_test = (design_test - all_mean) / all_std
    y_mean = train_y.mean(axis=0)
    yc_all = train_y - y_mean
    gram = x_all.T @ x_all
    cross = x_all.T @ yc_all
    prediction = np.zeros_like(test_y, dtype=np.float64)
    for alpha_index, alpha in enumerate(ALPHAS):
        conf = np.flatnonzero(best == alpha_index)
        if not len(conf):
            continue
        beta = np.linalg.solve(gram + alpha * identity, cross[:, conf])
        prediction[:, conf] = x_test @ beta + y_mean[conf]
    residual = ((prediction - test_y) ** 2).sum(axis=0)
    total = ((test_y - test_y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None), ALPHAS[best]


def weighted_corr(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    left = left - np.average(left, weights=weight)
    right = right - np.average(right, weights=weight)
    denominator = np.sqrt(
        np.sum(weight * left**2) * np.sum(weight * right**2)
    )
    return float(np.sum(weight * left * right) / denominator)


def evaluate_model(
    estimates: pd.DataFrame,
    truth_entries: pd.DataFrame,
    truth_triples: pd.DataFrame,
    model: str,
) -> dict[str, object]:
    frame = estimates[estimates.model == model]
    pair_map = {
        (row.indices, row.conf): row.est
        for row in frame[frame["size"] == 2].itertuples()
    }
    parent_map = {
        (row.indices, row.conf): row.est
        for row in frame[frame["size"] == 3].itertuples()
    }
    joined = truth_entries.copy()
    joined["est"] = [
        parent_map[(indices, conf)]
        for indices, conf in zip(joined.indices, joined.conf)
    ]
    joined["best_pair_est"] = [
        max(
            pair_map[(tuple(sorted(pair)), conf)]
            for pair in combinations(indices, 2)
        )
        for indices, conf in zip(joined.indices, joined.conf)
    ]
    joined["syn3_est"] = joined.est - joined.best_pair_est
    w = joined.weight.to_numpy()

    scores = (
        joined.groupby("indices", as_index=False)
        .agg(score=("syn3_est", "max"), syn3_true=("syn3_true", "max"))
        .merge(
            truth_triples[["indices", "weight", "split"]],
            on="indices",
            validate="one_to_one",
        )
    )
    test_scores = scores[scores.split == "test"].copy()
    cutoff = max(1, round(len(test_scores) * 0.30))
    selected = set(test_scores.nlargest(cutoff, "score").indices)
    strong = test_scores.syn3_true > 0.10
    hit = test_scores.indices.isin(selected) & strong
    recall = float(
        test_scores.loc[hit, "weight"].sum()
        / test_scores.loc[strong, "weight"].sum()
    )

    global_recall = np.nan
    if frame.loc[frame["size"] == 3, "indices"].nunique() == 13244:
        all_triples = frame[frame["size"] == 3].copy()
        all_triples["best_pair_est"] = [
            max(
                pair_map[(tuple(sorted(pair)), conf)]
                for pair in combinations(indices, 2)
            )
            for indices, conf in zip(all_triples.indices, all_triples.conf)
        ]
        all_triples["syn3_est"] = all_triples.est - all_triples.best_pair_est
        global_scores = all_triples.groupby("indices").syn3_est.max()
        global_keep = set(global_scores.nlargest(round(13244 * 0.30)).index)
        strong_test = truth_triples[
            (truth_triples.split == "test") & truth_triples.strong
        ]
        global_recall = float(
            strong_test.loc[
                strong_test.indices.isin(global_keep), "weight"
            ].sum()
            / strong_test.weight.sum()
        )

    true_max = joined.loc[joined.groupby("indices").syn3_true.idxmax()]
    extreme = true_max[true_max.syn3_true > 0.20]
    test_joined = joined[joined.split == "test"]
    tw = test_joined.weight.to_numpy()
    return {
        "model": model,
        "test_parent_mae": float(
            np.average(np.abs(test_joined.est - test_joined.parent_true), weights=tw)
        ),
        "test_parent_bias": float(
            np.average(test_joined.est - test_joined.parent_true, weights=tw)
        ),
        "test_syn_mae": float(
            np.average(
                np.abs(test_joined.syn3_est - test_joined.syn3_true), weights=tw
            )
        ),
        "test_s1_spearman": weighted_corr(
            rankdata(test_scores.score),
            rankdata(test_scores.syn3_true),
            test_scores.weight.to_numpy(),
        ),
        "pool_top30_recall": recall,
        "global_top30_recall": global_recall,
        "extreme_parent_error_mean": float(
            (extreme.est - extreme.parent_true).mean()
        ),
        "extreme_pair_error_mean": float(
            (extreme.best_pair_est - extreme.pair_true).mean()
        ),
        "extreme_syn_est_mean": float(extreme.syn3_est.mean()),
        "extreme_negative_count": int((extreme.syn3_est < 0).sum()),
        "truth_parent_exceeded_rate": float((joined.est > joined.parent_true).mean()),
        "truth_pair_exceeded_rate": float(
            (joined.best_pair_est > joined.pair_true).mean()
        ),
    }


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
    started = time.time()
    log(
        "STRUCTURED",
        "START",
        f"显式算术特征闭式低阶攻击者；scope={args.scope}, models={selected_models}",
    )
    data = d82.prepare_data(
        {
            **__import__("yaml").safe_load(
                (R77 / "base.yaml").read_text(encoding="utf-8")
            ),
            "dataset": {
                **__import__("yaml").safe_load(
                    (R77 / "base.yaml").read_text(encoding="utf-8")
                )["dataset"],
                "csv_path": str(
                    REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
                ),
            },
        }
    )
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    train_z = data["train_data"][:, general_idx].astype(np.float64)
    test_z = data["test_data"][:, general_idx].astype(np.float64)
    train_y = data["train_data"][:, conf_idx].astype(np.float64)
    test_y = data["test_data"][:, conf_idx].astype(np.float64)
    means = data["mean"][data["general"]].to_numpy(dtype=np.float64)
    stds = data["std"][data["general"]].to_numpy(dtype=np.float64)
    train_raw = train_z * stds + means
    test_raw = test_z * stds + means
    conf_names = data["confidential"]

    truth_entries, truth_triples = load_correct_truth()
    triple_keys = (
        list(combinations(range(len(general_idx)), 3))
        if args.scope == "all"
        else sorted(set(truth_entries.indices))
    )
    keys = list(combinations(range(len(general_idx)), 2)) + triple_keys
    permutation = np.random.RandomState(20260724).permutation(len(train_z))
    n_val = round(len(permutation) * 0.15)
    val_idx = permutation[:n_val]
    fit_idx = permutation[n_val:]

    rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    for done, key in enumerate(keys, 1):
        for model in selected_models:
            r2, alphas = solve_one(
                train_z,
                train_raw,
                train_y,
                test_z,
                test_raw,
                test_y,
                key,
                model,
                fit_idx,
                val_idx,
            )
            for conf, estimate, alpha in zip(conf_names, r2, alphas):
                rows.append(
                    {
                        "indices": repr(tuple(key)),
                        "size": len(key),
                        "model": model,
                        "conf": conf,
                        "est": float(estimate),
                    }
                )
                alpha_rows.append(
                    {
                        "indices": repr(tuple(key)),
                        "model": model,
                        "conf": conf,
                        "alpha": float(alpha),
                    }
                )
        if done % 250 == 0:
            print(f"{done}/{len(keys)} [{time.time() - started:.1f}s]", flush=True)

    estimates = pd.DataFrame(rows)
    suffix = "" if args.scope == "truth" else "_all"
    estimates.to_csv(
        OUT / f"structured_estimates{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(alpha_rows).to_csv(
        OUT / f"structured_alphas{suffix}.csv.gz",
        index=False,
        compression="gzip",
    )
    estimates["indices"] = estimates.indices.map(ast.literal_eval)
    summary = pd.DataFrame(
        [
            evaluate_model(estimates, truth_entries, truth_triples, model)
            for model in selected_models
        ]
    )
    summary.to_csv(OUT / f"structured_summary{suffix}.csv", index=False)
    elapsed = time.time() - started
    log("STRUCTURED", "DONE", "闭式低阶攻击者完成", elapsed_s=elapsed)
    print(summary.to_string(index=False))
    print(f"elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
