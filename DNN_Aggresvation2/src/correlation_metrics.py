from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau as scipy_kendalltau
except Exception:
    scipy_kendalltau = None

try:
    from sklearn.feature_selection import mutual_info_regression
except Exception:
    mutual_info_regression = None

try:
    import dcor  # type: ignore
except Exception:
    dcor = None


DEFAULT_METRICS = ["pearson", "spearman", "kendall", "nmi", "distance_corr", "hsic"]


def _finite_pair(x_raw: np.ndarray, y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _sample_pair(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(x) <= max_samples:
        return x, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx], y[idx]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson_corr(rx, ry)


def kendall_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if scipy_kendalltau is not None:
        stat = scipy_kendalltau(x, y).statistic
        return 0.0 if not np.isfinite(stat) else float(stat)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    tri = np.triu_indices(len(x), k=1)
    prod = np.sign(dx[tri] * dy[tri])
    return float(np.mean(prod)) if len(prod) else 0.0


def normalized_mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    if mutual_info_regression is not None:
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]

        def entropy(arr: np.ndarray, bins: int = 20) -> float:
            hist, _ = np.histogram(arr, bins=bins)
            prob = hist.astype(float) / max(float(hist.sum()), 1.0)
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log(prob))) if len(prob) else 0.0

        denom = min(entropy(x), entropy(y))
        if denom <= 0 or not np.isfinite(mi):
            return 0.0
        return float(np.clip(mi / denom, 0.0, 1.0))

    hist_xy, _, _ = np.histogram2d(x, y, bins=20)
    pxy = hist_xy / max(float(hist_xy.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    h_min = min(hx, hy)
    return float(np.clip(mi / h_min, 0.0, 1.0)) if h_min > 0 else 0.0


def fallback_distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov = np.sqrt(max(float(np.mean(A * B)), 0.0))
    dvar_x = np.sqrt(max(float(np.mean(A * A)), 0.0))
    dvar_y = np.sqrt(max(float(np.mean(B * B)), 0.0))
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(dcov / denom, 0.0, 1.0))


def distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if dcor is not None:
        try:
            return float(dcor.distance_correlation(x, y))
        except Exception:
            return fallback_distance_corr(x, y)
    return fallback_distance_corr(x, y)


def normalized_hsic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    xs = (x - np.mean(x)) / (np.std(x) + 1e-8)
    ys = (y - np.mean(y)) / (np.std(y) + 1e-8)
    x2 = xs.reshape(-1, 1)
    y2 = ys.reshape(-1, 1)
    K = np.exp(-((x2 - x2.T) ** 2))
    L = np.exp(-((y2 - y2.T) ** 2))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic = np.trace(KH @ LH) / ((n - 1) ** 2)
    norm = np.sqrt(np.trace(KH @ KH) * np.trace(LH @ LH)) / ((n - 1) ** 2)
    if norm <= 1e-12:
        return 0.0
    return float(np.clip(hsic / norm, 0.0, 1.0))


def compute_pair_metrics(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    metrics: Sequence[str],
    sample_size: int,
    expensive_sample_size: int,
    random_state: int,
) -> list[float]:
    x, y = _finite_pair(x_raw, y_raw)
    if len(x) < 3:
        return [0.0 for _ in metrics]
    x_std, y_std = _sample_pair(x, y, sample_size, random_state)
    x_exp, y_exp = _sample_pair(x, y, expensive_sample_size, random_state)

    values: list[float] = []
    for metric in metrics:
        try:
            if metric == "pearson":
                value = abs(pearson_corr(x_std, y_std))
            elif metric == "spearman":
                value = abs(spearman_corr(x_std, y_std))
            elif metric == "kendall":
                value = abs(kendall_corr(x_exp, y_exp))
            elif metric == "nmi":
                value = normalized_mutual_info(x_std, y_std)
            elif metric == "distance_corr":
                value = distance_corr(x_exp, y_exp)
            elif metric == "hsic":
                value = normalized_hsic(x_exp, y_exp)
            else:
                value = 0.0
        except Exception:
            value = 0.0
        values.append(float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0)
    return values


def compute_metric_tensor(
    df: pd.DataFrame,
    columns: Sequence[str],
    metrics: Sequence[str],
    sample_size: int,
    expensive_sample_size: int,
    random_state: int,
    progress_every: int = 100,
) -> np.ndarray:
    n = len(columns)
    k = len(metrics)
    arr = np.zeros((n, n, k), dtype=np.float32)
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        xi = df[columns[i]].to_numpy(dtype=float)
        for j in range(i + 1, n):
            done += 1
            yj = df[columns[j]].to_numpy(dtype=float)
            values = compute_pair_metrics(
                xi,
                yj,
                metrics=metrics,
                sample_size=sample_size,
                expensive_sample_size=expensive_sample_size,
                random_state=random_state + done,
            )
            arr[i, j, :] = values
            arr[j, i, :] = values
            if progress_every and (done == 1 or done % progress_every == 0 or done == total):
                print(f"  relationships {done}/{total}")
    return arr

