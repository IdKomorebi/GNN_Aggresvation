"""
相关性指标计算模块。

计算5种字段间相关性指标：
- Pearson 线性相关系数
- Spearman 秩相关系数
- Kendall's Tau 秩相关系数
- 归一化互信息 (NMI)
- 距离相关系数 (dCor)

返回形状为 (N, N, 5) 的三维张量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau as scipy_kendalltau
except ImportError:
    scipy_kendalltau = None

try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    mutual_info_regression = None

try:
    import dcor as dcor_lib
except ImportError:
    dcor_lib = None


# ---------- 单指标计算函数 ----------


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _sample(
    x: np.ndarray, y: np.ndarray, max_n: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if max_n is None or len(x) <= max_n:
        return x, y
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx], y[idx]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(x, y)[0, 1]))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson(rx, ry)


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if scipy_kendalltau is not None:
        stat = scipy_kendalltau(x, y).statistic
        return 0.0 if not np.isfinite(stat) else abs(float(stat))
    # 无scipy时的降级实现
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    tri = np.triu_indices(len(x), k=1)
    prod = np.sign(dx[tri] * dy[tri])
    return abs(float(np.mean(prod))) if len(prod) else 0.0


def nmi(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if mutual_info_regression is not None:
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]

        def _entropy(arr: np.ndarray, bins: int = 20) -> float:
            hist, _ = np.histogram(arr, bins=bins)
            prob = hist.astype(float) / max(float(hist.sum()), 1.0)
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log(prob))) if len(prob) else 0.0

        denom = min(_entropy(x), _entropy(y))
        if denom <= 0 or not np.isfinite(mi):
            return 0.0
        return float(np.clip(mi / denom, 0.0, 1.0))

    # 无sklearn时的降级实现
    hist_xy, _, _ = np.histogram2d(x, y, bins=20)
    pxy = hist_xy / max(float(hist_xy.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom_mat = px[:, None] * py[None, :]
    mi_val = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom_mat[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    h_min = min(hx, hy)
    return float(np.clip(mi_val / h_min, 0.0, 1.0)) if h_min > 0 else 0.0


def distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if dcor_lib is not None:
        try:
            return float(dcor_lib.distance_correlation(x, y))
        except Exception:
            pass
    # 降级实现
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov = np.sqrt(max(float(np.mean(A * B)), 0.0))
    dvar_x = np.sqrt(max(float(np.mean(A * A)), 0.0))
    dvar_y = np.sqrt(max(float(np.mean(B * B)), 0.0))
    denom = np.sqrt(dvar_x * dvar_y)
    return float(np.clip(dcov / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0


# ---------- 指标函数映射 ----------

METRIC_FUNCS = {
    "pearson": pearson,
    "spearman": spearman,
    "kendall": kendall,
    "nmi": nmi,
    "distance_corr": distance_corr,
}

# 计算开销较大的指标（使用更小的样本量）
EXPENSIVE_METRICS = {"kendall", "nmi", "distance_corr"}


# ---------- 张量计算 ----------


def compute_metric_tensor(
    df: pd.DataFrame,
    columns: list[str],
    metrics: list[str],
    sample_size: int = 3000,
    expensive_sample_size: int = 1200,
    seed: int = 42,
) -> np.ndarray:
    """
    计算字段间的相关性张量。

    返回: shape=(N, N, K) 的float32数组，N=字段数，K=指标数
    """
    n = len(columns)
    k = len(metrics)
    tensor = np.zeros((n, n, k), dtype=np.float32)
    total = n * (n - 1) // 2
    done = 0
    rng = np.random.default_rng(seed)

    for i in range(n):
        xi = df[columns[i]].to_numpy(dtype=float)
        for j in range(i + 1, n):
            done += 1
            yj = df[columns[j]].to_numpy(dtype=float)
            x_clean, y_clean = _finite_pair(xi, yj)

            if len(x_clean) < 3:
                continue

            # 标准采样与昂贵采样
            x_std, y_std = _sample(x_clean, y_clean, sample_size, rng)
            x_exp, y_exp = _sample(x_clean, y_clean, expensive_sample_size, rng)

            for m_idx, metric_name in enumerate(metrics):
                func = METRIC_FUNCS[metric_name]
                if metric_name in EXPENSIVE_METRICS:
                    val = func(x_exp, y_exp)
                else:
                    val = func(x_std, y_std)
                val = float(np.clip(val, 0.0, 1.0)) if np.isfinite(val) else 0.0
                tensor[i, j, m_idx] = val
                tensor[j, i, m_idx] = val

            if done == 1 or done % 200 == 0 or done == total:
                print(f"  相关性计算进度: {done}/{total}")

    return tensor
