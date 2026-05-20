from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


NODE_STAT_NAMES = [
    "mean",
    "std",
    "max",
    "min",
    "median",
    "q25",
    "q75",
    "missing_rate",
    "diff_mean",
    "diff_std",
    "autocorr1",
    "skew",
    "kurtosis",
]


def _safe_autocorr(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_node_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    n_rows = max(len(df), 1)
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            finite = np.asarray([0.0])
        diffs = np.diff(finite) if len(finite) > 1 else np.asarray([0.0])
        rows.append(
            {
                "field": col,
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "max": float(np.max(finite)),
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "q25": float(np.quantile(finite, 0.25)),
                "q75": float(np.quantile(finite, 0.75)),
                "missing_rate": float(1.0 - len(finite) / n_rows),
                "diff_mean": float(np.mean(diffs)),
                "diff_std": float(np.std(diffs)),
                "autocorr1": _safe_autocorr(values),
                "skew": float(pd.Series(finite).skew()) if len(finite) > 2 else 0.0,
                "kurtosis": float(pd.Series(finite).kurt()) if len(finite) > 3 else 0.0,
            }
        )
    stats = pd.DataFrame(rows)
    for name in NODE_STAT_NAMES:
        values = stats[name].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std <= 1e-8:
            stats[name] = 0.0
        else:
            stats[name] = (values - mean) / std
    return stats

