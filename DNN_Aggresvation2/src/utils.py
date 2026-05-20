from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_yaml(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def make_run_dir(output_root: str | Path) -> Path:
    root = resolve_project_path(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(root / f"run_{stamp}")


def read_numeric_table(
    path: str | Path,
    drop_columns: list[str] | None = None,
    drop_constant_columns: bool = False,
) -> pd.DataFrame:
    path = resolve_project_path(path)
    df = pd.read_csv(path)
    drops = [c for c in (drop_columns or []) if c in df.columns]
    if drops:
        df = df.drop(columns=drops)

    converted = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= 3:
            converted[col] = series.astype(float)
    out = pd.DataFrame(converted)
    out = out.loc[:, out.notna().any(axis=0)]
    if drop_constant_columns:
        keep_cols = []
        for col in out.columns:
            finite = out[col].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if len(finite) >= 3 and np.nanstd(finite) > 1e-12:
                keep_cols.append(col)
        out = out[keep_cols]
    return out


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask].astype(float), y[mask].astype(float)


def chronological_split(n_rows: int, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    n_train = max(1, min(n_rows - 1, int(n_rows * train_ratio)))
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_rows)
    return train_idx, test_idx


def standardize_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    train_filled = np.where(np.isfinite(train), train, mean)
    test_filled = np.where(np.isfinite(test), test, mean)
    return (train_filled - mean) / std, (test_filled - mean) / std, mean, std


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def torch_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
