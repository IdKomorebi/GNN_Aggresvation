"""
数据加载与预处理模块。

职责：
- 读取CSV，删除指定列与常量列
- 划分General与Confidential字段
- 切分训练集/测试集（支持 temporal 时序切分与 shuffle 随机切分）
- 标准化（仅使用训练集统计量）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_and_preprocess(
    csv_path: str,
    drop_columns: list[str],
    confidential_columns: list[str],
    drop_constant: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """读取CSV并预处理。返回 (df, general字段列表, confidential字段列表)。"""
    df = pd.read_csv(csv_path)

    existing_drops = [c for c in drop_columns if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    df = df.select_dtypes(include=[np.number])

    if drop_constant:
        stds = df.std()
        constant_cols = stds[stds < 1e-10].index.tolist()
        if constant_cols:
            print(f"  删除常量列 ({len(constant_cols)}): {constant_cols}")
            df = df.drop(columns=constant_cols)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  删除含NaN的行: {before} -> {after}")

    confidential = [c for c in confidential_columns if c in df.columns]
    missing = sorted(set(confidential_columns) - set(confidential))
    if missing:
        print(f"  警告: 以下Confidential字段在数据中未找到: {missing}")
    general = [c for c in df.columns if c not in set(confidential)]

    columns = general + confidential
    df = df[columns]

    return df, general, confidential


def chronological_split(
    df: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分训练集和测试集。"""
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def shuffle_split(
    df: pd.DataFrame, train_ratio: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """随机打乱后切分训练集和测试集，打破时序漂移。"""
    rng = np.random.RandomState(seed)
    n = len(df)
    idx = rng.permutation(n)
    split_idx = int(n * train_ratio)
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def standardize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """使用训练集统计量标准化。返回 (train_array, test_array, mean, std)。"""
    mean = train_df[columns].mean()
    std = train_df[columns].std().replace(0, 1)
    train_arr = ((train_df[columns] - mean) / std).values.astype(np.float32)
    test_arr = ((test_df[columns] - mean) / std).values.astype(np.float32)
    return train_arr, test_arr, mean, std


def prepare_data(cfg: dict) -> dict:
    """主数据准备入口。返回包含所有信息的字典。"""
    csv_path = cfg["dataset"]["csv_path"]
    drop_columns = cfg["fields"].get("drop_columns", [])
    confidential_columns = cfg["fields"]["confidential"]
    drop_constant = cfg["fields"].get("drop_constant_columns", True)

    print("=" * 60)
    print("  [数据准备] 加载并预处理数据")
    print("=" * 60)
    df, general, confidential = load_and_preprocess(
        csv_path, drop_columns, confidential_columns, drop_constant
    )
    print(f"  字段总数: {len(df.columns)}")
    print(f"  General字段: {len(general)}")
    print(f"  Confidential字段: {len(confidential)}")
    print(f"  时间步总数: {len(df)}")

    train_ratio = cfg["training"]["train_ratio"]
    split_mode = str(cfg.get("dataset", {}).get("split_mode", "temporal"))
    seed = int(cfg.get("runtime", {}).get("seed", 42))

    if split_mode == "shuffle":
        train_df, test_df = shuffle_split(df, train_ratio, seed=seed)
        print(f"  切分方式: shuffle (seed={seed})")
    else:
        train_df, test_df = chronological_split(df, train_ratio)
        print(f"  切分方式: temporal (时序)")
    print(f"  训练集: {len(train_df)} 行 ({train_ratio*100:.0f}%)")
    print(f"  测试集: {len(test_df)} 行 ({(1-train_ratio)*100:.0f}%)")

    all_columns = general + confidential
    train_data, test_data, mean, std = standardize(train_df, test_df, all_columns)

    n_general = len(general)
    n_confidential = len(confidential)
    general_indices = list(range(n_general))
    confidential_indices = list(range(n_general, n_general + n_confidential))

    return {
        "train_data": train_data,
        "test_data": test_data,
        "general": general,
        "confidential": confidential,
        "all_columns": all_columns,
        "general_indices": general_indices,
        "confidential_indices": confidential_indices,
        "n_general": n_general,
        "n_confidential": n_confidential,
        "n_nodes": n_general + n_confidential,
        "mean": mean,
        "std": std,
        "raw_df": df,
        "split_mode": split_mode,
    }
