"""
数据加载与预处理模块。

职责：
- 读取CSV，删除指定列与常量列
- 划分General与Confidential字段
- 按时间顺序切分训练集/测试集
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

    # 删除指定列
    existing_drops = [c for c in drop_columns if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    # 仅保留数值列
    df = df.select_dtypes(include=[np.number])

    # 删除常量列（标准差为零）
    if drop_constant:
        stds = df.std()
        constant_cols = stds[stds < 1e-10].index.tolist()
        if constant_cols:
            print(f"  删除常量列 ({len(constant_cols)}): {constant_cols}")
            df = df.drop(columns=constant_cols)

    # 删除含NaN的行
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  删除含NaN的行: {before} -> {after}")

    # 划分Confidential与General
    confidential = [c for c in confidential_columns if c in df.columns]
    missing = sorted(set(confidential_columns) - set(confidential))
    if missing:
        print(f"  警告: 以下Confidential字段在数据中未找到: {missing}")
    general = [c for c in df.columns if c not in set(confidential)]

    # 重新排列列序：General在前，Confidential在后
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


def standardize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """使用训练集统计量标准化。返回 (train_array, test_array, mean, std)。"""
    mean = train_df[columns].mean()
    std = train_df[columns].std().replace(0, 1)  # 避免除零
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

    # 时间切分
    train_ratio = cfg["training"]["train_ratio"]
    train_df, test_df = chronological_split(df, train_ratio)
    print(f"  训练集: {len(train_df)} 行 ({train_ratio*100:.0f}%)")
    print(f"  测试集: {len(test_df)} 行 ({(1-train_ratio)*100:.0f}%)")

    # 标准化
    all_columns = general + confidential
    train_data, test_data, mean, std = standardize(train_df, test_df, all_columns)

    # 索引信息
    n_general = len(general)
    n_confidential = len(confidential)
    general_indices = list(range(n_general))
    confidential_indices = list(range(n_general, n_general + n_confidential))

    return {
        "train_data": train_data,          # (T_train, N)
        "test_data": test_data,             # (T_test, N)
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
        "raw_df": df,                       # 原始DataFrame（用于计算相关性）
    }
