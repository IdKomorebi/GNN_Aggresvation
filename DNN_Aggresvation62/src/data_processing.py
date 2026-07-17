"""数据加载与预处理模块（沿用 60 号约定，精简为本子项目所需）。

职责：
- 读取CSV，删除指定列与常量列
- 划分General与Confidential字段
- shuffle 切分训练集/测试集（种子与 59/60 号一致）
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

    df = df[general + confidential]
    return df, general, confidential


def shuffle_split(
    df: pd.DataFrame, train_ratio: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """随机打乱后切分训练集和测试集。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    split_idx = int(len(df) * train_ratio)
    return df.iloc[idx[:split_idx]].copy(), df.iloc[idx[split_idx:]].copy()
