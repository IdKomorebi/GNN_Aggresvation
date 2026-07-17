"""两跳链式攻击张量预计算（62 号孤岛实验的核心加速）。

61 号的链式攻击验证 21min 太慢，因为在双重循环里反复重算第一跳预测。
这里一次性算好两个张量并缓存，之后任意孤岛/遮蔽配置只是对张量掩码 + 取 max：

  R2_direct[i, c]     = W[i, c]                         直接攻击（需 (i,c) 联合数据）
  R2_2hop[i, j, c]    = R²( f_{j->c}( f_{i->j}(x_i) ), c )   经桥接 j 的两跳攻击

其中 f_{i->j}、f_{j->c} 都是 61 号建边阶段训练好的单字段回归器（只需成对联合数据，
即"孤岛内可观测"）。两跳攻击组合它们，全程不需要 (i,c) 联合数据——这正是
跨孤岛拼接攻击的形式化。

约定：i, j 取自 general，c 取自 confidential，j != i。不可用组合置 NaN。
第一跳预测 ĵ = f_{i->j}(x_i) 在测试集上只算一次并缓存复用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score


def _twohop_from_src(
    i: int,
    general: list[str],
    confidential: list[str],
    models: dict,
    test_X: dict,
    y_conf: np.ndarray,
) -> tuple[int, np.ndarray]:
    """从源字段 i 出发，算所有 (桥接 j, 机密 c) 的两跳攻击 R²。"""
    n_gen, n_conf = len(general), len(confidential)
    out = np.full((n_gen, n_conf), np.nan, dtype=np.float64)
    src = general[i]
    x_src = test_X[src].reshape(-1, 1)
    for j, mid in enumerate(general):
        if j == i:
            continue
        # 第一跳：ĵ = f_{i->j}(x_i)，只算一次
        j_hat = models[(src, mid)].predict(x_src).reshape(-1, 1)
        for c, conf in enumerate(confidential):
            # 第二跳：ŷ_c = f_{j->c}(ĵ)
            y_pred = models[(mid, conf)].predict(j_hat)
            out[j, c] = np.clip(r2_score(y_conf[:, c], y_pred), 0.0, 1.0)
    return i, out


def compute_twohop_tensor(
    W: pd.DataFrame,
    models: dict,
    test_df: pd.DataFrame,
    general: list[str],
    confidential: list[str],
    n_jobs: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (R2_direct[n_gen, n_conf], R2_2hop[n_gen, n_gen, n_conf])。"""
    n_gen, n_conf = len(general), len(confidential)
    test_X = {c: test_df[c].to_numpy() for c in general + confidential}
    y_conf = np.column_stack([test_X[c] for c in confidential])

    R2_direct = W.loc[general, confidential].to_numpy().astype(np.float64)

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_twohop_from_src)(i, general, confidential, models, test_X, y_conf)
        for i in range(n_gen)
    )
    R2_2hop = np.full((n_gen, n_gen, n_conf), np.nan, dtype=np.float64)
    for i, mat in results:
        R2_2hop[i] = mat
    return R2_direct, R2_2hop
