"""成对可推断度建边模块。

核心定义：有向边权 w[i->j] = 用单字段 i 的非线性回归器预测字段 j，
在测试集上的 R²（截断到 [0,1]）。含义是"攻击者拿到 i 能把 j 猜到几成准"。

- 非线性由树模型（HistGradientBoosting）承接，不做任何函数形式假设；
- 方向性天然存在（i 推 j 与 j 推 i 的精度可以不同）；
- 只训练推断链会用到的方向：general -> general 与 general -> confidential
  （机密节点是吸收态，无出边）。

产出：
- W: DataFrame（行=源字段，列=目标字段），未过阈值的原始可推断度矩阵
- models: {(src, tgt): fitted_model}，供链式攻击验证阶段复用
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor


def _make_model(cfg: dict):
    if cfg["model"] == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=cfg["hgb_max_iter"],
            early_stopping=False,
            random_state=cfg["seed"],
        )
    if cfg["model"] == "knn":
        return KNeighborsRegressor(n_neighbors=cfg["knn_k"])
    raise ValueError(f"未知边模型: {cfg['model']}")


def _fit_pair(x_tr, y_tr, x_te, y_te, src: str, tgt: str, cfg: dict):
    model = _make_model(cfg)
    model.fit(x_tr, y_tr)
    r2 = r2_score(y_te, model.predict(x_te))
    return src, tgt, float(np.clip(r2, 0.0, 1.0)), model


def build_edge_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    general: list[str],
    confidential: list[str],
    cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """训练全部单字段回归器，返回可推断度矩阵 W 和模型字典。"""
    all_fields = general + confidential
    pairs = [
        (src, tgt) for src in general for tgt in all_fields if tgt != src
    ]
    print(f"  成对回归任务数: {len(pairs)} "
          f"(general={len(general)}, confidential={len(confidential)})")

    tr = {c: train_df[c].to_numpy() for c in all_fields}
    te = {c: test_df[c].to_numpy() for c in all_fields}

    results = Parallel(n_jobs=cfg["n_jobs"], verbose=1)(
        delayed(_fit_pair)(
            tr[src].reshape(-1, 1), tr[tgt],
            te[src].reshape(-1, 1), te[tgt],
            src, tgt, cfg,
        )
        for src, tgt in pairs
    )

    W = pd.DataFrame(0.0, index=general, columns=all_fields)
    models: dict = {}
    for src, tgt, r2, model in results:
        W.loc[src, tgt] = r2
        models[(src, tgt)] = model
    return W, models
