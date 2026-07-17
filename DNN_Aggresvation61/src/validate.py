"""链式攻击验证模块。

敏感度分数的操作性检验：分数声称"该字段能(链式)推出机密数据"，
就真的去推，看分数排序与真实攻击成功率排序的吻合度（Spearman）。

攻击者模型（与建边阶段同一批单字段回归器，模拟"攻击者只有成对推断关系、
没有 (源,机密) 联合数据"的威胁场景）：
- 直接攻击: r2_direct(i, c) = W[i, c]（建边阶段已在测试集上算好）
- 两跳链式攻击: ĵ = f_{i->j}(x_i)，ŷ_c = f_{j->c}(ĵ)，对测试集算 R²
- realized(i) = max_c max(直接, 最优两跳)

对照基线：
- s_1hop: max_c W[i, c]，只看一跳（等于直接攻击本身）
- pearson: max_c |皮尔逊相关|，IGNN 式相关系数打分
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def _chained_attack_from(
    src: str,
    general: list[str],
    confidential: list[str],
    W: pd.DataFrame,
    models: dict,
    test_X: dict,
    threshold: float,
) -> tuple[str, float, str]:
    """从 src 出发的最优两跳链式攻击。返回 (字段, 最优R², 最优链描述)。"""
    best_r2, best_desc = 0.0, ""
    x_src = test_X[src].reshape(-1, 1)
    for mid in general:
        if mid == src or W.loc[src, mid] < threshold:
            continue
        j_hat = models[(src, mid)].predict(x_src).reshape(-1, 1)
        for conf in confidential:
            if W.loc[mid, conf] < threshold:
                continue
            y_pred = models[(mid, conf)].predict(j_hat)
            r2 = float(np.clip(r2_score(test_X[conf], y_pred), 0.0, 1.0))
            if r2 > best_r2:
                best_r2 = r2
                best_desc = f"{src} -> {mid} -> {conf} (R2={r2:.3f})"
    return src, best_r2, best_desc


def run_validation(
    W: pd.DataFrame,
    models: dict,
    df_full: pd.DataFrame,
    test_df: pd.DataFrame,
    general: list[str],
    confidential: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """执行链式攻击验证，返回每字段的 realized 攻击成功率与基线分数。"""
    threshold = cfg["chain_edge_threshold"]
    test_X = {c: test_df[c].to_numpy() for c in general + confidential}

    results = Parallel(n_jobs=8, verbose=1)(
        delayed(_chained_attack_from)(
            src, general, confidential, W, models, test_X, threshold
        )
        for src in general
    )

    corr = df_full[general + confidential].corr().abs()
    records = []
    for src, best2hop, desc in results:
        direct = float(W.loc[src, confidential].max())
        records.append({
            "field": src,
            "realized_direct": direct,
            "realized_2hop": best2hop,
            "realized_chained": max(direct, best2hop),
            "best_2hop_chain": desc,
            "s_1hop": direct,
            "pearson_base": float(corr.loc[src, confidential].max()),
        })
    return pd.DataFrame(records).set_index("field")


def score_alignment(merged: pd.DataFrame, score_cols: list[str]) -> dict:
    """各打分与真实链式攻击成功率的 Spearman 相关。"""
    out = {}
    for col in score_cols:
        rho, p = spearmanr(merged[col], merged["realized_chained"])
        out[col] = {"spearman": round(float(rho), 4), "p": float(p)}
    return out
