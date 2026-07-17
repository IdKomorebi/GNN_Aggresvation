"""推断链敏感度模块。

在可推断度图上，把机密字段设为吸收节点，定义：

  链强度(path) = Π(边权) × α^(跳数-1)
  s_max(i)     = max_c 最强推断链强度(i -> c)     —— 主敏感度指标
  s_or(i)      = 1 - Π_c (1 - 最强链强度(i -> c)) —— 多目标聚合暴露度

实现：边成本取 -ln(w·α)（非负），用 Dijkstra 求最小成本路径即最强链，
前驱数组重建路径得到每个字段的"泄露链清单"，解释性由此而来。

另提供矩阵闭式对照 s_walk：T = (I - αQ)^{-1} αR 的行最大值，
Q/R 为 general->general / general->confidential 的加权邻接块。
该量是"全路径强度之和"，可能超过 1，仅作排序对照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def chain_sensitivity(
    W: pd.DataFrame,
    general: list[str],
    confidential: list[str],
    alpha: float,
    edge_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """计算链敏感度。返回 (每字段得分表, {(src, conf): (强度, 路径)} )。"""
    nodes = general + confidential
    idx = {name: k for k, name in enumerate(nodes)}
    n, n_gen = len(nodes), len(general)

    # 有向图：仅 general 有出边；成本 -ln(w*alpha) >= 0
    rows, cols, costs = [], [], []
    for src in general:
        for tgt in nodes:
            if tgt == src:
                continue
            w = W.loc[src, tgt]
            if w >= edge_threshold:
                rows.append(idx[src])
                cols.append(idx[tgt])
                costs.append(-np.log(max(w * alpha, 1e-12)))
    graph = csr_matrix((costs, (rows, cols)), shape=(n, n))

    dist, pred = dijkstra(
        graph, directed=True, indices=list(range(n_gen)),
        return_predecessors=True,
    )

    chains: dict = {}
    records = []
    for gi, src in enumerate(general):
        strengths = {}
        for conf in confidential:
            ci = idx[conf]
            if np.isinf(dist[gi, ci]):
                continue
            strength = float(np.exp(-dist[gi, ci]) / alpha)
            # 由前驱数组回溯路径
            path, cur = [conf], ci
            while cur != gi:
                cur = pred[gi, cur]
                path.append(nodes[cur])
            path.reverse()
            strengths[conf] = strength
            chains[(src, conf)] = (strength, path)
        s_max = max(strengths.values()) if strengths else 0.0
        s_or = 1.0 - float(np.prod([1.0 - v for v in strengths.values()])) \
            if strengths else 0.0
        records.append({"field": src, "s_max": s_max, "s_or": s_or,
                        "n_reachable_conf": len(strengths)})

    scores = pd.DataFrame(records).set_index("field")
    return scores, chains


def walk_sensitivity(
    W: pd.DataFrame,
    general: list[str],
    confidential: list[str],
    alpha: float,
    edge_threshold: float,
) -> tuple[pd.Series, float]:
    """矩阵闭式对照：全路径强度之和 T = (I - αQ)^{-1} αR。

    若 αQ 谱半径 >= 1（级数发散），按 0.99/ρ 缩放并返回缩放比。
    """
    Wt = W.where(W >= edge_threshold, 0.0)
    Q = alpha * Wt.loc[general, general].to_numpy()
    R = alpha * Wt.loc[general, confidential].to_numpy()

    rho = float(np.max(np.abs(np.linalg.eigvals(Q))))
    scale = 1.0
    if rho >= 1.0:
        scale = 0.99 / rho
        Q, R = Q * scale, R * scale

    T = np.linalg.solve(np.eye(len(general)) - Q, R)
    return pd.Series(T.max(axis=1), index=general, name="s_walk"), scale


def format_chain(path: list[str], W: pd.DataFrame) -> str:
    """把路径渲染成带边权的可读字符串。"""
    parts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        parts.append(f" --{W.loc[a, b]:.2f}--> {b}")
    return "".join(parts)
