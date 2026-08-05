# -*- coding: utf-8 -*-
"""90 号：高阶不可约增量的【偏回归 t/F 检验】——替代 R² 差值。

为什么换：89 号（Codex）实证 syn=R²差 在强尾部的 search→audit Spearman 为**负**
（−0.03 / −0.12），即当前统计量在最需要它的地方不可靠。原因有三：
  1. R² 差不校正自由度——高阶集合参数更多，差值天生有正偏；
  2. 没有噪声尺度——syn=0.15 在噪声大的地方可能完全不显著；
  3. max over 12 conf + 百万级挑极值 = 双重赢家诅咒。
文献亦明确：测高阶交互**必须同时放入所有低阶项**，否则方差被错误归因给高阶。

本模块用统计学的标准答案：对集合 S（|S|=m）与目标 c，
以 S 的**全部非空子集乘积项**为基（2^m−1 个），
检验**最高阶那一项**在控制所有低阶项后的偏回归系数是否显著。

用 Frisch–Waugh–Lovell 的 Gram 分块闭式（数值稳定、无需显式 QR 每个 conf）：
    G = TᵀT, H = Tᵀy, 分块 G=[[G_ll,g_lh],[g_hl,g_hh]], H=[H_l; h_h]
    d   = g_hh − g_hl G_ll⁻¹ g_lh                （= 高阶项对低阶正交化后的能量）
    num = h_h  − g_hl G_ll⁻¹ H_l                 （= 正交化高阶项与 y 的内积，逐 conf）
    β   = num / d,  SSE_full = yᵀy − HᵀG⁻¹H
    se² = SSE_full/(n−p) / d,  t = β/se
一次 Gram 分解服务全部 12 个 conf。**全闭式、不需要任何微调。**

t 有已知零分布 ⟹ 可出 p 值 ⟹ 可做 Benjamini–Hochberg FDR 控制，
这正面解决"百万级组合里挑极值"的多重比较问题。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def build_terms(Z: np.ndarray, m: int) -> np.ndarray:
    """由 m 个（已中心化的）字段构造全部 2^m−1 个非空子集乘积项，最高阶排在最后。"""
    cols = []
    for r in range(1, m + 1):
        for combo in combinations(range(m), r):
            cols.append(np.prod(Z[:, combo], axis=1))
    return np.column_stack(cols)


def highorder_t(Z: np.ndarray, Y: np.ndarray, ridge: float = 1e-8):
    """返回 (t, beta, partial_r2)，每个都是长度 nC 的向量（逐 confidential）。

    Z: (n, m) 该集合的字段（本函数内部中心化+标准化）
    Y: (n, nC) 目标（内部中心化）
    最高阶项 = 全部 m 个字段的乘积。
    """
    n, m = Z.shape
    Z = Z - Z.mean(0)
    sd = Z.std(0)
    sd[sd < 1e-12] = 1.0
    Z = Z / sd

    T = build_terms(Z, m)
    T = T - T.mean(0)
    ts = T.std(0)
    ts[ts < 1e-12] = 1.0
    T = T / ts
    Yc = Y - Y.mean(0)

    p = T.shape[1]
    G = T.T @ T + ridge * np.eye(p)
    H = T.T @ Yc
    yy = np.einsum("ij,ij->j", Yc, Yc)

    G_ll = G[:-1, :-1]
    g_lh = G[:-1, -1]
    g_hh = G[-1, -1]
    H_l = H[:-1, :]
    h_h = H[-1, :]

    try:
        sol_lh = np.linalg.solve(G_ll, g_lh)          # G_ll⁻¹ g_lh
        sol_Hl = np.linalg.solve(G_ll, H_l)           # G_ll⁻¹ H_l
    except np.linalg.LinAlgError:
        nan = np.full(Y.shape[1], np.nan)
        return nan, nan, nan

    d = float(g_hh - g_lh @ sol_lh)
    if d <= 1e-10:                                    # 高阶项被低阶完全线性解释
        z = np.zeros(Y.shape[1])
        return z, z, z
    num = h_h - g_lh @ sol_Hl                         # (nC,)
    beta = num / d

    # 全模型 SSE（逐 conf）
    try:
        sse_full = yy - np.einsum("ij,ij->j", H, np.linalg.solve(G, H))
    except np.linalg.LinAlgError:
        nan = np.full(Y.shape[1], np.nan)
        return nan, nan, nan
    sse_full = np.clip(sse_full, 1e-12, None)
    df = max(n - p, 1)
    se = np.sqrt(sse_full / df / d)
    t = beta / np.clip(se, 1e-30, None)
    # 偏决定系数：高阶项单独解释掉的那部分
    partial_r2 = (num ** 2 / d) / np.clip(yy, 1e-12, None)
    return t, beta, partial_r2


def t_to_p(t: np.ndarray, df: int) -> np.ndarray:
    """双侧 p 值。"""
    return 2.0 * stats.t.sf(np.abs(t), df)


def bh_fdr(p: np.ndarray, q: float = 0.05):
    """Benjamini–Hochberg：返回 (拒绝布尔数组, 阈值)。"""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = q * np.arange(1, n + 1) / n
    passed = sorted_p <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool), 0.0
    kmax = np.max(np.flatnonzero(passed))
    cutoff = sorted_p[kmax]
    return p <= cutoff, float(cutoff)
