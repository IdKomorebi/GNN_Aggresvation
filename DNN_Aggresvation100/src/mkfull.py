# -*- coding: utf-8 -*-
"""全规模 M^(K) 计算核心（集合值表 → 单调闭包 → M^(K)、见证背景）。

值表约定：keys[r] 是候选字段索引的升序元组（44 维 general 下标），V[r] 为 (12,) 逐目标值。
空集价值：无基底时为 0；有基底时 keys 中含 ()（仅基底）。

经验攻击能力的单调闭包（论文 §1 口径）：
  V̄(S) = V(argmax_{T⊆S} V_val(T))  —— 在验证集上选子集，测试集上报告，避免在测试集上取 max 的正偏；
  若没有 val 值（估计器），退化为 max_{T⊆S} V(T)。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def index_of(keys):
    return {k: r for r, k in enumerate(keys)}


def subset_matrix(keys, idx, empty_row):
    """(N, 2^kmax−1) 子集行号矩阵（含自身），不足处填 empty_row。"""
    kmax = max(len(k) for k in keys); width = 2 ** kmax
    S = np.full((len(keys), width), empty_row, dtype=np.int64)
    for r, k in enumerate(keys):
        c = 0
        for s in range(0, len(k) + 1):
            for sub in combinations(k, s):
                S[r, c] = idx.get(sub, empty_row); c += 1
    return S


def closure(V, keys, Vval=None, has_empty=False):
    """返回单调闭包 V̄ (N,C)。无基底时空集价值为 0。"""
    idx = index_of(keys); N, C = V.shape
    empty_row = idx[()] if has_empty else N
    Vt = np.vstack([V, np.zeros((1, C), V.dtype)])
    Vs = np.vstack([Vval if Vval is not None else V, np.full((1, C), 0.0 if not has_empty else -np.inf, V.dtype)])
    if not has_empty:
        Vs[N] = 0.0
    S = subset_matrix(keys, idx, empty_row)
    sel = np.take_along_axis(Vs[S], Vs[S].argmax(1)[:, None, :], 1)        # 仅为取 argmax 形状
    arg = Vs[S].argmax(1)                                                    # (N,C) 子集列号
    rows = np.take_along_axis(S, arg, 1)                                     # (N,C) 选中的子集行
    return Vt[rows, np.arange(C)[None, :]]


def mk_table(Vbar, keys, active, K_list=(0, 1, 2, 3), base=False):
    """M[i_pos, K_pos, C] 与见证背景（行号，-1 表示空背景）。active 为候选字段（不含基底）。"""
    idx = index_of(keys); N, C = Vbar.shape
    V0 = Vbar[idx[()]] if base else np.zeros(C)
    kmax = max(len(k) for k in keys)
    M = np.full((len(active), len(K_list), C), -np.inf); W = np.full((len(active), len(K_list), C), -1, np.int64)
    for p, i in enumerate(active):
        others = [a for a in active if a != i]
        best = np.full(C, -np.inf); barg = np.full(C, -1, np.int64); kpos = 0
        for s in range(0, max(K_list) + 1):
            if s + 1 > kmax: break
            Ts = list(combinations(others, s))
            rT = np.array([idx[T] if T else (idx[()] if base else -1) for T in Ts])
            rTi = np.array([idx[tuple(sorted(T + (i,)))] for T in Ts])
            vT = np.where(rT[:, None] >= 0, Vbar[np.maximum(rT, 0)], V0[None, :])
            d = Vbar[rTi] - vT
            j = d.argmax(0); v = d[j, np.arange(C)]
            upd = v > best; best = np.where(upd, v, best); barg = np.where(upd, rT[j], barg)
            if s in K_list:
                M[p, K_list.index(s)] = best; W[p, K_list.index(s)] = barg
    return M, W


def attack_max(vals, vals_val):
    """多攻击器：逐 (集合,目标) 在 val 上选攻击器，测试集报告。vals/vals_val: list of (N,C)。"""
    A = np.stack(vals); Av = np.stack(vals_val)
    pick = Av.argmax(0)
    return np.take_along_axis(A, pick[None], 0)[0], np.take_along_axis(Av, pick[None], 0)[0], pick
