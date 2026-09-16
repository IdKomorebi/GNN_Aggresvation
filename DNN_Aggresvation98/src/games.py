# -*- coding: utf-8 -*-
"""精确博弈算子：输入 V (2^p, C)，第 b 行 = 位掩码 b 的联盟价值。

shapley      φ_i = Σ_{S∌i} |S|!(p-|S|-1)!/p! · [V(S∪i)-V(S)]
sii          Shapley interaction index（Grabisch 1997）二阶
stii         Shapley-Taylor（Sundararajan 2020）二阶顶层项：(2/p) Σ_{S∌i,j} Δ_ij V(S)/C(p-1,|S|)
faith2       Faith-Shap（Tsai 2023）二阶：Shapley 核加权最小二乘，空集/全集硬约束
maxmarg      背景规模 ≤K 的最大边际 M_i^(K) = max_{|T|≤K, i∉T} V(T∪i)-V(T)  （风险分级语义）
loo          全集删除边际 V(N)-V(N\\i)
"""
from __future__ import annotations

from itertools import combinations
from math import comb, factorial

import numpy as np


def popcount(x: np.ndarray) -> np.ndarray:
    return np.array([bin(int(v)).count("1") for v in x])


def _setup(p):
    b = np.arange(2 ** p)
    return b, popcount(b)


def shapley(V: np.ndarray) -> np.ndarray:
    p = int(np.log2(len(V))); b, sz = _setup(p)
    w = np.array([factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)])
    out = np.zeros((p,) + V.shape[1:])
    for i in range(p):
        S = b[(b >> i) & 1 == 0]
        out[i] = (w[sz[S]][:, None] * (V[S | (1 << i)] - V[S])).sum(0)
    return out


def _pair_deltas(V, i, j, b):
    S = b[((b >> i) & 1 == 0) & ((b >> j) & 1 == 0)]
    d = V[S | (1 << i) | (1 << j)] - V[S | (1 << i)] - V[S | (1 << j)] + V[S]
    return S, d


def sii(V):
    p = int(np.log2(len(V))); b, sz = _setup(p)
    w = np.array([factorial(s) * factorial(p - s - 2) / factorial(p - 1) for s in range(p - 1)])
    out = {}
    for i, j in combinations(range(p), 2):
        S, d = _pair_deltas(V, i, j, b)
        out[(i, j)] = (w[sz[S]][:, None] * d).sum(0)
    return out


def stii(V):
    p = int(np.log2(len(V))); b, sz = _setup(p)
    w = np.array([2 / p / comb(p - 1, s) for s in range(p - 1)])
    out = {}
    for i, j in combinations(range(p), 2):
        S, d = _pair_deltas(V, i, j, b)
        out[(i, j)] = (w[sz[S]][:, None] * d).sum(0)
    return out


def faith2(V):
    """返回 (一阶 p×C, 二阶 dict)。"""
    p = int(np.log2(len(V))); b, sz = _setup(p)
    pairs = list(combinations(range(p), 2))
    bits = ((b[:, None] >> np.arange(p)) & 1).astype(float)
    X = np.concatenate([np.ones((len(b), 1)), bits,
                        np.stack([bits[:, i] * bits[:, j] for i, j in pairs], 1)], 1)
    mu = np.zeros(len(b))
    mid = (sz > 0) & (sz < p)
    mu[mid] = (p - 1) / (np.array([comb(p, s) for s in sz[mid]]) * sz[mid] * (p - sz[mid]))
    mu[~mid] = 1e6 * mu[mid].max()
    Xw = X * np.sqrt(mu)[:, None]
    coef = np.linalg.lstsq(Xw, V * np.sqrt(mu)[:, None], rcond=None)[0]
    return coef[1:p + 1], {pr: coef[p + 1 + k] for k, pr in enumerate(pairs)}


def maxmarg(V, K):
    p = int(np.log2(len(V))); b, sz = _setup(p)
    out = np.full((p,) + V.shape[1:], -np.inf)
    for i in range(p):
        S = b[((b >> i) & 1 == 0) & (sz <= K)]
        out[i] = (V[S | (1 << i)] - V[S]).max(0)
    return out


def loo(V):
    p = int(np.log2(len(V))); full = 2 ** p - 1
    return np.stack([V[full] - V[full ^ (1 << i)] for i in range(p)])


def perm_shapley(V, n_perm, rng):
    """置换采样 Shapley；返回 (估计, 查询到的不同联盟数)。"""
    p = int(np.log2(len(V))); est = np.zeros((p,) + V.shape[1:]); seen = set()
    for _ in range(n_perm):
        order = rng.permutation(p); cur = 0; seen.add(0)
        for i in order:
            nxt = cur | (1 << i); est[i] += V[nxt] - V[cur]; seen.add(nxt); cur = nxt
    return est / n_perm, len(seen)


def kernel_shapley(V, n_samp, rng):
    """成对 KernelSHAP（Covert & Lee 2021）：按 Shapley 核抽规模、再抽补集成对，约束 Σφ=v(N)-v(∅)。"""
    p = int(np.log2(len(V))); full = 2 ** p - 1
    ks = np.arange(1, p); pk = (p - 1) / (ks * (p - ks)); pk /= pk.sum()
    rows, seen = [], {0, full}
    for _ in range(n_samp // 2):
        k = rng.choice(ks, p=pk); idx = rng.choice(p, k, replace=False)
        b = int(sum(1 << int(i) for i in idx)); rows += [b, full ^ b]; seen |= {b, full ^ b}
    rows = np.array(rows); Z = ((rows[:, None] >> np.arange(p)) & 1).astype(float)
    y = V[rows] - V[0]; tot = V[full] - V[0]
    A = Z.T @ Z / len(rows); bvec = Z.T @ y / len(rows); one = np.ones(p)
    Ainv = np.linalg.pinv(A)
    lam = (one @ Ainv @ bvec - tot) / (one @ Ainv @ one)
    return (Ainv @ (bvec - one[:, None] * lam[None, :])), len(seen)
