# -*- coding: utf-8 -*-
"""99 号算子：背景受限最大边际 M^(K) 及其数学关系的精确计算。

记号：V 形状 (2^p, C)，第 b 行 = 位掩码 b 的联盟价值；v(∅)=V[0]=0。
  Δ_i(T)      = v(T∪i) − v(T)
  M_i^(K)     = max_{T⊆N\\i, |T|≤K} Δ_i(T)
  v̄(S)        = max_{T⊆S} v(T)                 （单调包络：攻击者可以丢弃字段）
  m(U)        = Σ_{W⊆U} (−1)^{|U|−|W|} v(W)   （Möbius / Harsanyi 红利）
"""
from __future__ import annotations

import numpy as np


def popcount(x):
    x = np.asarray(x, dtype=np.int64); c = np.zeros_like(x)
    while np.any(x):
        c += x & 1; x = x >> 1
    return c


def envelope(V):
    """v̄(S)=max_{T⊆S} v(T)，子集最大 zeta 变换 O(p·2^p)。"""
    G = V.copy(); p = int(np.log2(len(V))); b = np.arange(len(V))
    for i in range(p):
        has = (b >> i) & 1 == 1
        G[has] = np.maximum(G[has], G[b[has] ^ (1 << i)])
    return G


def mobius(V):
    """快速 Möbius 变换：m(U)=Σ_{W⊆U}(−1)^{|U\\W|}v(W)。"""
    m = V.astype(np.float64).copy(); p = int(np.log2(len(V))); b = np.arange(len(V))
    for i in range(p):
        has = (b >> i) & 1 == 1
        m[has] -= m[b[has] ^ (1 << i)]
    return m


def mk_all(V):
    """返回 M (p, p, C)：M[i, K] = M_i^(K)，K=0..p−1；以及取到最大值的背景位掩码 arg (p, p, C)。"""
    p = int(np.log2(len(V))); b = np.arange(len(V)); sz = popcount(b)
    C = V.shape[1]; M = np.zeros((p, p, C)); arg = np.zeros((p, p, C), dtype=np.int64)
    for i in range(p):
        S = b[(b >> i) & 1 == 0]; d = V[S | (1 << i)] - V[S]; s = sz[S]
        best = np.full(C, -np.inf); barg = np.zeros(C, dtype=np.int64)
        for K in range(p):
            k = s == K
            if k.any():
                dk = d[k]; j = dk.argmax(0); v = dk[j, np.arange(C)]
                upd = v > best; best = np.where(upd, v, best); barg = np.where(upd, S[k][j], barg)
            M[i, K] = best; arg[i, K] = barg
    return M, arg


def shapley(V):
    from math import factorial
    p = int(np.log2(len(V))); b = np.arange(len(V)); sz = popcount(b)
    w = np.array([factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)])
    out = np.zeros((p, V.shape[1]))
    for i in range(p):
        S = b[(b >> i) & 1 == 0]; out[i] = (w[sz[S]][:, None] * (V[S | (1 << i)] - V[S])).sum(0)
    return out


def minimal_unsafe(V, tau):
    """最小不安全集合：v(E)>τ 且所有真子集 v≤τ（用子集最大判定，不依赖单调性）。"""
    p = int(np.log2(len(V))); b = np.arange(len(V)); G = envelope(V)
    maxproper = np.full(V.shape, -np.inf)
    for i in range(p):
        has = (b >> i) & 1 == 1
        maxproper[has] = np.maximum(maxproper[has], G[b[has] ^ (1 << i)])
    maxproper[0] = -np.inf
    return (V > tau) & (maxproper <= tau)


def extend_with_copies(V, src, m):
    """在博弈里加入 m 个 src 号玩家的精确副本（新玩家位 p..p+m−1），v'(S)=v(π(S))。"""
    p = int(np.log2(len(V))); n = p + m; b = np.arange(2 ** n)
    base = b & ((1 << p) - 1)
    anycopy = ((b >> p) & ((1 << m) - 1)) > 0
    return V[np.where(anycopy, base | (1 << src), base)]
