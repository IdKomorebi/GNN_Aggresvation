# -*- coding: utf-8 -*-
"""DNN75：训练掩码的**尺寸分配**采样器（本子项目唯一的"方法"代码）。

为什么只动采样分布
------------------
把"目标函数 mean→sup"（中期总结 D3）严格推导一遍会发现它塌缩成"改采样分布"（D1）：

  1. y 逐列 z-score 标准化（69/src/data_processing.py:standardize），故 Var(y_c)≈1，
     于是 R²_c(S) ≈ 1 − MSE_c(S)，二者是**仿射关系**；
  2. 73 号实测每一行 bias ≈ −MAE（mlp/ours/K0: MAE 0.059668 / bias −0.059580），
     即 v̂ < v 几乎 100% 成立，于是 MAE = E[v] − E[v̂]，而 E[v] 与训练无关；
  3. 所以 最小化 MAE ≡ 最大化 E_{S,c}[R²] ≡ 最小化 E_{S,c}[MSE] —— 这**正是**现有损失。

结论：损失函数层面没有可改的；唯一自由的是期望里的**测度 E_S**，即哪些子集被赋予权重。
本文件就是在枚举这个测度的不同选择。

（旁证：68 号分带专家给 5× 容量无效 —— 它没改测度；73 号 Bernoulli→两段式 MAE 降 43%
—— 它改了测度。）

签名统一
--------
所有采样器：`(batch:int, nG:int, rng:np.random.RandomState) -> (batch, nG) float32`
"""
from __future__ import annotations

import numpy as np

# 实测查询负载（读 68/scan_triples.py 与 72/direct_greedy.py 得出）：
#   协同扫描  44 + 946 + 13244 = 14234 次查询，全部在 |S| <= 3
#   防护贪心  约 900 次查询，在 |S| ∈ [13, 43]
#   |S| ∈ [4, 12] 几乎不查
WORKLOAD_SMALL_HI = 3
WORKLOAD_LARGE_LO = 13


def _from_sizes(ks: np.ndarray, nG: int, rng: np.random.RandomState) -> np.ndarray:
    """给定每个样本的保留数 k，均匀抽 k 个可见字段。"""
    m = np.zeros((len(ks), nG), dtype=np.float32)
    for i, k in enumerate(ks):
        idx = rng.choice(nG, size=int(k), replace=False)
        m[i, idx] = 1.0
    return m


def _mix(batch: int, rng: np.random.RandomState, p: float,
         lo_a: int, hi_a: int, lo_b: int, hi_b: int) -> np.ndarray:
    """以概率 p 从 U{lo_a..hi_a} 抽 k，否则从 U{lo_b..hi_b} 抽。"""
    pick = rng.random_sample(batch) < p
    ks = np.where(pick,
                  rng.randint(lo_a, hi_a + 1, size=batch),
                  rng.randint(lo_b, hi_b + 1, size=batch))
    return ks


# ---------------------------------------------------------------- #
def s_none(batch, nG, rng):
    """恒为满输入。= 常规模型 / 置零口径 / IGNN 式。阶梯底。"""
    return np.ones((batch, nG), dtype=np.float32)


def s_bern50(batch, nG, rng):
    """每字段独立 Bernoulli(0.5)。尺寸集中在 nG/2，两端训练不足。"""
    m = (rng.random_sample((batch, nG)) < 0.5).astype(np.float32)
    empty = m.sum(1) == 0
    if empty.any():                                  # 避免全零掩码
        m[np.where(empty)[0], rng.randint(0, nG, size=int(empty.sum()))] = 1.0
    return m


def s_uniform(batch, nG, rng):
    """两段式尺寸均匀：k ~ U{1..nG} 再均匀取 k 个。
    **必须与 69/src/oracle.py:sample_mask 逐位相同**（控制组，V2 自检会校验）。"""
    ks = rng.randint(1, nG + 1, size=batch)
    return _from_sizes(ks, nG, rng)


def s_logunif(batch, nG, rng):
    """对数均匀：k ≈ exp(U(ln1, ln nG))。无硬截断地把质量偏向小集合。"""
    ks = np.clip(np.round(np.exp(rng.uniform(0.0, np.log(nG), size=batch))), 1, nG).astype(int)
    return _from_sizes(ks, nG, rng)


def s_small50(batch, nG, rng):
    """50% 概率 k ~ U{1..4}，否则 k ~ U{1..nG}。"""
    return _from_sizes(_mix(batch, rng, 0.5, 1, 4, 1, nG), nG, rng)


def s_small80(batch, nG, rng):
    """80% 概率 k ~ U{1..4}，否则 k ~ U{1..nG}。强小集合倾斜。"""
    return _from_sizes(_mix(batch, rng, 0.8, 1, 4, 1, nG), nG, rng)


def s_workload(batch, nG, rng):
    """实测查询负载（双峰）：50% k ~ U{1..3}，50% k ~ U{13..nG}。
    注意实际负载是 ~93%/7%，这里软化到 50/50 —— 极端版本会彻底放弃中段，
    而中段仍是"任意集合查询"这一主张的一部分。极端版见 s_workload_hard。"""
    return _from_sizes(_mix(batch, rng, 0.5, 1, WORKLOAD_SMALL_HI, WORKLOAD_LARGE_LO, nG), nG, rng)


def s_workload_hard(batch, nG, rng):
    """字面负载：90% k ~ U{1..3}，10% k ~ U{13..nG}。中段完全放弃。"""
    return _from_sizes(_mix(batch, rng, 0.9, 1, WORKLOAD_SMALL_HI, WORKLOAD_LARGE_LO, nG), nG, rng)


def s_large50(batch, nG, rng):
    """50% 概率 k ~ U{14..nG}，否则 k ~ U{1..nG}。大集合专家（供路由用）。"""
    return _from_sizes(_mix(batch, rng, 0.5, 14, nG, 1, nG), nG, rng)


SAMPLERS = {
    "none":          s_none,
    "bern50":        s_bern50,
    "uniform":       s_uniform,
    "logunif":       s_logunif,
    "small50":       s_small50,
    "small80":       s_small80,
    "workload":      s_workload,
    "workload_hard": s_workload_hard,
    "large50":       s_large50,
}

# 阶梯（用户要求：对比必须含"无随机失活"）+ 新方案
LADDER = ["none", "bern50", "uniform"]
NEW = ["logunif", "small50", "small80", "workload", "workload_hard", "large50"]
ALL = LADDER + NEW


def size_histogram(name: str, nG: int = 44, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """返回该采样器的尺寸分布（长度 nG+1，索引 = 可见字段数）。供 V2 自检与作图。"""
    rng = np.random.RandomState(seed)
    counts = np.zeros(nG + 1, dtype=np.int64)
    done = 0
    while done < n:
        b = min(20_000, n - done)
        m = SAMPLERS[name](b, nG, rng)
        np.add.at(counts, m.sum(1).astype(int), 1)
        done += b
    return counts / counts.sum()
