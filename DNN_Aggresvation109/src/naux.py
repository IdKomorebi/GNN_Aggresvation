# -*- coding: utf-8 -*-
"""109 号：攻击者辅助标签量（n_aux）受限时的数据构造。

威胁模型：攻击者只掌握 n_aux 条带敏感标签的历史辅助数据，他必须用这 n_aux 条
同时完成「训练」与「模型选择」；审计方的测试集不变（代表总体），仅用于最终报告。

实现要点：
  1. 子集**嵌套**——10% ⊂ 25% ⊂ 50% ⊂ 100%，用同一个 permutation 取前 k 条，
     从而消除比例之间的采样随机性，让差异只来自样本量本身。
  2. fit/val 在 n_aux 子集**内部**再划 15%，攻击者不能借用子集外的数据选轮次。
  3. 标准化沿用 69 号 prepare_data 的全 train 统计量（简化）：标准化是线性变换，
     DNN 首层可吸收；此简化只影响输入尺度，不给攻击者额外的标签信息。
"""
from __future__ import annotations
import numpy as np

NAUX_SEED = 20260918
VAL_FRAC = 0.15


def subsample(D: dict, frac: float) -> dict:
    """返回攻击者只有 frac 比例辅助数据时的 D（test 不变）。frac=1.0 时与原 D 的样本集合相同。"""
    n = len(D["Xtr"])
    k = max(32, int(round(n * frac)))
    sub = np.random.RandomState(NAUX_SEED).permutation(n)[:k]      # 嵌套
    perm = np.random.RandomState(NAUX_SEED + 1).permutation(k)
    nv = max(8, int(round(k * VAL_FRAC)))
    E = dict(D)
    E["Xtr"] = np.ascontiguousarray(D["Xtr"][sub]); E["Ytr"] = np.ascontiguousarray(D["Ytr"][sub])
    E["fit_idx"] = perm[nv:]; E["val_idx"] = perm[:nv]
    E["n_aux"] = k
    return E
