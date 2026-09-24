# -*- coding: utf-8 -*-
"""117 号 步骤 2：机理小例子——已知结构的合成数据，看各打分方法分别在哪种结构上失效。

  Y  = a·A·B + b·P + σ·ε          （a²=0.55, b²=0.42, σ²=0.03）
  A, B     : 协同对。单独与 Y 无关（相关、互信息、单字段 R² 都≈0），合起来解释 55% 方差。
  C1, C2   : 冗余对。都是 P 的带噪副本，单独即可解释约 41%；有了一个，另一个几乎不增加信息（LOCO≈0）。
  D        : 间接字段。P 的强噪副本，单独约 21%。
  N        : 纯噪声。
真值：规模 ≤6 的全部 63 个集合，梯度提升树（两种配置 val 选择）专用重训，测试集 R²，单调闭包。
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"   # 共享服务器：每个进程单线程（dcor 会拉起 numba 线程池）
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts")); import scores  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "..", "DNN_Aggresvation111", "src")); import pipe  # noqa: E402

rng = np.random.RandomState(20260924); n = 8000
A, B, P = rng.randn(n), rng.randn(n), rng.randn(n)
Y = np.sqrt(0.55) * A * B + np.sqrt(0.42) * P + np.sqrt(0.03) * rng.randn(n)
C1, C2 = P + 0.15 * rng.randn(n), P + 0.15 * rng.randn(n); Dd = P + 1.0 * rng.randn(n); N = rng.randn(n)
names = ["A", "B", "C1", "C2", "D", "N"]
X = np.stack([A, B, C1, C2, Dd, N], 1).astype(np.float32); Yc = Y[:, None].astype(np.float32)
Dct = pipe.prep(pd.DataFrame(np.concatenate([X, Yc], 1), columns=names + ["Y"]), names, ["Y"])
keys = [s for k in range(1, 7) for s in itertools.combinations(range(6), k)]
if __name__ == "__main__":
    t, v = pipe.truth_tree(Dct, keys, n_jobs=40)
    V, _ = pipe.official_truth([(t, v)], keys)
    idx = {k: r for r, k in enumerate(keys)}
    S, W = scores.score_all(Dct["Xtr"], Dct["Ytr"], Dct["fit_idx"], Dct["val_idx"], names, ["Y"], jobs=12)
    M, Wit, _, _, _ = pipe.m_table(V, keys, list(range(6)), 2)
    S["M0"] = M[:, 0, 0]; S["M1"] = M[:, 1, 0]; S["M2"] = M[:, 2, 0]
    S["M2见证"] = [" + ".join(names[j] for j in Wit[i][2][0]) or "∅" for i in range(6)]
    for tau in (0.5, 0.7):
        mus = pipe.mus_list(V, keys, list(range(6)), 0, tau, 3)
        S[f"τ{tau}关键"] = [any(i in m for m in mus) for i in range(6)]
    S.to_csv(os.path.join(ROOT, "outputs", "toy_scores.csv"), index=False)
    pd.DataFrame(dict(集合=[" + ".join(names[i] for i in k) for k in keys], V=V[:, 0])).to_csv(
        os.path.join(ROOT, "outputs", "toy_truth.csv"), index=False)
    print(S.drop(columns="目标").round(3).to_string(index=False))
    for k in [(0,), (0, 1), (2,), (2, 3), (0, 1, 2), (4,), (0, 1, 4)]:
        print(" + ".join(names[i] for i in k), round(float(V[idx[k], 0]), 3))
