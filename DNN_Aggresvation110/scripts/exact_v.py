# -*- coding: utf-8 -*-
"""全部子集的精确 V（逐子集专用训练，val 选择，test 报告），单调闭包后算 M^(K) 与 witness。"""
import itertools, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
df = pd.read_csv("case30_market.csv")
Y = [c for c in df.columns if c.startswith("Y_")]
F = [c for c in df.columns if not c.startswith("Y_")]
X = df[F].values; n = len(df); p = len(F)
rs = np.random.RandomState(42); perm = rs.permutation(n)
te, tr = perm[: int(.3 * n)], perm[int(.3 * n):]
va, fit = tr[: int(.15 * len(tr))], tr[int(.15 * len(tr)):]
subsets = [s for k in range(1, p + 1) for s in itertools.combinations(range(p), k)]


def run(s):
    out = []
    for y in Y:
        yv = df[y].values
        best = (-9, None)
        for lr, leaf in [(0.05, 20), (0.1, 40)]:
            m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaf, max_iter=400,
                                              early_stopping=True, validation_fraction=0.15, random_state=0)
            m.fit(X[fit][:, s], yv[fit])
            v = r2_score(yv[va], m.predict(X[va][:, s]))
            if v > best[0]: best = (v, m)
        out.append((best[0], r2_score(yv[te], best[1].predict(X[te][:, s]))))
    return s, out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(40) as pool:
        res = pool.map(run, subsets, chunksize=4)
    print(f"{len(subsets)} 个子集 × {len(Y)} 目标，用时 {time.time()-t0:.0f}s")
    rows = []
    for s, out in res:
        r = {"S": s, "k": len(s)}
        for y, (vv, vt) in zip(Y, out):
            r[f"val|{y}"] = vv; r[f"test|{y}"] = vt
        rows.append(r)
    T = pd.DataFrame(rows); T.to_pickle("v_table.pkl")
    print("已保存 v_table.pkl；字段:", F)
