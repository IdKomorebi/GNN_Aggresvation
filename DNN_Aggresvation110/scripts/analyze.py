# -*- coding: utf-8 -*-
import itertools, numpy as np, pandas as pd
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
T = pd.read_pickle("v_table.pkl")
df = pd.read_csv("case30_market.csv")
F = [c for c in df.columns if not c.startswith("Y_")]; Y = [c for c in df.columns if c.startswith("Y_")]
p = len(F); idx = {s: r for r, s in enumerate(T.S)}
for y in Y:
    vt = np.clip(T[f"test|{y}"].values, 0, 1); vv = T[f"val|{y}"].values
    # 单调闭包：val 上挑子集，test 上报告
    V = {(): 0.0}; Vval = {(): 0.0}; cl = {}
    for r, s in enumerate(T.S):
        best_s, best_v = s, vv[r]
        for k in range(1, len(s)):
            for sub in itertools.combinations(s, k):
                if vv[idx[sub]] > best_v: best_s, best_v = sub, vv[idx[sub]]
        V[s] = vt[idx[best_s]]
    corr = [abs(np.corrcoef(df[f], df[y])[0, 1]) for f in F]
    print(f"\n==================== 目标 {y} ====================")
    print(f"全部 11 个公开字段合用：V = {V[tuple(range(p))]:.3f}")
    rows = []
    for i in range(p):
        oth = [j for j in range(p) if j != i]
        prof, wit = [], []
        for K in [0, 1, 2, 3, p - 1]:
            best, bt = -9, ()
            for k in range(0, K + 1):
                for Tt in itertools.combinations(oth, k):
                    d = V[tuple(sorted(Tt + (i,)))] - V[Tt]
                    if d > best: best, bt = d, Tt
            prof.append(best); wit.append(bt)
        # 不同背景下的增益分布（K=1）
        g1 = sorted([(V[tuple(sorted((j, i)))] - V[(j,)], F[j]) for j in oth], reverse=True)
        rows.append(dict(字段=F[i], 相关系数=corr[i], M0=prof[0], M1=prof[1], M2=prof[2], M3=prof[3], MCI=prof[4],
                         升级=prof[2] - prof[0],
                         K1最佳伙伴=g1[0][1], 其增益=g1[0][0], K1最差伙伴=g1[-1][1], 其增益_=g1[-1][0],
                         K2见证="+".join(F[j] for j in wit[2]) or "∅"))
    R = pd.DataFrame(rows).sort_values("M2", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(R.round(3).to_string(index=False))
    # K 饱和
    print("K 饱和：mean M2/MCI =", round((R.M2 / R.MCI.clip(1e-9)).mean(), 3), " mean M1/MCI =", round((R.M1 / R.MCI.clip(1e-9)).mean(), 3))
    # τ 下的最小不安全集合
    for tau in [0.5, 0.7, 0.9]:
        mus = [s for s in T.S if V[s] > tau and all(V[sub] <= tau for k in range(1, len(s)) for sub in itertools.combinations(s, k))]
        from collections import Counter
        print(f"τ={tau}: MUS 数={len(mus)}, 规模分布={dict(Counter(len(m) for m in mus))}, 单字段越阈={sum(1 for m in mus if len(m)==1)}")
