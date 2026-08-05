# -*- coding: utf-8 -*-
"""DNN72 Part D：防护集跨 τ 稳定性（零 GPU，现成真值）。

问题：低阶贪心防护集对风险容忍度 τ 是否鲁棒？
- τ∈{0.4,0.5,0.6,0.7} 各自跑低阶贪心到覆盖全部危险项（k*）；
- 前缀 Jaccard 重合度矩阵（前 12 步）；
- τ-鲁棒核心：在所有 τ 的防护集中都出现的字段 → 给数据方的"无论风险容忍度都该防"清单。
"""
import json
import numpy as np
import pandas as pd

SRC = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/outputs"
OUT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation72/outputs"

reg = json.load(open(f"{SRC}/subsets.json"))
tl = pd.read_csv(f"{SRC}/truth_long.csv")
truth = {(r.sid, r.conf): r.dnn for r in tl.itertuples()}
CONFS = sorted(tl.conf.unique())
single_lk, pair_lk, FIELDS = {}, {}, []
for sid, m in reg.items():
    if m["group"] == "single":
        single_lk[m["fields"][0]] = max(truth.get((sid, c), 0.0) for c in CONFS)
        FIELDS.append(m["fields"][0])
    elif m["group"] == "pair":
        pair_lk[tuple(sorted(m["fields"]))] = max(truth.get((sid, c), 0.0) for c in CONFS)
FIELDS = sorted(set(FIELDS))


def greedy_for_tau(tau):
    ds = {f: v for f, v in single_lk.items() if v > tau}
    dp = {fs: v for fs, v in pair_lk.items() if v > tau}
    W = sum(ds.values()) + sum(dp.values())

    def cov(P):
        P = set(P)
        return sum(v for f, v in ds.items() if f in P) + \
            sum(v for (a, b), v in dp.items() if a in P or b in P)

    P, rem, fc = [], set(FIELDS), 0.0
    while fc < W - 1e-9 and rem:
        bf, bg = None, -1
        for f in sorted(rem):
            g = cov(P + [f]) - fc
            if g > bg: bg, bf = g, f
        if bg <= 0: break
        P.append(bf); rem.discard(bf); fc = cov(P)
    return P  # 长度即 k*(tau)


TAUS = [0.4, 0.5, 0.6, 0.7]
seqs = {t: greedy_for_tau(t) for t in TAUS}
for t, P in seqs.items():
    print(f"τ={t}: k*={len(P)}")

# 前缀 Jaccard（前 m 步）
def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0

rows = []
for m in [6, 12, 18]:
    for i, t1 in enumerate(TAUS):
        for t2 in TAUS[i + 1:]:
            rows.append(dict(prefix=m, tau1=t1, tau2=t2,
                             jaccard=jac(seqs[t1][:m], seqs[t2][:m])))
jd = pd.DataFrame(rows)
jd.to_csv(f"{OUT}/tau_stability.csv", index=False)
print("\n前缀 Jaccard：")
print(jd.pivot_table(index=["tau1", "tau2"], columns="prefix", values="jaccard").round(3))

# τ-鲁棒核心：所有 τ 防护集的交集（按 τ=0.7 序位排序）
core = set(seqs[TAUS[0]])
for t in TAUS[1:]:
    core &= set(seqs[t])
core_ranked = [f for f in seqs[0.7] if f in core] + sorted(core - set(seqs[0.7]))
pd.DataFrame(dict(field=core_ranked,
                  single_leak=[single_lk[f] for f in core_ranked])).to_csv(
    f"{OUT}/tau_robust_core.csv", index=False)
print(f"\nτ-鲁棒核心字段（{len(core)} 个，所有 τ 都进防护集）：")
for f in core_ranked:
    print(f"  {f}  (单字段泄露 {single_lk[f]:.3f})")
