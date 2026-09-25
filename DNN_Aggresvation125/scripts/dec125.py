# -*- coding: utf-8 -*-
"""【125 号派生自 122 号 principle122.py：只换候选列表（E_old/C1/C2 三实现/C3），输出到 outputs/select】
122 号 步骤 5：原理分析——为什么 V 的改进传不到 M？
对每个 (数据, 目标, 变体)，在规模 ≤3 的集合上（闭包后）取误差 e(S)=V̂(S)−V(S)，
对全部 (T, T∪{i})（|T|≤2）计算：
  |e| 均值（集合值误差）、|d| 均值（d=e(T∪i)−e(T)，即边际 Δ 的误差）、
  ρ = corr(e(T), e(T∪i))（共模程度）、共模占比 = 1 − Var(d)/(2Var(e))。
若某变体 |e| 降而 |d| 不降，说明它消掉的是在差分中本就抵消的共模误差。
另外按真实 V 分档统计严重低估（V̂ < V − 0.15）的个数：读出塌缩（V̂≈0）的离群值直接进入 Δ 与 M。
尾部：M 是对背景取最大，边际误差的上尾（高估）才进入 M̂。按字段取 max_T d(T,i)（最大边际高估）、d 的 99% 分位，
以及 M̂^(2)−M^(2) 的均值（偏差）与绝对值，看 M 的误差是否由"均值好、尾部差"造成。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
A = os.path.join(ROOT, "outputs", "select"); os.makedirs(A, exist_ok=True)
B122 = os.path.join(REPO, "DNN_Aggresvation122")
SRC = {"uniformE": B122, "reconE_cy": B122, "recon+random_cy": B122, "recon+random@1_cy": B122, "recon+random@2_cy": B122, "RRE_cy": ROOT}
VARS = ["uniformE", "reconE_cy", "recon+random_cy", "recon+random@1_cy", "recon+random@2_cy", "RRE_cy"]
rows, pairs_out = [], []
for tag, rel, ef, ek in registry.DATASETS:
    T = tag.replace("/", "_"); E = os.path.join(B122, "outputs", "est", T)
    if not os.path.exists(os.path.join(E, "eval_sets.json")):
        continue
    d = registry.load_ds(rel); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"])
    k3 = [k for k in keys if len(k) <= 3]; s3 = np.array([len(k) <= 3 for k in keys]); V3 = d["V"][s3]
    ix = {k: r for r, k in enumerate(k3)}; ix[()] = -1
    P = [(ix[t], ix[tuple(sorted(t + (i,)))]) for t in k3 + [()] if len(t) <= 2 for i in range(p) if i not in t]
    P = np.array([q for q in P if q[0] >= 0])          # 空背景的 V(∅)=0 无误差，不计入共模统计
    n3 = json.load(open(os.path.join(E, "eval_sets.json")))["n3"]
    for v in VARS:
        f = os.path.join(SRC[v], "outputs", "est", T, f"{v}.npz")
        if not os.path.exists(f):
            continue
        e3 = pipe.closure_max(np.load(f)["est"][:n3], k3)
        for c, y in enumerate(spec["targ"]):
            e = e3[:, c] - V3[:, c]; a, b = e[P[:, 0]], e[P[:, 1]]; dd = b - a
            Dt = V3[P[:, 1], c] - V3[P[:, 0], c]; Mt = np.zeros(p); Me = np.zeros(p)
            for i in range(p):                   # 与主分析同口径：含空背景（Δ(∅)=V({i})），截断在 0 以上
                q = np.array([r for r, t in enumerate(P[:, 1]) if i in k3[t] and i not in k3[P[r, 0]]])
                s1 = ix[(i,)]; Mt[i] = max(V3[s1, c], Dt[q].max()); Me[i] = max(e3[s1, c], (Dt[q] + dd[q]).max())
            fmax = [dd[np.array([r for r, t in enumerate(P[:, 1]) if i in k3[t] and i not in k3[P[r, 0]]])].max() for i in range(p)]
            rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), 变体=v, 集合误差=float(np.abs(e).mean()), 偏差=float(e.mean()),
                             字段最大高估=float(np.mean(fmax)), 边际99分位=float(np.quantile(dd, 0.99)),
                             M2偏差=float((Me - Mt).mean()), M2误差=float(np.abs(Me - Mt).mean()),
                             边际误差=float(np.abs(dd).mean()), 共模相关=float(np.corrcoef(a, b)[0, 1]),
                             共模占比=float(1 - dd.var() / (a.var() + b.var())),
                             严重低估=int((e < -0.15).sum()), 严重高估=int((e > 0.15).sum())))
    print("完成", tag, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(A, "error_decomposition.csv"), index=False)
Mn = R.groupby("变体")[["集合误差", "偏差", "边际误差", "共模相关", "共模占比", "边际99分位", "字段最大高估", "M2偏差", "M2误差", "严重低估"]].mean()
Mn = Mn.reindex([v for v in VARS if v in Mn.index]); Mn.to_csv(os.path.join(A, "error_decomposition_mean.csv"))
print(Mn.round(4).to_string())
print(R.pivot_table(index="变体", columns="数据", values="字段最大高估").round(4).to_string())
print(R.pivot_table(index="变体", columns="数据", values="边际误差").round(4).to_string())
