# -*- coding: utf-8 -*-
"""124 号：字段风险画像——背景依赖的边际增益、背景放大量与危险背景（只用已有真值与估计，CPU）。
对 10 个目标的每个 (字段 i, 目标 y)：
  Δ_i(T) = V(T∪i) − V(T)，T 取遍其余字段中规模 ≤2 的全部背景（规模 ≤3 真值）；NEM 另用 119 号规模 4 真值得到 M^(3)。
  M^(K) = max_{|T|≤K} Δ_i(T)；背景放大量 Γ^(K) = M^(K) − M^(0)；平均背景增益 Δ̄ = 所有 |T|≤2 背景上 Δ 的平均（"平均语义"的代表）；
  Top-5 危险背景（按 Δ 排序）。
检查：K=1 闭式 Γ^(1) = max(0, max_j I_ij)，I_ij = V(ij) − V(i) − V(j)。
估计器（新主口径 E）：Γ^(2) 误差；估计器前 k 个背景是否包含真实最危险背景（背景召回）；认证前 3 个背景恢复的 Γ 比例。"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
A = os.path.join(ROOT, "outputs"); os.makedirs(A, exist_ok=True)
rows, ctx, dist = [], [], []
for tag, rel, ef, ek in registry.DATASETS:
    d = registry.load_ds(rel, ef, ek); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"]); F = list(range(p))
    s3 = d["sel3"]; k3 = [k for k, t in zip(keys, s3) if t]; V3 = d["V"][s3]; Ve = pipe.closure_max(d["Vhat3"], k3)
    M, W, Dt, BK, bsz = pipe.m_table(V3, k3, F, 2); _, _, De, _, _ = pipe.m_table(Ve, k3, F, 2)
    if tag == "NEM":
        O4 = os.path.join(REPO, "DNN_Aggresvation119/groups/nem_k4/outputs")
        k4 = k3 + [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O4, "D.npz"))["keys"]]
        M3 = pipe.m_table(np.load(os.path.join(O4, "V_official_k4.npy")), k4, F, 3)[0][:, 3, :]
    else:
        M3 = pipe.m_table(d["V"], keys, F, 3)[0][:, 3, :]
    idx = {k: r for r, k in enumerate(k3)}; v1 = lambda i, c: V3[idx[(i,)], c]
    for c, y in enumerate(spec["targ"]):
        for i in F:
            dd = Dt[i, :, c]; de = De[i, :, c]; order_t = np.argsort(-dd); order_e = np.argsort(-de)
            # K=1 闭式检查
            I = max(V3[idx[tuple(sorted((i, j)))], c] - v1(i, c) - v1(j, c) for j in F if j != i)
            M2e = de[bsz <= 2].max(); M0e = de[bsz == 0][0]
            top_true = order_t[0]
            rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), 字段=spec["cand"][i], M0=M[i, 0, c], M1=M[i, 1, c], M2=M[i, 2, c], M3=M3[i, c],
                             G1=M[i, 1, c] - M[i, 0, c], G2=M[i, 2, c] - M[i, 0, c], G3=M3[i, c] - M[i, 0, c],
                             平均背景增益=float(dd.mean()), 中位背景增益=float(np.median(dd)),
                             放大背景占比=float((dd > M[i, 0, c] + 0.01).mean()),
                             K1闭式误差=abs((M[i, 1, c] - M[i, 0, c]) - max(0.0, I)),
                             G2估计=M2e - M0e, 背景召回前1=float(top_true in order_e[:1]), 背景召回前3=float(top_true in order_e[:3]),
                             背景召回前5=float(top_true in order_e[:5]),
                             G2认证前3=float(dd[order_e[:3]].max() - M[i, 0, c]) if M[i, 2, c] - M[i, 0, c] > 1e-9 else np.nan))
            for r_, j in enumerate(order_t[:5]):
                ctx.append(dict(数据=tag, 目标=y.replace("Y_", ""), 字段=spec["cand"][i], 名次=r_ + 1,
                                背景=" + ".join(spec["cand"][t] for t in BK[i][j]) or "∅", Δ=float(dd[j])))
            if tag in ("RTS-GMLC", "NEM", "CAISO-load"):
                for j in range(len(dd)):
                    dist.append(dict(数据=tag, 目标=y.replace("Y_", ""), 字段=spec["cand"][i], 背景规模=int(bsz[j]), Δ=float(dd[j])))
    print("完成", tag, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(A, "profile.csv"), index=False)
pd.DataFrame(ctx).to_csv(os.path.join(A, "top_contexts.csv"), index=False)
pd.DataFrame(dist).to_csv(os.path.join(A, "context_gain_distribution.csv"), index=False)
R["G2误差"] = (R.G2估计 - R.G2).abs(); R["G2认证比"] = R.G2认证前3 / R.G2.where(R.G2 > 1e-9)
S = R.groupby(["数据", "目标"]).agg(字段数=("字段", "size"), M0均值=("M0", "mean"), G1均值=("G1", "mean"), G2均值=("G2", "mean"),
                                   G3均值=("G3", "mean"), G2最大=("G2", "max"), 平均背景增益均值=("平均背景增益", "mean"),
                                   K1闭式最大误差=("K1闭式误差", "max"), G2误差=("G2误差", "mean"),
                                   背景召回前1=("背景召回前1", "mean"), 背景召回前3=("背景召回前3", "mean"), 背景召回前5=("背景召回前5", "mean")).reset_index()
S.to_csv(os.path.join(A, "profile_summary.csv"), index=False); print(S.round(3).to_string(index=False))
num = R.G2认证前3.sum() / R.G2[R.G2 > 1e-9].sum(); print("认证前 3 个背景恢复的 Γ^(2) 比例（按总量）", round(num, 3))
