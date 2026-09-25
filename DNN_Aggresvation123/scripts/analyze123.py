# -*- coding: utf-8 -*-
"""123 号 步骤 5：受控机制算例的分析（TAG123=narrow / wide）。
真值 = 单目标 DNN + 梯度提升树（val 选择）+ 单调闭包，11 个字段的全部 2,047 个子集。
  (1) M^(K) 阶梯：K = 0..10（K=10 即完整 MCI）；背景放大量 Γ^(K) = M^(K) − M^(0)；Top-5 危险背景
  (2) 背景依赖：出力字段 P_g 在每个单背景 {j} 下的边际增益 Δ_P({j})，与机理预期对照
  (3) 各打分方法（117 号 score_all）对 11 个字段的排序，与已知机理（P_g 与本节点/邻近电价共同决定目标）对照"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
TAG = os.environ.get("TAG123", "narrow"); OUTD = os.path.join(ROOT, "outputs", TAG); A = os.path.join(OUTD, "analysis"); os.makedirs(A, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "scripts")); import scores  # noqa: E402

if __name__ == "__main__":
    spec = json.load(open(os.path.join(OUTD, "fields.json"), encoding="utf-8")); cand = spec["cand"]; p = len(cand)
    z = np.load(os.path.join(OUTD, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
    parts = []
    for nm in ("single", "tree"):
        t = np.load(os.path.join(OUTD, f"truth_{nm}.npz")); parts.append((t["clean"], t["val"]))
    V, share = pipe.official_truth(parts, keys); np.save(os.path.join(OUTD, "V_official.npy"), V)
    idx = {k: r for r, k in enumerate(keys)}; F = list(range(p)); v = lambda S: V[idx[tuple(sorted(S))], 0] if S else 0.0
    M, W, Dm, BK, bsz = pipe.m_table(V, keys, F, p - 1)
    lad = pd.DataFrame({"字段": cand, **{f"M{K}": M[:, K, 0] for K in range(p)}})
    lad["Γ1"], lad["Γ2"], lad["Γ3"], lad["Γ_MCI"] = lad.M1 - lad.M0, lad.M2 - lad.M0, lad.M3 - lad.M0, lad[f"M{p - 1}"] - lad.M0
    lad["K=2 见证"] = [" + ".join(cand[j] for j in W[i][2][0]) or "∅" for i in F]
    lad["MCI 见证"] = [" + ".join(cand[j] for j in W[i][p - 1][0]) or "∅" for i in F]
    lad.to_csv(os.path.join(A, "ladder.csv"), index=False)
    P = cand.index("P_g (unit 313 output)")
    single_bg = pd.DataFrame([dict(背景=("∅" if j is None else cand[j]), Δ_P=(v([P]) if j is None else v([P, j]) - v([j])),
                                   V背景=(0.0 if j is None else v([j])), V合并=(v([P]) if j is None else v([P, j])))
                              for j in [None] + [j for j in F if j != P]])
    single_bg.to_csv(os.path.join(A, "contextual_gain_P.csv"), index=False)
    ctx = []
    for i in F:
        dd = Dm[i, :, 0]; bs = bsz <= 2; o = np.argsort(-np.where(bs, dd, -9))[:5]
        ctx += [dict(字段=cand[i], 名次=r + 1, 背景=" + ".join(cand[t] for t in BK[i][j]) or "∅", Δ=float(dd[j])) for r, j in enumerate(o)]
    pd.DataFrame(ctx).to_csv(os.path.join(A, "top_contexts.csv"), index=False)
    S, _ = scores.score_all(z["Xtr"], z["Ytr"], z["fit_idx"], z["val_idx"], cand, ["Y_markup"], jobs=12)
    S["M0"], S["M2"], S["MCI"] = M[:, 0, 0], M[:, 2, 0], M[:, p - 1, 0]
    S.to_csv(os.path.join(A, "method_scores.csv"), index=False)
    print("攻击器被选中比例（单目标 DNN / 树）", np.round(share, 3)); print("最大 V", V[:, 0].max().round(3))
    print(lad[["字段", "M0", "M1", "M2", "M3", f"M{p - 1}", "Γ2", "K=2 见证"]].round(3).to_string(index=False))
    print(single_bg.round(3).to_string(index=False))
    print(S.drop(columns="目标").round(3).to_string(index=False))
