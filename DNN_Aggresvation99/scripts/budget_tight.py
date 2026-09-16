# -*- coding: utf-8 -*-
"""P3b：从字段级 M 曲线到集合风险上界——加性预算 vs 逐阶预算，以及与阈值无关的 AUC。

加性预算    B_K(A) = Σ_{i∈A} M_i^(K)                        （|A|≤K+1 时严格上界）
逐阶预算    U_K(∅)=0, U_K(A) = min_{j∈A} [U_K(A\\j) + M_j^(min(|A|−1,K))]
            （同样在 |A|≤K+1 时严格上界，且 U_K ≤ B_K；|A|>K+1 时需要协同阶 κ≤K 才成立）
对照分数    单字段和 Σ a_i（次模假设下的上界）、Shapley 和 Σ φ_i
指标        τ=0.5/0.7/0.9 下的漏判/误拒；以 v̄(A)>τ 为正类的 ROC-AUC（分数尺度无关）；上界违例率与平均松弛
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; R98 = ROOT.parent / "DNN_Aggresvation98"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(R98 / "scripts"))
from mk import envelope, mk_all, shapley, popcount
from analyze import load_truth, CONF

rows = []
for game in ["gameA", "gameB"]:
    Tr = load_truth(game); V = (Tr[(0, "clean")] + Tr[(1, "clean")]) / 2
    p = int(np.log2(len(V))); b = np.arange(2 ** p); sz = popcount(b); bits = ((b[:, None] >> np.arange(p)) & 1).astype(float)
    Vb = envelope(V); M, _ = mk_all(V); phi = shapley(V)
    scores = {"Shapley 和": bits @ phi, "单字段和 Σa": bits @ M[:, 0]}
    for K in [1, 2, 3, p - 1]:
        scores[f"加性预算 ΣM^({K})"] = bits @ M[:, K]
        U = np.zeros_like(V, dtype=np.float64)
        for s in range(1, p + 1):
            idx = b[sz == s]; best = np.full((len(idx), V.shape[1]), np.inf)
            for j in range(p):
                has = (idx >> j) & 1 == 1
                cand = U[idx[has] ^ (1 << j)] + M[j, min(s - 1, K)]
                best[has] = np.minimum(best[has], cand)
            U[idx] = best
        scores[f"逐阶预算 U^({K})"] = U
    bands = [("|A|≤2", sz <= 2), ("|A|≤4", sz <= 4), ("全部", sz >= 1)]
    for nm, S in scores.items():
        for tau in [0.5, 0.7, 0.9]:
            unsafe = Vb > tau
            r = dict(game=game, score=nm, tau=tau)
            for bn, k in bands:
                u, o = unsafe[k], (S <= tau)[k]
                r[f"漏判_{bn}"] = (o & u).sum() / max(u.sum(), 1); r[f"误拒_{bn}"] = (~o & ~u).sum() / max((~u).sum(), 1)
            aucs = [roc_auc_score(unsafe[1:, c], S[1:, c]) for c in range(V.shape[1]) if 0 < unsafe[1:, c].sum() < len(b) - 1]
            r["AUC"] = float(np.mean(aucs)) if aucs else np.nan
            rows.append(r)
        viol = (V - S)[sz >= 1]
        rows[-1].update(上界违例率=float((viol > 1e-9).mean()), 平均松弛=float((S - V)[sz >= 1].mean()))
df = pd.DataFrame(rows); df.to_csv(ROOT / "outputs/P3b_budget_tight.csv", index=False)
show = df[df.tau == 0.7].drop(columns="tau")
print(show.round(3).to_string(index=False))
v = df[df.tau == 0.9][["game", "score", "上界违例率", "平均松弛"]]
print(v.round(4).to_string(index=False))
