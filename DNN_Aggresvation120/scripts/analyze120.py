# -*- coding: utf-8 -*-
"""120 号（只汇报、不进论文）步骤 3：大字段空间（p=87，规模 ≤3 共 109,823 个集合）上的组合风险与扫描—认证。
真值：单目标 DNN + 梯度提升树（目标只有一个，多目标 DNN 与单目标相同），val 选择取最大 + 单调闭包。
估计：单种子读出（scan120.py）。决策与处置的评测口径与 117 号相同（选择只用估计值，真值只用于评价）。"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
O = os.path.join(ROOT, "outputs"); A = os.path.join(O, "analysis")


def hit(mus, p):
    if not mus:
        return 0, set()
    Am = np.zeros((len(mus), p))
    for r, m in enumerate(mus):
        Am[r, list(m)] = 1
    x = milp(c=np.ones(p), constraints=LinearConstraint(Am, lb=1), integrality=np.ones(p), bounds=Bounds(0, 1)).x
    return int(round(x.sum())), set(np.where(x > .5)[0])


spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; p = len(spec["cand"]); fields = list(range(p))
parts = [(np.load(os.path.join(O, f"truth_{n}.npz"))["clean"], np.load(os.path.join(O, f"truth_{n}.npz"))["val"]) for n in ("single", "tree")]
V, share = pipe.official_truth(parts, keys); np.save(os.path.join(O, "V_official.npy"), V)
Ve = pipe.closure_max(np.load(os.path.join(O, "est1.npz"))["E"], keys); sec_e = float(np.load(os.path.join(O, "est1.npz"))["sec"])
M, W, Dt, BK, bsz = pipe.m_table(V, keys, fields, 2); Me, _, De, _, _ = pipe.m_table(Ve, keys, fields, 2)
idx = {k: r for r, k in enumerate(keys)}; c = 0; rows = []
Mt, Mest, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2, top=3)
summary = dict(字段数=p, 集合数=len(keys), 树模型选中比例=share[1], 最强单字段=float(max(V[idx[(i,)], c] for i in fields)),
               规模3最大V=float(V[:, c].max()), V误差=float(np.abs(Ve - V).mean()), V偏差=float((Ve - V).mean()),
               M2误差=float(np.abs(Mest - Mt).mean()), M2排序Spearman=float(pd.Series(Mest).corr(pd.Series(Mt), method="spearman")),
               认证前1=float(L1.sum() / Mt.sum()), 认证前3=float(L3.sum() / Mt.sum()), 估计耗时ms=sec_e * 1000)
for tau in (0.5, 0.7):
    mus = pipe.mus_list(V, keys, fields, c, tau, 3); crit = np.array([any(i in m for m in mus) for i in fields])
    single = np.array([V[idx[(i,)], c] > tau for i in fields]); kopt, _ = hit(mus, p)
    rec = {f"τ{tau}_单字段即危险": int(single.sum()), f"τ{tau}_组合危险字段": int((crit & ~single).sum()), f"τ{tau}_危险小组合": len(mus),
           f"τ{tau}_单字段定级后仍暴露": sum(1 for m in mus if not set(m) & set(np.where(single)[0])), f"τ{tau}_最少扣留": kopt}
    for dlt in (0.0, 0.05):
        thr = tau - dlt; mus_e = pipe.mus_list(Ve, keys, fields, c, thr, 3); ce = np.array([any(i in m for m in mus_e) for i in fields])
        cert = np.zeros(p, bool); n_cert = 0
        for i in np.where(ce)[0]:
            cands = []
            for T in BK[i]:
                if len(T) > 2:
                    continue
                vT = Ve[idx[T], c] if T else 0.0; vTi = Ve[idx[tuple(sorted(T + (i,)))], c]
                if vT <= thr < vTi:
                    cands.append((vTi, T))
            for _, T in sorted(cands, reverse=True)[:3]:
                n_cert += 2
                vT = V[idx[T], c] if T else 0.0
                if vT <= tau < V[idx[tuple(sorted(T + (i,)))], c]:
                    cert[i] = True
        k_e, Wset = hit(mus_e, p)
        rec.update({f"τ{tau}_δ{dlt}_扫描召回": float((ce & crit).sum() / max(crit.sum(), 1)), f"τ{tau}_δ{dlt}_扫描精确": float((ce & crit).sum() / max(ce.sum(), 1)),
                    f"τ{tau}_δ{dlt}_认证召回": float((cert & crit).sum() / max(crit.sum(), 1)), f"τ{tau}_δ{dlt}_认证重训集合数": n_cert,
                    f"τ{tau}_δ{dlt}_估计命中集扣留": k_e, f"τ{tau}_δ{dlt}_估计命中集残余": sum(1 for m in mus if not set(m) & Wset)})
    summary.update(rec)
seq = np.load(os.path.join(REPO, "DNN_Aggresvation118/outputs/est/RTS-GMLC/seq.npz"))["t"]; t_seq = float(seq[:, 0].mean() / 3 + seq[:, 1].mean() / 3)
summary.update(串行重训单集合秒=t_seq, 全枚举串行重训小时=t_seq * len(keys) / 3600, 扫描小时=sec_e * len(keys) / 3600,
               认证重训小时_τ05=t_seq * summary["τ0.5_δ0.0_认证重训集合数"] / 3600)
pd.Series(summary).to_csv(os.path.join(A, "summary120.csv"), header=["值"])
P = pd.DataFrame(dict(字段=spec["cand"], M0=M[:, 0, c], M2=M[:, 2, c], M2估计=Me[:, 2, c], 见证=[" + ".join(spec["cand"][j] for j in W[i][2][c]) for i in fields]))
P.sort_values("M2", ascending=False).to_csv(os.path.join(A, "field_profile120.csv"), index=False)
print(pd.Series(summary).to_string())
