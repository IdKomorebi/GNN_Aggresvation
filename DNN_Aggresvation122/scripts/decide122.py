# -*- coding: utf-8 -*-
"""122 号 步骤 8：决策层面的比较——与 117 号完全相同的扫描—认证规则，只替换估计器。
关键字段：规模 ≤3 的 τ-最小不安全集合的成员（真值）。扫描：估计值上阈值 τ−δ 处的关键字段；
认证：对扫描标记的字段取估计值上越过 τ−δ 的背景中 V̂(T∪i) 最大的前 3 个，查重训真值确认 V(T)≤τ<V(T∪i)。
另报 M̂^(2) 作为排序的 R-精确率（τ=0.7）。估计器：uniformE（论文口径 E）、uniformE_cy、reconE_cy、recon+random_cy、random_cy。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
EST = ["random", "reconE", "uniformE", "uniformE_cy", "random_cy", "reconE_cy", "recon+random_cy", "recon+random@1_cy", "recon+random@2_cy"]; TAUS = (0.5, 0.7)
rows = []
for tag, rel, ef, ek in registry.DATASETS:
    T_ = tag.replace("/", "_"); E = os.path.join(ROOT, "outputs", "est", T_)
    d = registry.load_ds(rel); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"]); F = list(range(p))
    k3 = [k for k in keys if len(k) <= 3]; V3 = d["V"][np.array([len(k) <= 3 for k in keys])]; idx3 = {k: r for r, k in enumerate(k3)}
    n3 = json.load(open(os.path.join(E, "eval_sets.json")))["n3"]
    M, _, Dt, BK, bsz = pipe.m_table(V3, k3, F, 2)
    for v in EST:
        Ve = pipe.closure_max(np.load(os.path.join(E, f"{v}.npz"))["est"][:n3], k3); Me = pipe.m_table(Ve, k3, F, 2)[0]
        for c, y in enumerate(spec["targ"]):
            for tau in TAUS:
                mus = pipe.mus_list(V3, k3, F, c, tau, 3); crit = np.array([any(i in m for m in mus) for i in F])
                s = Me[:, 2, c]; rp = float(crit[np.argsort(-s, kind="stable")[:crit.sum()]].mean()) if crit.sum() else np.nan
                for dlt in (0.0, 0.05):
                    thr = tau - dlt; mus_e = pipe.mus_list(Ve, k3, F, c, thr, 3)
                    crit_e = np.array([any(i in mm for mm in mus_e) for i in F]); cert = np.zeros(p, bool)
                    for i in np.where(crit_e)[0]:
                        cands = []
                        for T in BK[i]:
                            if len(T) > 2:
                                continue
                            vT = Ve[idx3[T], c] if T else 0.0; vTi = Ve[idx3[tuple(sorted(T + (i,)))], c]
                            if vT <= thr < vTi:
                                cands.append((vTi, T))
                        for _, T in sorted(cands, reverse=True)[:3]:
                            vT = V3[idx3[T], c] if T else 0.0
                            if vT <= tau < V3[idx3[tuple(sorted(T + (i,)))], c]:
                                cert[i] = True
                    for rule, flag in [("扫描标记（估计）", crit_e), ("扫描—认证标记", cert)]:
                        rows.append(dict(数据=tag, 目标=y, 估计器=v, τ=tau, δ=dlt, 规则=rule, 关键数=int(crit.sum()), 标记数=int(flag.sum()),
                                         召回=float((flag & crit).sum() / crit.sum()) if crit.sum() else np.nan,
                                         精确=float((flag & crit).sum() / flag.sum()) if flag.sum() else np.nan, R精确率=rp))
    print("完成", tag, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "analysis", "decisions.csv"), index=False)
P = R.groupby(["规则", "δ", "τ", "估计器"])[["召回", "精确", "R精确率"]].mean().round(3)
print(P.to_string())
