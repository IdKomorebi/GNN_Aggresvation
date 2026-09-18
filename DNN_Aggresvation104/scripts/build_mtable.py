# -*- coding: utf-8 -*-
"""P0-4：正式 M 全表。
对每个 (数据集, 目标 y, 字段 i, K∈{0,1,2})：
  exact M（正式攻击器族真值）、estimated M（主估计器 φ+x,x²三种子集成；另存单种子与元训练）、
  top-3 witness（估计器给出）及其真值边际、档位、是否 τ-critical（τ=0.5/0.7/0.9）、偏差与绝对误差。
K=3 另用 100 号规模 4 真值（多目标 DNN ∪ 树，口径不同）单独出表并标注。
"""
import sys, glob, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100, R102, R103 = REPO / "DNN_Aggresvation100", REPO / "DNN_Aggresvation102", REPO / "DNN_Aggresvation103"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of
from analyze_est import marg_tables
from runlog import log
A = ROOT / "outputs/analysis"; GR = [0.05, 0.2, 0.5]; TAUS = [0.5, 0.7, 0.9]
TIER = ["可忽略(<0.05)", "低(0.05-0.2)", "中(0.2-0.5)", "高(≥0.5)"]


def shards(pat, n, key, ncol=12):
    fs = sorted(glob.glob(pat))
    if not fs: return None
    if len(fs) != int(fs[0].split("of")[-1].split(".")[0]): return None
    out = np.zeros((n, ncol), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


rows, crit_rows = [], []
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb"))
    keys, act, gen, conf = meta["keys_k3"], meta["active"], meta["general"], meta["conf"]; n = len(keys)
    Vt = np.load(R103 / f"outputs/analysis/{ds}_truth_official.npz")["V"]
    est = {"集成": shards(str(R103 / f"outputs/est/{ds}_k3_L0ensx_s*of3.npz"), n, "v"),
           "单种子": shards(str(R102 / f"outputs/est/{ds}_k3_multix_s*of3.npz"), n, "v")}
    e1 = shards(str(R103 / f"outputs/est/{ds}_k3_L1x_s*of3.npz"), n, "v")
    if e1 is not None: est["元训练"] = e1
    Eb = {k: closure(np.clip(v, 0, 1), keys) for k, v in est.items() if v is not None}
    Dt, bk = marg_tables(Vt, keys, act, kmax=2); bsz = np.array([len(T) for T in bk[0]])
    De = {k: marg_tables(v, keys, act, kmax=2)[0] for k, v in Eb.items()}
    idx = index_of(keys)
    for K in [0, 1, 2]:
        sel = bsz <= K; Ts = [bk[0][j] for j in np.where(sel)[0]]
        for p, i in enumerate(act):
            others = [x for x in act if x != i]
            TsK = [T for s in range(K + 1) for T in combinations(others, s)]
            for c in range(12):
                mt = Dt[p, sel, c]; jt = int(np.argmax(mt)); Mt = float(mt[jt])
                r = dict(数据集=ds, 目标=conf[c], 字段=gen[i], K=K, exact_M=Mt,
                         exact_witness=",".join(gen[j] for j in TsK[jt]) or "∅",
                         exact_V_T=float(Vt[idx[TsK[jt]], c]) if TsK[jt] else 0.0,
                         exact_V_Ti=float(Vt[idx[tuple(sorted(TsK[jt] + (i,)))], c]),
                         档位=TIER[int(np.digitize(Mt, GR))])
                for nm, Dx in De.items():
                    me = Dx[p, sel, c]; je = int(np.argmax(me)); r[f"est_M_{nm}"] = float(me[je])
                    r[f"est_bias_{nm}"] = float(me[je]) - Mt; r[f"cert_L_{nm}"] = float(mt[je])
                    if nm == "集成":
                        top3 = np.argsort(-me)[:3]
                        for q, j in enumerate(top3):
                            r[f"top{q+1}_witness"] = ",".join(gen[x] for x in TsK[j]) or "∅"
                            r[f"top{q+1}_est"] = float(me[j]); r[f"top{q+1}_exact"] = float(mt[j])
                for tau in TAUS:
                    vT = np.array([Vt[idx[T], c] if T else 0.0 for T in TsK]); vTi = np.array([Vt[idx[tuple(sorted(T + (i,)))], c] for T in TsK])
                    r[f"critical_τ{tau}"] = bool(((vT <= tau) & (vTi > tau)).any())
                    if "集成" in De:
                        eT = np.array([Eb["集成"][idx[T], c] if T else 0.0 for T in TsK]); eTi = np.array([Eb["集成"][idx[tuple(sorted(T + (i,)))], c] for T in TsK])
                        r[f"est_critical_τ{tau}"] = bool(((eT <= tau) & (eTi > tau)).any())
                rows.append(r)
        log("MTABLE", "DONE", f"{ds} K={K} 完成")
T = pd.DataFrame(rows); T.to_csv(A / "M_table_official.csv", index=False)
summ = []
for (ds, K), d in T.groupby(["数据集", "K"]):
    s = dict(数据集=ds, K=K, 条目数=len(d), exact_M均值=d.exact_M.mean())
    for nm in ["集成", "单种子", "元训练"]:
        if f"est_M_{nm}" in d:
            s[f"MAE_{nm}"] = (d[f"est_M_{nm}"] - d.exact_M).abs().mean(); s[f"bias_{nm}"] = (d[f"est_M_{nm}"] - d.exact_M).mean()
            s[f"认证下界比_{nm}"] = d[f"cert_L_{nm}"].sum() / max(d.exact_M.sum(), 1e-9)
    for tau in TAUS:
        ct, ce = d[f"critical_τ{tau}"], d.get(f"est_critical_τ{tau}")
        s[f"critical数_τ{tau}"] = int(ct.sum())
        if ce is not None and ct.sum():
            s[f"recall_τ{tau}"] = float((ct & ce).sum() / ct.sum()); s[f"precision_τ{tau}"] = float((ct & ce).sum() / max(ce.sum(), 1))
    summ.append(s)
S = pd.DataFrame(summ); S.to_csv(A / "M_table_summary.csv", index=False)
print(S.to_markdown(index=False, floatfmt=".4f"))
print("\n全表行数", len(T), "→", A / "M_table_official.csv")
