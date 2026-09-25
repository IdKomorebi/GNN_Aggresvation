# -*- coding: utf-8 -*-
"""122 号 步骤 4：各主干变体的评测与诊断汇总。
  规模 ≤3：V 误差（闭包后）、M^(2) 误差与排序、Γ^(2) 误差、认证前 1 / 前 3 下界比；
  规模 4：V 误差（抽样 ≤1,500 个，真值来自各号规模 4 的重训；不做闭包，与真值同口径的原始值比较）；
  表征诊断：失效单元比例、有效维度；大集合（规模 6/10/16，树真值）：V 误差随规模的变化。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
A = os.path.join(ROOT, "outputs", "analysis")
VARS = ["random", "uniform", "small", "recon", "small_recon", "uniform+random", "recon+random", "uniformE", "reconE", "random_cy", "uniform_cy", "recon_cy", "recon+random_cy", "uniformE_cy", "reconE_cy", "recon+random@1_cy", "recon+random@2_cy"]
rows, large = [], []
for tag, rel, ef, ek in registry.DATASETS:
    T = tag.replace("/", "_"); E = os.path.join(ROOT, "outputs", "est", T)
    if not os.path.exists(os.path.join(E, "eval_sets.json")):
        continue
    d = registry.load_ds(rel); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"]); F = list(range(p))
    k3 = [k for k in keys if len(k) <= 3]; s3 = np.array([len(k) <= 3 for k in keys]); V3 = d["V"][s3]
    ev = json.load(open(os.path.join(E, "eval_sets.json"))); n3 = ev["n3"]; k4 = [tuple(int(x) for x in k.split("|")) for k in ev["keys4"]]
    if tag == "NEM":
        O4 = os.path.join(REPO, "DNN_Aggresvation119/groups/nem_k4/outputs"); kk = k3 + [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O4, "D.npz"))["keys"]]
        Vall = np.load(os.path.join(O4, "V_official_k4.npy"))
    else:
        kk, Vall = keys, d["V"]
    ix = {k: r for r, k in enumerate(kk)}; V4 = Vall[[ix[k] for k in k4]]
    _, _, Dt, _, bsz = pipe.m_table(V3, k3, F, 2); s2 = bsz <= 2
    diag = json.load(open(os.path.join(E, "diagnose.json"))) if os.path.exists(os.path.join(E, "diagnose.json")) else {}
    for v in VARS:
        f = os.path.join(E, f"{v}.npz")
        if not os.path.exists(f):
            continue
        est = np.load(f)["est"]; e3 = pipe.closure_max(est[:n3], k3); e4 = np.clip(est[n3:], 0, 1)
        De = pipe.m_table(e3, k3, F, 2)[2]
        for c, y in enumerate(spec["targ"]):
            Mt, Me, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2)
            Gt = Mt - Dt[:, 0, c]; Ge = Me - De[:, 0, c]
            s2i = np.where(bsz <= 2)[0]; ot = s2i[Dt[:, s2i, c].argmax(1)]; oe = np.argsort(-De[:, s2i, c], 1)
            rc1 = float(np.mean([ot[i] in s2i[oe[i, :1]] for i in F])); rc3 = float(np.mean([ot[i] in s2i[oe[i, :3]] for i in F]))
            rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), 变体=v, V误差_3=float(np.abs(e3[:, c] - V3[:, c]).mean()),
                             V误差_4=float(np.abs(e4[:, c] - V4[:, c]).mean()) if len(k4) else np.nan,
                             M2误差=float(np.abs(Me - Mt).mean()), Γ2误差=float(np.abs(Ge - Gt).mean()),
                             M2排序=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")),
                             认证前1=float(L1.sum() / Mt.sum()), 认证前3=float(L3.sum() / Mt.sum()), 背景召回前1=rc1, 背景召回前3=rc3,
                             失效单元=diag.get(v.replace("_cy", ""), [np.nan, np.nan])[0], 有效维度=diag.get(v.replace("_cy", ""), [np.nan, np.nan])[1]))
    fl = os.path.join(ROOT, "outputs", "truth_large", f"{T}.npz")
    if os.path.exists(fl):
        z = np.load(fl); sz = z["sizes"]
        for v in ["random", "uniform", "recon", "small_recon"]:
            for s in np.unique(sz):
                m = sz == s
                large.append(dict(数据=tag, 变体=v, 规模=int(s), V误差=float(np.abs(np.clip(z[v][m], 0, 1) - z["truth"][m]).mean())))
    print("完成", tag, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(A, "variants_by_target.csv"), index=False)
cols = ["V误差_3", "V误差_4", "M2误差", "Γ2误差", "M2排序", "认证前1", "认证前3", "背景召回前1", "背景召回前3", "失效单元", "有效维度"]
Mn = R.groupby("变体")[cols].mean().reindex([v for v in VARS if v in set(R.变体)]); Mn["目标数"] = R.groupby("变体").size()
Mn.to_csv(os.path.join(A, "variants_mean.csv")); print(Mn.round(4).to_string())
print(R.pivot_table(index="变体", columns="数据", values="M2误差").round(4).to_string())
print(R.pivot_table(index="变体", columns="数据", values="认证前1").round(3).to_string())
if large:
    L = pd.DataFrame(large); L.to_csv(os.path.join(A, "large_sets.csv"), index=False)
    print(L.pivot_table(index=["数据", "规模"], columns="变体", values="V误差").round(4).to_string())
