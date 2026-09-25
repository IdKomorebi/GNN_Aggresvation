# -*- coding: utf-8 -*-
"""125 号 步骤 2：按 base.yaml 事先写定的规则比较候选估计器，固定论文估计器。
指标口径与 122 号 analyze122 / decide122 完全相同（规模 ≤3 闭包后；规模 4 抽样不闭包；扫描—认证规则同 117 号）。
C2 按"方法"评价：种子 0/1/2 三个实现的指标取平均。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
B122 = os.path.join(REPO, "DNN_Aggresvation122")
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
A = os.path.join(ROOT, "outputs", "select"); os.makedirs(A, exist_ok=True)
CAND = {"E_old": [("122", "uniformE")], "C1": [("122", "reconE_cy")],
        "C2": [("122", "recon+random_cy"), ("122", "recon+random@1_cy"), ("122", "recon+random@2_cy")],
        "C3": [("125", "RRE_cy")]}


def est_path(src, T, v):
    return os.path.join(B122 if src == "122" else ROOT, "outputs", "est", T, f"{v}.npz")


def crit_cert(V3, Ve, k3, idx3, BK, F, c, tau, dlt):
    thr = tau - dlt; mus = pipe.mus_list(V3, k3, F, c, tau, 3); crit = np.array([any(i in m for m in mus) for i in F])
    mus_e = pipe.mus_list(Ve, k3, F, c, thr, 3); crit_e = np.array([any(i in m for m in mus_e) for i in F]); cert = np.zeros(len(F), bool)
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
    rec = float((cert & crit).sum() / crit.sum()) if crit.sum() else np.nan
    pre = float((cert & crit).sum() / cert.sum()) if cert.sum() else np.nan
    return rec, pre


rows = []
for tag, rel, ef, ek in registry.DATASETS:
    T_ = tag.replace("/", "_"); d = registry.load_ds(rel); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"]); F = list(range(p))
    k3 = [k for k in keys if len(k) <= 3]; V3 = d["V"][np.array([len(k) <= 3 for k in keys])]; idx3 = {k: r for r, k in enumerate(k3)}
    ev = json.load(open(os.path.join(B122, "outputs", "est", T_, "eval_sets.json"))); n3 = ev["n3"]
    k4 = [tuple(int(x) for x in k.split("|")) for k in ev["keys4"]]
    if tag == "NEM":
        O4 = os.path.join(REPO, "DNN_Aggresvation119/groups/nem_k4/outputs")
        kk = k3 + [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O4, "D.npz"))["keys"]]; Vall = np.load(os.path.join(O4, "V_official_k4.npy"))
    else:
        kk, Vall = keys, d["V"]
    ixa = {k: r for r, k in enumerate(kk)}; V4 = Vall[[ixa[k] for k in k4]]
    _, _, Dt, BK, bsz = pipe.m_table(V3, k3, F, 2); s2i = np.where(bsz <= 2)[0]
    for cand, lst in CAND.items():
        for src, v in lst:
            f = est_path(src, T_, v)
            if not os.path.exists(f):
                continue
            est = np.load(f)["est"]; Ve = pipe.closure_max(est[:n3], k3); e4 = np.clip(est[n3:], 0, 1)
            De = pipe.m_table(Ve, k3, F, 2)[2]
            for c, y in enumerate(spec["targ"]):
                Mt, Me, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2)
                Gt = Mt - Dt[:, 0, c]; Ge = Me - De[:, 0, c]
                ot = s2i[Dt[:, s2i, c].argmax(1)]; oe = np.argsort(-De[:, s2i, c], 1)
                r = dict(数据=tag, 目标=y.replace("Y_", ""), 候选=cand, 实现=v,
                         V误差_3=float(np.abs(Ve[:, c] - V3[:, c]).mean()), V误差_4=float(np.abs(e4[:, c] - V4[:, c]).mean()),
                         V偏差_3=float((Ve[:, c] - V3[:, c]).mean()),
                         边际误差=float(np.abs(De[:, bsz <= 2, c] - Dt[:, bsz <= 2, c]).mean()),
                         M2误差=float(np.abs(Me - Mt).mean()), M2偏差=float((Me - Mt).mean()), Γ2误差=float(np.abs(Ge - Gt).mean()),
                         M2排序=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")),
                         认证前1=float(L1.sum() / Mt.sum()), 认证前3=float(L3.sum() / Mt.sum()),
                         背景召回前1=float(np.mean([ot[i] in s2i[oe[i, :1]] for i in F])),
                         背景召回前3=float(np.mean([ot[i] in s2i[oe[i, :3]] for i in F])),
                         背景召回前5=float(np.mean([ot[i] in s2i[oe[i, :5]] for i in F])))
                for tau in (0.5, 0.7):
                    for dl in (0.0, 0.05):
                        rc, pr = crit_cert(V3, Ve, k3, idx3, BK, F, c, tau, dl)
                        r[f"召回_τ{tau}_δ{dl}"] = rc; r[f"精确_τ{tau}_δ{dl}"] = pr
                rows.append(r)
    print("完成", tag, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(A, "candidates_by_target.csv"), index=False)
# 先在"实现"层面对目标平均，再对 C2 的三个实现平均
cols = [c for c in R.columns if c not in ("数据", "目标", "候选", "实现")]
per_impl = R.groupby(["候选", "实现"])[cols].mean(); M = per_impl.groupby("候选").mean().reindex([c for c in CAND if c in set(R.候选)])
M.to_csv(os.path.join(A, "candidates_mean.csv")); pd.set_option("display.width", 250)
print(M.T.round(4).to_string())
# 选择规则
main = [("V误差_3", -1), ("V误差_4", -1), ("M2误差", -1), ("Γ2误差", -1), ("M2排序", 1), ("认证前1", 1), ("认证前3", 1), ("背景召回前3", 1),
        ("召回_τ0.5_δ0.0", 1), ("召回_τ0.7_δ0.0", 1)]
tol = {"V误差_3": .002, "V误差_4": .002, "M2误差": .002, "Γ2误差": .002}
cs = [c for c in ["C1", "C2", "C3"] if c in M.index]; score = {}
for c in cs:
    ok = 0
    for m, sgn in main:
        t = tol.get(m, .005); best = max(sgn * M.loc[o, m] for o in cs)
        ok += int(sgn * M.loc[c, m] >= best - t)
    score[c] = ok
print("不劣于其余候选的指标数：", score)
if "E_old" in M.index:
    for c in cs:
        worse = [m for m, sgn in main if sgn * M.loc[c, m] < sgn * M.loc["E_old", m] - tol.get(m, .005)]
        print(c, "劣于 E_old 的指标：", worse or "无")
json.dump(dict(score=score), open(os.path.join(A, "selection.json"), "w"), ensure_ascii=False, indent=1)
