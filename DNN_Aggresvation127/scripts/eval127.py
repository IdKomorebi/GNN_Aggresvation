# -*- coding: utf-8 -*-
"""127 号 步骤 4：汇总评测（口径与 125 号 select125 相同）。
全表（规模 ≤3 全部集合，闭包后）：V 误差、M2 误差、Γ2 误差、M2 排序、认证前1/前3、背景召回前3、关键召回（τ=0.5/0.7，δ=0，扫描—认证）。
样本（每组同一批 200 个集合，未闭包，原始估计截断到 [0,1]）：V 误差——所有方法（含只在样本上运行的 LazyVI）都报告。
来源：本文 = 125 号 final_est；代理模型 = 118 号 head3；逐集合线性 / [x,x²] 回归 = 118 号 lin / poly；
      Dropout、热启动 = 本号 outputs/est；TabPFN = 本号 outputs/tabpfn 分片合并；LazyVI = 本号 lazy_s200。"""
import os, sys, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
A = os.path.join(ROOT, "outputs", "analysis"); os.makedirs(A, exist_ok=True)


def crit_cert(V3, Ve, k3, idx3, BK, F, c, tau):
    mus = pipe.mus_list(V3, k3, F, c, tau, 3); crit = np.array([any(i in m for m in mus) for i in F])
    mus_e = pipe.mus_list(Ve, k3, F, c, tau, 3); crit_e = np.array([any(i in m for m in mus_e) for i in F]); cert = np.zeros(len(F), bool)
    for i in np.where(crit_e)[0]:
        cands = []
        for T in BK[i]:
            if len(T) > 2:
                continue
            vT = Ve[idx3[T], c] if T else 0.0; vTi = Ve[idx3[tuple(sorted(T + (i,)))], c]
            if vT <= tau < vTi:
                cands.append((vTi, T))
        for _, T in sorted(cands, reverse=True)[:3]:
            vT = V3[idx3[T], c] if T else 0.0
            if vT <= tau < V3[idx3[tuple(sorted(T + (i,)))], c]:
                cert[i] = True
    return float((cert & crit).sum() / crit.sum()) if crit.sum() else np.nan


def load_full(tag, T, n3):
    """返回 {方法: (n3,C) 估计}（只含全表方法）。"""
    out = {}
    f = os.path.join(REPO, "DNN_Aggresvation125", "outputs", "final_est", f"{T}.npz"); out["ours"] = np.load(f)["E"]
    for k, v in [("surrogate", "head3"), ("lin", "lin"), ("poly", "poly")]:
        f = os.path.join(REPO, "DNN_Aggresvation118", "outputs", "est", T, f"{v}.npz")
        if os.path.exists(f):
            out[k] = np.load(f)["est"]
    for k in ("dropout", "ws"):
        f = os.path.join(ROOT, "outputs", "est", T, f"{k}.npz")
        if os.path.exists(f):
            out[k] = np.load(f)["est"]
    sh = sorted(glob.glob(os.path.join(ROOT, "outputs", "tabpfn", f"{T}_shard*.npz")))
    if len(sh) == 6:
        e = None
        for s in sh:
            z = np.load(s); e = np.zeros((n3, z["est"].shape[1]), np.float32) if e is None else e; e[z["idx"]] = z["est"]
        out["tabpfn"] = e
    return out


rows, samp = [], []
for tag, rel, _, _ in registry.DATASETS:
    T = tag.replace("/", "_"); d = registry.load_ds(rel); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"]); F = list(range(p))
    k3 = [k for k in keys if len(k) <= 3]; V3 = d["V"][np.array([len(k) <= 3 for k in keys])]; idx3 = {k: r for r, k in enumerate(k3)}; n3 = len(k3)
    _, _, Dt, BK, bsz = pipe.m_table(V3, k3, F, 2); s2i = np.where(bsz <= 2)[0]
    sidx = np.random.RandomState(2).choice(n3, min(200, n3), replace=False)
    ests = load_full(tag, T, n3)
    for meth, est in ests.items():
        assert est.shape[0] == n3, (tag, meth, est.shape)
        Ve = pipe.closure_max(np.clip(est, 0, 1), k3); De = pipe.m_table(Ve, k3, F, 2)[2]
        for c, y in enumerate(spec["targ"]):
            Mt, Me, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2)
            ot = s2i[Dt[:, s2i, c].argmax(1)]; oe = np.argsort(-De[:, s2i, c], 1)
            rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), 方法=meth, V误差=float(np.abs(Ve[:, c] - V3[:, c]).mean()),
                             V偏差=float((Ve[:, c] - V3[:, c]).mean()), M2误差=float(np.abs(Me - Mt).mean()), M2偏差=float((Me - Mt).mean()),
                             Γ2误差=float(np.abs((Me - De[:, 0, c]) - (Mt - Dt[:, 0, c])).mean()),
                             M2排序=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")),
                             认证前1=float(L1.sum() / Mt.sum()), 认证前3=float(L3.sum() / Mt.sum()),
                             背景召回前3=float(np.mean([ot[i] in s2i[oe[i, :3]] for i in F])),
                             关键召回05=crit_cert(V3, Ve, k3, idx3, BK, F, c, 0.5), 关键召回07=crit_cert(V3, Ve, k3, idx3, BK, F, c, 0.7)))
            samp.append(dict(数据=tag, 目标=y.replace("Y_", ""), 方法=meth, 样本V误差=float(np.abs(np.clip(est[sidx, c], 0, 1) - V3[sidx, c]).mean())))
    f = os.path.join(ROOT, "outputs", "est", T, "lazy_s200.npz")
    if os.path.exists(f):
        z = np.load(f); assert (z["idx"] == sidx).all()
        for c, y in enumerate(spec["targ"]):
            samp.append(dict(数据=tag, 目标=y.replace("Y_", ""), 方法="lazyvi", 样本V误差=float(np.abs(np.clip(z["est"][:, c], 0, 1) - V3[sidx, c]).mean())))
    print("完成", tag, sorted(ests), flush=True)
R = pd.DataFrame(rows); S = pd.DataFrame(samp); R.to_csv(os.path.join(A, "full_by_target.csv"), index=False); S.to_csv(os.path.join(A, "sample_by_target.csv"), index=False)
M = R.groupby("方法").mean(numeric_only=True).join(R.groupby("方法").size().rename("目标数"))
M = M.join(S.groupby("方法").样本V误差.mean(), how="outer"); M.to_csv(os.path.join(A, "summary.csv"))
pd.set_option("display.width", 250); print(M.round(4).to_string())
