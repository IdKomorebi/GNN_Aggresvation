# -*- coding: utf-8 -*-
"""109 号分析：攻击者辅助标签量 n_aux ∈ {10%,25%,50%,100%} 下的 V / M^(K) / τ-critical / 发布审查。

两个方向：
  【方向 A：攻击者能力】M^(K)(n_aux) 随样本量如何变化，排序与档位是否稳定。
  【方向 B：审计方低估攻击者的后果】审计方按 n_aux 档评估并放行，而真实攻击者拥有 full 数据时，
    有多少危险集合被放行。这直接接 106 号的发布审查。
口径：单一攻击器（多目标 DNN），比例之间可比；与 103 号三攻击器正式真值口径不同，结论只在本号内部横比。
"""
import sys, glob, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100 = REPO / "DNN_Aggresvation100"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of
from analyze_est import marg_tables
from runlog import log
A = ROOT / "outputs/analysis"; A.mkdir(parents=True, exist_ok=True)
FRACS = [10, 25, 50, 100]; TAUS = [0.5, 0.7, 0.9]; GR = [0.05, 0.2, 0.5]


def load_truth(ds, f):
    fs = sorted(glob.glob(str(ROOT / f"outputs/truth/{ds}_f{f:03d}_seed0_s*of3.npz")))
    if len(fs) != 3: return None, None
    n = None; out = {}
    zs = [np.load(x) for x in fs]
    n = max(int(z["idx"].max()) for z in zs) + 1
    for k in ["clean", "val_r2"]:
        a = np.zeros((n, 12), np.float32)
        for z in zs: a[z["idx"]] = z[k]
        out[k] = a
    return out, int(zs[0]["n_aux"])


vrows, mrows, crows, arows = [], [], [], []
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb"))
    keys, act, gen, conf = meta["keys_k3"], meta["active"], meta["general"], meta["conf"]
    sz = np.array([len(k) for k in keys]); idx = index_of(keys)
    Vb, Ms, naux = {}, {}, {}
    for f in FRACS:
        T, na = load_truth(ds, f)
        if T is None: print(f"[warn] {ds} f{f} 未完成"); continue
        Vb[f] = closure(np.clip(T["clean"], 0, 1), keys, Vval=T["val_r2"])
        D, bk = marg_tables(Vb[f], keys, act, kmax=2); bsz = np.array([len(t) for t in bk[0]])
        Ms[f] = {K: D[:, bsz <= K].max(1) for K in [0, 1, 2]}; naux[f] = na
    if 100 not in Vb: continue
    full = Vb[100]
    # ---- A1：V 按规模
    for f in Vb:
        for s in [1, 2, 3]:
            k = sz == s
            vrows.append(dict(数据集=ds, 比例=f"{f}%", n_aux=naux[f], 规模=s,
                              V均值=float(Vb[f][k].mean()), V相对full=float(Vb[f][k].mean() / full[k].mean()),
                              V减full均值=float((Vb[f][k] - full[k]).mean())))
    # ---- A2：M^(K)
    for f in Ms:
        for K in [0, 1, 2]:
            m, mf = Ms[f][K], Ms[100][K]
            mrows.append(dict(数据集=ds, 比例=f"{f}%", n_aux=naux[f], K=K, M均值=float(m.mean()),
                              M相对full=float(m.mean() / mf.mean()),
                              Kendall对full=float(np.nanmean([kendalltau(m[:, c], mf[:, c])[0] for c in range(12)])),
                              档位一致对full=float((np.digitize(m, GR) == np.digitize(mf, GR)).mean())))
    # ---- B1：τ-critical（以 full 为真）
    for f in Ms:
        for K in [0, 1, 2]:
            for tau in TAUS:
                tp = fp = fn = 0
                for p, i in enumerate(act):
                    others = [x for x in act if x != i]
                    TsK = [T for s in range(K + 1) for T in combinations(others, s)]
                    rT = np.array([idx[T] if T else -1 for T in TsK])
                    rTi = np.array([idx[tuple(sorted(T + (i,)))] for T in TsK])
                    def crit(V):
                        vT = np.where(rT[:, None] >= 0, V[np.maximum(rT, 0)], 0.0)
                        return ((vT <= tau) & (V[rTi] > tau)).any(0)
                    t_, e_ = crit(full), crit(Vb[f])
                    tp += int((t_ & e_).sum()); fp += int((~t_ & e_).sum()); fn += int((t_ & ~e_).sum())
                crows.append(dict(数据集=ds, 比例=f"{f}%", K=K, tau=tau, full_critical数=tp + fn,
                                  recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                                  漏判数=fn))
    # ---- B2：审计方低估攻击者 → 危险放行（全部规模 3 集合）
    k3 = sz == 3
    for f in Vb:
        for tau in TAUS:
            unsafe = full[k3] > tau                                  # 真实攻击者（full）能推断
            release = Vb[f][k3] <= tau                               # 审计方按 n_aux 档放行
            n_uns = int(unsafe.sum())
            arows.append(dict(数据集=ds, 比例=f"{f}%", n_aux=naux[f], tau=tau, 规模3集合数=int(k3.sum()) * 12,
                              不安全数=n_uns, 放行率=float(release.mean()),
                              危险放行率=float((unsafe & release).sum() / max(n_uns, 1)),
                              放行集合中不安全占比=float((unsafe & release).sum() / max(int(release.sum()), 1))))
    log("ANALYZE", "DONE", f"{ds} n_aux 分析完成")

for nm, rows in [("109_A_V.csv", vrows), ("109_B_M.csv", mrows), ("109_C_critical.csv", crows), ("109_D_release.csv", arows)]:
    pd.DataFrame(rows).to_csv(A / nm, index=False)
out = ["# 109 号：攻击者辅助标签量 n_aux sensitivity（自动生成；结论见 CHANGELOG.md）\n",
       "> 口径：单一攻击器（多目标 DNN）。比例之间可比；与 103 号三攻击器正式真值不可直接并列。\n"]
for t, nm in [("A. V 随 n_aux（按集合规模）", "109_A_V.csv"), ("B. M^(K) 随 n_aux", "109_B_M.csv"),
              ("C. τ-critical（以 full 为真值）", "109_C_critical.csv"),
              ("D. 审计方按 n_aux 档放行、真实攻击者为 full 时的危险放行", "109_D_release.csv")]:
    out.append(f"## {t}\n\n" + pd.read_csv(A / nm).to_markdown(index=False, floatfmt=".4f") + "\n")
(A / "report109.md").write_text("\n".join(out), encoding="utf-8")
print("分析完成 →", A)
for nm in ["109_B_M.csv", "109_D_release.csv"]:
    print(f"\n=== {nm} ==="); print(pd.read_csv(A / nm).to_markdown(index=False, floatfmt=".4f"))
