# -*- coding: utf-8 -*-
"""108-A1：把 RQ-A4 从「点估计 + recall」改成「[L, U] 区间」口径。

三件事（全部在 103 号正式真值 + 已有估计表上重算，不重训、不占 GPU）：
  L：认证下界。对照 top-1 见证（104 号现有 cert_L）与 **top-3 见证取 max**。
     top-1/2/3 的背景 T 满足 |T|≤K≤2，其 V(T) 与 V(T∪{i}) 都在 k3 真值表内，
     所以「重训认证」在正式真值表上已经完成，无需再训。
  U：上界。沿用 100 号 E4 构造 U = max_T [Δ̃(T) + q_α(层(Δ̃(T)))]，
     q 为按 Δ̃ 分层的低估量 1−α 分位，**留一字段交叉校准**（当前字段的 q 由其余 40 个字段估）。
     本号改用 103 号正式三攻击器真值重新校准。
  区间下的 τ-critical：用 V 的校准分位 δ_α 做保守判定
     ∃T:(ê(T) − δ ≤ τ) ∧ (ê(T∪i) + δ > τ)，换取 recall。
"""
import sys, glob, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100, R102, R103, R104 = (REPO / f"DNN_Aggresvation{n}" for n in (100, 102, 103, 104))
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of
from analyze_est import marg_tables
from runlog import log
A = ROOT / "outputs/analysis"; A.mkdir(parents=True, exist_ok=True)
TAUS = [0.5, 0.7, 0.9]; ALPHAS = [0.05, 0.01]


def shards(pat, n, key, ncol=12):
    fs = sorted(glob.glob(pat))
    if not fs or len(fs) != int(fs[0].split("of")[-1].split(".")[0]): return None
    out = np.zeros((n, ncol), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


def loo_q(err, lay, act_n, p, nlay, alpha):
    """留一字段：用除 p 外所有字段的低估量，按层给 1−α 分位。err/lay 形状 (n_act, nT, C)。"""
    other = np.ones(act_n, bool); other[p] = False
    neg = -err[other].ravel(); lo = lay[other].ravel()
    return np.array([np.quantile(neg[lo == l], 1 - alpha) if (lo == l).any() else 0.0 for l in range(nlay)])


int_rows, cov_rows, crit_rows, wit_rows = [], [], [], []
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb"))
    keys, act, gen, conf = meta["keys_k3"], meta["active"], meta["general"], meta["conf"]; n = len(keys)
    Vt = np.load(R103 / f"outputs/analysis/{ds}_truth_official.npz")["V"]          # 正式真值（已闭包）
    Ee = shards(str(R103 / f"outputs/est/{ds}_k3_L0ensx_s*of3.npz"), n, "v")
    Eb = closure(np.clip(Ee, 0, 1), keys)
    Dt, bk = marg_tables(Vt, keys, act, kmax=2); bsz = np.array([len(T) for T in bk[0]])
    De = marg_tables(Eb, keys, act, kmax=2)[0]
    idx = index_of(keys); na = len(act)
    # V 的低估分位（用于保守 critical）：留一字段，全局（不分层）
    Verr = np.zeros((na, n, 12), np.float32)     # 仅用于按字段留一，故按"含该字段的集合"归属
    belong = np.zeros((na, n), bool)
    for p, i in enumerate(act):
        for r, k in enumerate(keys):
            if i in k: belong[p, r] = True
    dV = (Eb - Vt)
    for K in [0, 1, 2]:
        sel = bsz <= K
        Des, Dts = De[:, sel], Dt[:, sel]
        err = Des - Dts
        edges = np.quantile(Des, np.linspace(0, 1, 11)); lay = np.clip(np.searchsorted(edges, Des, "right") - 1, 0, 9)
        Mt = Dts.max(1); je = Des.argmax(1)                                        # (na,12)
        Me = np.take_along_axis(Des, je[:, None], 1)[:, 0]
        L1 = np.take_along_axis(Dts, je[:, None], 1)[:, 0]                          # top-1 认证下界
        top3 = np.argsort(-Des, 1)[:, :3, :]                                       # (na,3,12)
        L3 = np.take_along_axis(Dts, top3, 1).max(1)                               # top-3 取 max
        for alpha in ALPHAS:
            U = np.zeros_like(Mt)
            for p in range(na):
                q = loo_q(err, lay, na, p, 10, alpha)
                U[p] = (Des[p] + np.maximum(q[lay[p]], 0)).max(0)
            cov_rows.append(dict(数据集=ds, K=K, alpha=alpha,
                                 覆盖率_M小于等于U=float((Mt <= U + 1e-9).mean()),
                                 覆盖率_区间含M=float(((L3 <= Mt + 1e-9) & (Mt <= U + 1e-9)).mean()),
                                 区间宽度_U减L3=float((U - L3).mean()), 紧度_U减M=float((U - Mt).mean()),
                                 下界差_M减L3=float((Mt - L3).mean())))
            if alpha == 0.05:
                Ukeep = U.copy()
        int_rows.append(dict(数据集=ds, K=K, 条目数=int(Mt.size), exact_M均值=float(Mt.mean()),
                             est_M均值=float(Me.mean()), MAE=float(np.abs(Me - Mt).mean()),
                             认证下界比_top1=float(L1.sum() / Mt.sum()), 认证下界比_top3=float(L3.sum() / Mt.sum()),
                             下界比提升=float((L3 - L1).sum() / Mt.sum()),
                             top3优于top1占比=float((L3 > L1 + 1e-9).mean()),
                             召回_L3大于M减002=float((L3 >= Mt - 0.02).mean()),
                             召回_L1大于M减002=float((L1 >= Mt - 0.02).mean())))
        # 区间口径下的 τ-critical
        TsK_all = {}
        for p, i in enumerate(act):
            others = [x for x in act if x != i]
            TsK_all[p] = [T for s in range(K + 1) for T in combinations(others, s)]
        # V 的保守边界：留一字段的全局低估/高估分位
        for alpha in ALPHAS:
            for tau in TAUS:
                tp = fp = fn = tp0 = fn0 = 0
                for p, i in enumerate(act):
                    other = np.ones(na, bool); other[p] = False
                    mask = belong[other].any(0)
                    d = dV[mask].ravel()
                    delta = float(np.quantile(np.abs(d), 1 - alpha))
                    TsK = TsK_all[p]
                    rT = np.array([idx[T] if T else -1 for T in TsK])
                    rTi = np.array([idx[tuple(sorted(T + (i,)))] for T in TsK])
                    vT = np.where(rT[:, None] >= 0, Vt[np.maximum(rT, 0)], 0.0); vTi = Vt[rTi]
                    eT = np.where(rT[:, None] >= 0, Eb[np.maximum(rT, 0)], 0.0); eTi = Eb[rTi]
                    truth = ((vT <= tau) & (vTi > tau)).any(0)
                    point = ((eT <= tau) & (eTi > tau)).any(0)
                    cons = ((eT - delta <= tau) & (eTi + delta > tau)).any(0)
                    tp += int((truth & cons).sum()); fp += int((~truth & cons).sum()); fn += int((truth & ~cons).sum())
                    tp0 += int((truth & point).sum()); fn0 += int((truth & ~point).sum())
                crit_rows.append(dict(数据集=ds, K=K, alpha=alpha, tau=tau, 真critical数=tp + fn,
                                      点估计recall=tp0 / max(tp0 + fn0, 1),
                                      区间保守recall=tp / max(tp + fn, 1),
                                      区间保守precision=tp / max(tp + fp, 1)))
        log("INTERVAL", "DONE", f"{ds} K={K} 区间口径完成")

pd.DataFrame(int_rows).to_csv(A / "108_A1_bounds.csv", index=False)
pd.DataFrame(cov_rows).to_csv(A / "108_A1_coverage.csv", index=False)
pd.DataFrame(crit_rows).to_csv(A / "108_A1_critical_interval.csv", index=False)
print(pd.DataFrame(int_rows).to_markdown(index=False, floatfmt=".4f"))
print()
print(pd.DataFrame(cov_rows).to_markdown(index=False, floatfmt=".4f"))
print()
print(pd.DataFrame(crit_rows).to_markdown(index=False, floatfmt=".4f"))
