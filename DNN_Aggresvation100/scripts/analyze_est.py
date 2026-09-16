# -*- coding: utf-8 -*-
"""100 号估计器侧分析（RQ1 边际保真 + RQ3 见证与上下界），主估计器 L0ensx，另报 L0 / direct / L1x(PJM)。

真值：DNN 专用重训（val 选轮次），规模≤4，经验攻击能力的单调闭包（val 选子集）。
估计：通用模型测试 R²，单调闭包取子集最大（估计器没有独立 val）；同时报告不闭包版本的价值误差。
E1 价值误差（按规模）、危险漏判
E2 边际误差：全部 (i, T, c)，|T|≤3：MAE、|e| 95/99% 分位、最大；按真实边际强度分层；低估(−e)分位
E3 M̃^(K) 误差、Kendall、档位一致；见证：同表认证下界 L=Δ(T̃)、L/M、召回@0.02、见证完全一致
E4 上界 U = max_T [Δ̃(T) + q(层(Δ̃(T)))]：q 为"按 Δ̃ 分层的低估量 1−α 分位"，留一字段交叉校准；
   对照：不分层的全局 q。报告覆盖率 P(M≤U) 与紧度 U−M
"""
import sys, glob, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of

A = ROOT / "outputs/analysis"; GR = [0.05, 0.2, 0.5]; rep = ["# 100 号估计器侧分析（自动生成）\n"]


def md(df): return df.to_markdown(floatfmt=".4f")


def load_npz_shards(pattern, n, key):
    fs = sorted(glob.glob(pattern))
    if not fs or len(fs) != int(fs[0].split("of")[-1].split(".")[0]): return None
    out = np.zeros((n, 12), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


def marg_tables(Vbar, keys, act, kmax=3):
    """返回 D (n_act, nT, C) 与背景键 bkeys[p] (list)。"""
    idx = index_of(keys); C = Vbar.shape[1]; Ds, bk = [], []
    for i in act:
        others = [a for a in act if a != i]
        Ts = [T for s in range(kmax + 1) for T in combinations(others, s)]
        rT = np.array([idx[T] if T else -1 for T in Ts]); rTi = np.array([idx[tuple(sorted(T + (i,)))] for T in Ts])
        vT = np.where(rT[:, None] >= 0, Vbar[np.maximum(rT, 0)], 0.0)
        Ds.append(Vbar[rTi] - vT); bk.append(Ts)
    return np.stack(Ds), bk


def run(ds):
    meta = pickle.load(open(ROOT / f"outputs/sets/{ds}_meta.pkl", "rb")); keys, act = meta["keys"], meta["active"]; n = len(keys)
    fs = sorted(glob.glob(str(ROOT / f"outputs/truth/{ds}_k4_seed0_s*of3.npz")))
    if len(fs) != 3: rep.append(f"## {ds} 真值未完成"); return
    Tc = np.zeros((n, 12), np.float32); Tv = np.zeros((n, 12), np.float32)
    for f in fs:
        z = np.load(f); Tc[z["idx"]] = z["clean"]; Tv[z["idx"]] = z["val_r2"]
    Vbar = closure(np.clip(Tc, -1, 1), keys, Vval=Tv)
    Dt, bk = marg_tables(Vbar, keys, act); nT = Dt.shape[1]
    sz = np.array([len(k) for k in keys]); bsz = np.array([len(T) for T in bk[0]])
    Mt = np.stack([Dt[:, bsz <= K].max(1) for K in range(4)], 1)                    # (n_act,4,C)
    At = np.stack([Dt[:, bsz <= K].argmax(1) for K in range(4)], 1)
    v_rows, m_rows, w_rows, u_rows, strat = [], [], [], [], []
    for est in ["L0ensx", "L0", "L1x", "direct"]:
        E = load_npz_shards(str(ROOT / f"outputs/est/{ds}_k4_{est}_s*of*.npz"), n, "v")
        if E is None: continue
        Eb = closure(np.clip(E, 0, 1), keys)
        r = dict(est=est)
        for s in [1, 2, 3, 4]:
            r[f"价值MAE_规模{s}"] = np.abs(Eb[sz == s] - Vbar[sz == s]).mean()
        r["价值MAE_不闭包"] = np.abs(np.clip(E, 0, 1) - np.clip(Tc, 0, 1)).mean(); r["价值偏差"] = (Eb - Vbar).mean()
        hi = Vbar > 0.7; r["危险漏判@0.7"] = ((Eb <= 0.7) & hi).sum() / hi.sum()
        v_rows.append(r)
        De, _ = marg_tables(Eb, keys, act); e = De - Dt; ae = np.abs(e)
        mr = dict(est=est, 边际MAE=ae.mean(), q95=np.quantile(ae, 0.95), q99=np.quantile(ae, 0.99), 最大=ae.max(),
                  低估q95=np.quantile(-e, 0.95), 低估q99=np.quantile(-e, 0.99), 边际相关=np.corrcoef(De.ravel(), Dt.ravel())[0, 1])
        m_rows.append(mr)
        for lo, hi_ in [(-1, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 2)]:
            k = (Dt > lo) & (Dt <= hi_)
            strat.append(dict(est=est, 真实边际层=f"({lo},{hi_}]", 占比=k.mean(), MAE=ae[k].mean(), 偏差=e[k].mean(), 低估q95=np.quantile(-e[k], 0.95)))
        for K in range(4):
            sel = bsz <= K
            Me = De[:, sel].max(1); Ae = De[:, sel].argmax(1)
            L = np.take_along_axis(Dt[:, sel], Ae[:, None, :], 1)[:, 0]; M = Mt[:, K]
            same = (Ae == At[:, K]).mean()
            w_rows.append(dict(est=est, K=K, 真M均值=M.mean(), 估计M均值=Me.mean(), 估计偏差=(Me - M).mean(), 估计MAE=np.abs(Me - M).mean(),
                               Kendall=np.nanmean([kendalltau(Me[:, c], M[:, c])[0] for c in range(12)]),
                               档位一致=(np.digitize(Me, GR) == np.digitize(M, GR)).mean(),
                               认证下界L均值=L.mean(), L除以M=L.sum() / M.sum(), 召回_L大于M减002=(L >= M - 0.02).mean(), 见证完全一致=same))
            # 上界（仅主估计器与 L0 做，K=1,2,3）
            if K >= 1 and est in ("L0ensx", "L0"):
                Des, Dts = De[:, sel], Dt[:, sel]; es = Des - Dts
                edges = np.quantile(Des, np.linspace(0, 1, 11)); lay = np.clip(np.searchsorted(edges, Des, side="right") - 1, 0, 9)
                for alpha in [0.05, 0.01]:
                    cover, tight, cover_g, tight_g = [], [], [], []
                    for p in range(len(act)):
                        other = np.ones(len(act), bool); other[p] = False
                        neg = -es[other]; lo_ = lay[other]
                        q = np.array([np.quantile(neg[lo_ == l], 1 - alpha) if (lo_ == l).any() else 0 for l in range(10)])
                        qg = np.quantile(neg, 1 - alpha)
                        U = (Des[p] + np.maximum(q[lay[p]], 0)).max(0); Ug = Des[p].max(0) + max(qg, 0)
                        cover.append(M[p] <= U + 1e-9); tight.append(U - M[p]); cover_g.append(M[p] <= Ug + 1e-9); tight_g.append(Ug - M[p])
                    u_rows.append(dict(est=est, K=K, alpha=alpha, 分层上界覆盖率=np.mean(cover), 分层上界紧度=np.mean(tight),
                                       全局q上界覆盖率=np.mean(cover_g), 全局q上界紧度=np.mean(tight_g)))
    for title, rows in [("E1 价值误差", v_rows), ("E2 边际误差（全部 |T|≤3 背景）", m_rows), ("E2 按真实边际强度分层", strat),
                        ("E3 M̃ 与见证（同表认证）", w_rows), ("E4 上界覆盖率与紧度（留一字段交叉校准）", u_rows)]:
        if rows:
            df = pd.DataFrame(rows); df.to_csv(A / f"{ds}_{title.split()[0]}_{len(rows)}.csv", index=False)
            rep.append(f"## {ds} {title}\n\n" + df.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["pjm", "caiso"]:
        run(ds)
    (A / "report_est.md").write_text("\n\n".join(rep), encoding="utf-8"); print("\n\n".join(rep))
