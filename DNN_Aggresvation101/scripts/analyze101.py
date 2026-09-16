# -*- coding: utf-8 -*-
"""101 号：独立认证（discovery / audit 完全不相交），避免 max 操作带来的 winner's curse。

对每个数据集、rep∈{0,1,2}、方向 h∈{0,1}：discovery = 半数据 H_h，audit = H_{1−h}。
  discovery：半数据上训练的 uniform 三种子 L0ensx 估计器 → 单调闭包 → M̃^(K)、见证 T̃；以及 discovery 专用重训真值 M_disc。
  audit：另一半数据上专用重训真值（不参与训练、选参、K 选择、见证搜索）→ M_aud、L=Δ_aud(T̃)。
指标（K=0/1/2）：
  C1 winner's curse：M̃_disc − M_aud；同数据认证 Δ_disc(T̃) 与独立认证 Δ_aud(T̃) 之差；真值在 disc 上取 max 的选择偏差 M_disc − Δ_aud(T*_disc)
  C2 独立认证下界：L/M_aud、召回(L ≥ M_aud − 0.02)、见证与 audit 真见证一致率
  C3 阈值跨越：discovery 估计判定 τ-关键的 (字段,目标,背景) 在 audit 真值上仍越过 τ 的比例；audit 真关键字段被发现的召回
  C4 强交互：discovery 估计 I_ij≥0.05 的对在 audit 上 I_ij>0 / >0.02 的比例；audit 真值 I≥0.05 的对被召回（估计 I≥0.02）
  C5 排序与档位：Kendall(M̃_disc, M_aud)、档位一致
  C6 跨数据上界：U = max_T[Δ̃(T)+q(层)]，q 在 discovery 上（估计 vs discovery 真值）分层校准，检查 audit 上 M_aud ≤ U
"""
import sys, glob, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, t as tdist
ROOT = Path(__file__).resolve().parents[1]; R100 = ROOT.parent / "DNN_Aggresvation100"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts"))
from mkfull import closure, index_of
from analyze_est import marg_tables

O = ROOT / "outputs"; GR = [0.05, 0.2, 0.5]; TAUS = [0.5, 0.7]


def shards(pattern, n, key):
    fs = sorted(glob.glob(pattern))
    if len(fs) != 3: return None
    out = np.zeros((n, 12), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


def truth_bar(ds, tag, keys):
    n = len(keys); c = shards(str(O / f"truth/{ds}_k3_{tag}_seed0_s*of3.npz"), n, "clean"); v = shards(str(O / f"truth/{ds}_k3_{tag}_seed0_s*of3.npz"), n, "val_r2")
    return None if c is None else closure(np.clip(c, -1, 1), keys, Vval=v)


def pair_I(Vbar, keys, act):
    idx = index_of(keys); out = []
    for i, j in combinations(act, 2):
        out.append(Vbar[idx[(min(i, j), max(i, j))]] - Vbar[idx[(i,)]] - Vbar[idx[(j,)]])
    return np.stack(out)                                                     # (npairs, C)


def one(ds, rep, h, keys, act):
    td = f"rep{rep}h{h}"; ta = f"rep{rep}h{1-h}"; n = len(keys)
    E = shards(str(O / f"est/{ds}_k3_{td}_L0ensx_s*of3.npz"), n, "v")
    Vd, Va = truth_bar(ds, td, keys), truth_bar(ds, ta, keys)
    if E is None or Vd is None or Va is None: return []
    Eb = closure(np.clip(E, 0, 1), keys)
    De, bk = marg_tables(Eb, keys, act, kmax=2); Dd, _ = marg_tables(Vd, keys, act, kmax=2); Da, _ = marg_tables(Va, keys, act, kmax=2)
    bsz = np.array([len(T) for T in bk[0]]); rows = []
    Ie, Ia = pair_I(Eb, keys, act), pair_I(Va, keys, act)
    for K in [0, 1, 2]:
        sel = bsz <= K
        Me, Ae = De[:, sel].max(1), De[:, sel].argmax(1)
        Md, Ad = Dd[:, sel].max(1), Dd[:, sel].argmax(1)
        Ma, Aa = Da[:, sel].max(1), Da[:, sel].argmax(1)
        L = np.take_along_axis(Da[:, sel], Ae[:, None, :], 1)[:, 0]            # 独立认证
        Ls = np.take_along_axis(Dd[:, sel], Ae[:, None, :], 1)[:, 0]           # 同数据认证
        Ld = np.take_along_axis(Da[:, sel], Ad[:, None, :], 1)[:, 0]           # disc 真见证在 audit 上
        r = dict(ds=ds, rep=rep, h=h, K=K, M_audit均值=Ma.mean(), 估计M_disc均值=Me.mean(),
                 C1_估计减audit=(Me - Ma).mean(), C1_同数据认证减独立认证=(Ls - L).mean(), C1_disc真值max减其在audit上=(Md - Ld).mean(),
                 C2_L除以M_audit=L.sum() / Ma.sum(), C2_同数据L除以M_disc=Ls.sum() / Md.sum(), C2_同数据召回=(Ls >= Md - 0.02).mean(), C2_召回=(L >= Ma - 0.02).mean(), C2_见证一致=(Ae == Aa).mean(),
                 C5_Kendall=np.nanmean([kendalltau(Me[:, c], Ma[:, c])[0] for c in range(12)]),
                 C5_档位一致=(np.digitize(Me, GR) == np.digitize(Ma, GR)).mean())
        # C3 阈值跨越（背景规模 ≤K）
        idx = index_of(keys)
        for tau in TAUS:
            conf_hit = tot = found = true_n = 0
            for p, i in enumerate(act):
                others = [a for a in act if a != i]
                Ts = [T for s in range(K + 1) for T in combinations(others, s)]
                rT = np.array([idx[T] if T else -1 for T in Ts]); rTi = np.array([idx[tuple(sorted(T + (i,)))] for T in Ts])
                eT = np.where(rT[:, None] >= 0, Eb[np.maximum(rT, 0)], 0); eTi = Eb[rTi]
                aT = np.where(rT[:, None] >= 0, Va[np.maximum(rT, 0)], 0); aTi = Va[rTi]
                ce = (eT <= tau) & (eTi > tau); ca = (aT <= tau) & (aTi > tau)
                conf_hit += (ce & ca).sum(); tot += ce.sum()
                found += (ce.any(0) & ca.any(0)).sum(); true_n += ca.any(0).sum()
            r[f"C3_τ{tau}_跨越背景复现率"] = conf_hit / max(tot, 1); r[f"C3_τ{tau}_关键字段召回"] = found / max(true_n, 1)
        rows.append(r)
    # C4 强交互（与 K 无关，记在 K=1 行）
    strong_e = Ie >= 0.05; strong_a = Ia >= 0.05
    rows[1].update(C4_估计强交互数=int(strong_e.sum()), C4_audit上I大于0=float((Ia[strong_e] > 0).mean()) if strong_e.any() else np.nan,
                   C4_audit上I大于002=float((Ia[strong_e] > 0.02).mean()) if strong_e.any() else np.nan,
                   C4_audit强交互召回=float((Ie[strong_a] >= 0.02).mean()) if strong_a.any() else np.nan)
    # C6 跨数据上界（K=1,2）
    for K in [1, 2]:
        sel = bsz <= K; Des, es = De[:, sel], De[:, sel] - Dd[:, sel]
        edges = np.quantile(Des, np.linspace(0, 1, 11)); lay = np.clip(np.searchsorted(edges, Des, side="right") - 1, 0, 9)
        Ma = Da[:, sel].max(1)
        for alpha in [0.05, 0.01]:
            q = np.array([np.quantile(-es[lay == l], 1 - alpha) if (lay == l).any() else 0 for l in range(10)])
            U = (Des + np.maximum(q[lay], 0)).max(1)
            rows[K][f"C6_α{alpha}_audit覆盖率"] = float((Ma <= U + 1e-9).mean()); rows[K][f"C6_α{alpha}_紧度"] = float((U - Ma).mean())
    return rows


if __name__ == "__main__":
    allrows = []
    for ds in ["pjm", "caiso"]:
        meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys, act = meta["keys_k3"], meta["active"]
        for rep in range(3):
            for h in range(2):
                allrows += one(ds, rep, h, keys, act)
    df = pd.DataFrame(allrows); df.to_csv(O / "analysis/certify_all.csv", index=False)
    num = df.drop(columns=["rep", "h"]).groupby(["ds", "K"])
    mean, std, cnt = num.mean(), num.std(), num.count()
    ci = std / np.sqrt(cnt) * tdist.ppf(0.975, np.maximum(cnt - 1, 1))
    summ = mean.round(4).astype(str) + " ± " + ci.round(4).astype(str)
    summ.T.to_csv(O / "analysis/certify_summary.csv")
    txt = "# 101 号独立认证汇总（均值 ± 95% t 区间，3 次随机对半 × 2 方向）\n\n" + summ.T.to_markdown()
    (O / "analysis/report101.md").write_text(txt, encoding="utf-8"); print(txt)
