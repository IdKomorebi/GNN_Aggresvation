# -*- coding: utf-8 -*-
"""99 号：M^(K) 的数学关系在精确真值博弈上的核验（复用 98 号 gameA/gameB 真值，纯 CPU）。

P1 命题核验（恒等式/不等式是否在真实非单调数据上成立，以及单调包络 v̄ 的作用）
P2 M^(K) 随 K 的饱和与"协同阶" κ
P3 字段分数作为"风险预算"的发布规则：Σ_{i∈A} s_i ≤ τ 的危险漏判/误拒（s = 单字段 / Shapley / M^(K)）
P4 复制稀释：给真值博弈加入精确副本，Shapley 被稀释、M^(K) 不变
P5 Harsanyi 分解：Shapley 与 M^(1) 分别如何吸收正/负红利（风电字段案例）
P6 估计器：用通用模型找最坏背景 → 真值查表得认证下界，对照真 M 与估计 M̂
"""
import sys, json
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; R98 = ROOT.parent / "DNN_Aggresvation98"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(R98 / "scripts"))
from mk import envelope, mobius, mk_all, shapley, minimal_unsafe, extend_with_copies, popcount
from analyze import load_truth, load_est, meta, CONF
from runlog import log

OUT = ROOT / "outputs"; rep = ["# 99 号分析输出（自动生成）\n"]
TAUS = [0.5, 0.7, 0.9]


def md(df): return df.to_markdown(floatfmt=".4f")


def truth(game):
    Tr = load_truth(game); return (Tr[(0, "clean")] + Tr[(1, "clean")]) / 2, Tr


# ---------------------------------------------------------------- P1 命题核验
def p1(game, V):
    p = int(np.log2(len(V))); b = np.arange(2 ** p); sz = popcount(b); C = V.shape[1]
    res = {"game": game}
    Vb = envelope(V)
    res["非单调幅度_max(v̄−v)"] = float((Vb - V).max()); res["非单调集合占比(v̄−v>0.01)"] = float(((Vb - V) > 0.01).mean())
    M, _ = mk_all(V); a = V[1 << np.arange(p)]
    # (a) M^(1) = a + max(0, max_j I_ij)
    I = np.full((p, p, C), -np.inf)
    for i, j in combinations(range(p), 2):
        I[i, j] = I[j, i] = V[(1 << i) | (1 << j)] - V[1 << i] - V[1 << j]
    res["(a) M1=a+max(0,maxI) 最大偏差"] = float(np.abs(a + np.maximum(0, I.max(1)) - M[:, 1]).max())
    # (b) Harsanyi: Δ_i(T)=Σ_{U⊆T} m(U∪i)；(c) φ=Σ m/|U|
    m = mobius(V); phi = shapley(V)
    phi_h = np.stack([(m[b[(b >> i) & 1 == 1]] / sz[b[(b >> i) & 1 == 1]][:, None]).sum(0) for i in range(p)])
    res["(c) φ=Σm/|U| 最大偏差"] = float(np.abs(phi_h - phi).max())
    # (d) 加性包络：|A|≤K+1 ⇒ v(A) ≤ Σ M^(K)
    bits = ((b[:, None] >> np.arange(p)) & 1).astype(float)
    viol = []
    for K in range(p):
        sums = bits @ M[:, K]; ok = sz <= K + 1
        viol.append(float(((V - sums)[ok] > 1e-9).mean()))
    res["(d) 包络在|A|≤K+1上的违例率(全K最大)"] = max(viol)
    # (e) syn(S)=v(S)−max_{T⊊S}v(T) ≤ min_{j∈S} M_j^(|S|−1)
    maxproper = np.full(V.shape, -np.inf)
    for i in range(p):
        has = (b >> i) & 1 == 1; maxproper[has] = np.maximum(maxproper[has], Vb[b[has] ^ (1 << i)])
    syn = V - maxproper; syn[0] = 0
    minM = np.full(V.shape, np.inf)
    for j in range(p):
        has = (b >> j) & 1 == 1
        minM[has] = np.minimum(minM[has], M[j, np.maximum(sz[has] - 1, 0)])
    res["(e) syn≤minM 违例数"] = int(((syn - minM)[sz >= 2] > 1e-9).sum())
    # (f) 关键性 ⇔ 属于某个 ≤K+1 的最小不安全集合；在 v 与 v̄ 上各测
    for lab, W in [("v", V), ("v̄", Vb)]:
        mis = 0; tot = 0
        for tau in TAUS:
            MUS = minimal_unsafe(W, tau)
            for K in [1, 2, 3]:
                for i in range(p):
                    S = b[((b >> i) & 1 == 0) & (sz <= K)]
                    piv = ((W[S] <= tau) & (W[S | (1 << i)] > tau)).any(0)
                    E = b[((b >> i) & 1 == 1) & (sz <= K + 1)]
                    inm = MUS[E].any(0)
                    mis += int((piv != inm).sum()); tot += C
        res[f"(f) 关键性⇔小MUS 不一致率[{lab}]"] = mis / tot
    # (g) Shapley 与 M 的序关系：φ ≤ M^(p−1)；v(i)/p ≤ φ（后者需单调）
    phib = shapley(Vb); Mb, _ = mk_all(Vb); ab = Vb[1 << np.arange(p)]
    res["(g) φ≤M^(p−1) 违例数[v]"] = int((phi - M[:, p - 1] > 1e-9).sum())
    res["(g) a/p≤φ 违例数[v̄]"] = int((ab / p - phib > 1e-9).sum())
    res["(h) M^(K) 对 K 单调违例数"] = int((np.diff(M, axis=1) < -1e-12).sum())
    return res, M


# ---------------------------------------------------------------- P2 饱和 / 协同阶
def p2(game, V, M):
    p = M.shape[0]; players = meta[game]; rows = []
    top = M[:, p - 1]                                   # MCI（不限背景）
    for c in range(len(CONF)):
        for i in range(p):
            gain = top[i, c] - M[i, 0, c]
            kap = int(np.argmax(M[i, :, c] >= top[i, c] - 1e-12))
            k90 = int(np.argmax(M[i, :, c] >= M[i, 0, c] + 0.9 * gain - 1e-12)) if gain > 1e-9 else 0
            k_eps = int(np.argmax(M[i, :, c] >= top[i, c] - 0.01))
            rows.append(dict(game=game, conf=CONF[c], field=players[i], a=M[i, 0, c], M1=M[i, 1, c], M2=M[i, 2, c],
                             M3=M[i, 3, c], Mtop=top[i, c], kappa_exact=kap, kappa_90=k90, kappa_eps=k_eps))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- P3 预算发布规则
def p3(game, V, M):
    p = M.shape[0]; b = np.arange(2 ** p); sz = popcount(b); bits = ((b[:, None] >> np.arange(p)) & 1).astype(float)
    Vb = envelope(V); phi = shapley(V)
    scores = {"单字段 a=M^(0)": M[:, 0], "Shapley φ": phi, "M^(1)": M[:, 1], "M^(2)": M[:, 2],
              "M^(3)": M[:, 3], f"M^({p-1})(MCI)": M[:, p - 1]}
    bands = [("|A|≤2", sz <= 2), ("3–4", (sz >= 3) & (sz <= 4)), ("5–8", (sz >= 5) & (sz <= 8)), (f"9–{p}", sz >= 9)]
    rows = []
    for tau in TAUS:
        unsafe = Vb > tau                                # 安全语义用单调包络
        for nm, s in scores.items():
            for rule in ["预算和 Σs≤τ", "字段阈值 max s≤τ"]:
                est = bits @ s if rule.startswith("预算") else np.stack([np.where(bits[:, [i]] > 0, s[i], -np.inf) for i in range(p)]).max(0)
                ok = est <= tau
                r = dict(game=game, tau=tau, score=nm, rule=rule)
                for bn, k in bands + [("全部", sz >= 1)]:
                    u = unsafe[k]; o = ok[k]
                    r[f"漏判_{bn}"] = (o & u).sum() / max(u.sum(), 1)
                    r[f"误拒_{bn}"] = (~o & ~u).sum() / max((~u).sum(), 1)
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- P4 复制稀释
def p4(V):
    players = meta["gameA"]; rows = []
    for src_name in ["forecast_load_mw_latest_available", "gen_fuel_wind_pct"]:
        src = players.index(src_name)
        for mcopy in range(0, 5):
            W = extend_with_copies(V, src, mcopy) if mcopy else V
            phi = shapley(W); Mw, _ = mk_all(W); p = int(np.log2(len(W)))
            Vb = W[-1]; loo = Vb - W[(2 ** p - 1) ^ (1 << src)]
            for c in ["metered_load_mw", "total_gen"]:
                ci = CONF.index(c)
                rows.append(dict(src=src_name, conf=c, copies=mcopy, shapley=phi[src, ci],
                                 shapley_copies_sum=phi[[src] + list(range(p - mcopy, p)), ci].sum(),
                                 single=Mw[src, 0, ci], M1=Mw[src, 1, ci], M2=Mw[src, 2, ci], loo=loo[ci]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- P5 Harsanyi 案例
def p5(V):
    players = meta["gameA"]; p = len(players); b = np.arange(2 ** p); sz = popcount(b); m = mobius(V); rows = []
    M, arg = mk_all(V)
    for c in ["metered_load_mw", "total_gen", "total_lmp_da"]:
        ci = CONF.index(c)
        for f in ["gen_fuel_wind_mw", "gen_fuel_wind_pct", "forecast_load_mw_latest_available", "forecast_load_mw_day_ahead", "gen_fuel_coal_mw"]:
            i = players.index(f); U = b[(b >> i) & 1 == 1]; share = m[U, ci] / sz[U]
            best = int(arg[i, 1, ci]); partner = [players[k] for k in range(p) if best >> k & 1]
            rows.append(dict(conf=c, field=f, single=m[1 << i, ci], shapley=share.sum(),
                             pos_dividend_share=share[share > 0].sum() - m[1 << i, ci] * (m[1 << i, ci] > 0),
                             neg_dividend_share=share[share < 0].sum(),
                             M1=M[i, 1, ci], M1_best_partner=",".join(partner) or "∅",
                             pair_interaction_with_partner=(m[(1 << i) | best, ci] if best and popcount([best])[0] == 1 else 0.0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- P6 估计器认证下界
def p6(game, V, M):
    p = M.shape[0]; rows = []
    for est in ["L1x", "L0ensx", "L0", "direct"]:
        E = load_est(game, est)
        if E is None: continue
        Me, arge = mk_all(E)
        for K in [0, 1, 2, 3]:
            T = arge[:, K, :]                                   # (p,C) 估计器选出的最坏背景
            cert = np.stack([V[T[i] | (1 << i), np.arange(V.shape[1])] - V[T[i], np.arange(V.shape[1])] for i in range(p)])
            tm = M[:, K]; em = Me[:, K]
            rows.append(dict(game=game, est=est, K=K, true_M=tm.mean(), est_M=em.mean(), est_bias=(em - tm).mean(),
                             est_MAE=np.abs(em - tm).mean(), cert_lower=cert.mean(),
                             cert_over_true=cert.sum() / tm.sum(), cert_gap_max=(tm - cert).max(),
                             cert_within_002=(tm - cert <= 0.02).mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P1, P2, P3, P6 = [], [], [], []
    for game in ["gameA", "gameB"]:
        V, _ = truth(game); r, M = p1(game, V); P1.append(r)
        P2.append(p2(game, V, M)); P3.append(p3(game, V, M)); P6.append(p6(game, V, M))
        if game == "gameA": VA = V
    P1 = pd.DataFrame(P1).set_index("game").T; P2 = pd.concat(P2); P3 = pd.concat(P3); P6 = pd.concat(P6)
    P4 = p4(VA); P5 = p5(VA)
    for nm, d in [("P1_propositions", P1), ("P2_saturation", P2), ("P3_budget_rules", P3), ("P4_copies", P4), ("P5_harsanyi", P5), ("P6_certify", P6)]:
        d.to_csv(OUT / f"{nm}.csv", index=(nm == "P1_propositions"))
    rep.append("## P1 命题核验\n\n" + md(P1))
    k = P2.assign(gain=P2.Mtop - P2.a)
    rep.append("## P2 协同阶\n\n" + md(k.groupby("game")[["kappa_exact", "kappa_90"]].describe().T))
    rep.append("## P3 预算规则（τ=0.7）\n\n" + md(P3[P3.tau == 0.7].set_index(["game", "rule", "score"]).drop(columns="tau")))
    rep.append("## P4 复制稀释\n\n" + md(P4.set_index(["src", "conf", "copies"])))
    rep.append("## P5 Harsanyi\n\n" + md(P5.set_index(["conf", "field"])))
    rep.append("## P6 认证下界\n\n" + md(P6.set_index(["game", "est", "K"])))
    (OUT / "report.md").write_text("\n\n".join(rep), encoding="utf-8")
    log("MK", "DONE", "P1–P6 完成 -> outputs/report.md")
    print("\n\n".join(rep))
