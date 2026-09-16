# -*- coding: utf-8 -*-
"""98 号分析：A 标签口径 / B 44 维保真度 / C 精确博弈（Shapley、交互、分级语义、误差界）。

真值 T = 两个训练种子 clean 口径的平均（val 选轮次，测试集只报告）。
噪声底 = |seed0 − seed1| / 2（T 相对"无限种子平均"的典型误差量级）。
所有 R² 截断到 [0,1]，与历史口径一致。
"""
import sys, json, glob, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import games as G

MODELS = ["lin", "lin2", "poly2", "direct", "L0", "L0x", "L1", "L1x", "L1ens", "L0ensx", "Lmixx", "rffx", "mono0", "mono1", "mono10", "mono0r", "mono1r", "mono10r"]
OUT = ROOT / "outputs/analysis"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(ROOT / "outputs/masks/masks_meta.json"))
CONF = meta["conf"]


def load_truth(s):
    parts = {}
    for f in sorted(glob.glob(str(ROOT / f"outputs/truth/{s}_seed*_s*of3.npz"))):
        seed = int(f.split("_seed")[1].split("_")[0]); z = np.load(f)
        for k in ["clean", "leak69", "oracle_best", "val_r2", "ridge"]:
            parts.setdefault((seed, k), []).append((z["idx"], z[k]))
    out = {}
    for (seed, k), lst in parts.items():
        n = sum(len(i) for i, _ in lst); arr = np.zeros((n, 12), np.float32)
        for i, v in lst: arr[i] = v
        out[(seed, k)] = np.clip(arr, 0, 1)
    return out


def load_est(s, m):
    fs = sorted(glob.glob(str(ROOT / f"outputs/est/{s}_{m}_s*of*.npz")))
    if not fs: return None
    need = int(fs[0].split("of")[-1].split(".")[0])
    if len(fs) != need: return None                      # 分片未齐，跳过
    zs = [np.load(f) for f in fs]; n = sum(len(z["idx"]) for z in zs)
    arr = np.zeros((n, 12), np.float32)
    for z in zs: arr[z["idx"]] = z["v"]
    return np.clip(arr, 0, 1)


def band(sz):
    return pd.cut(sz, [0, 1, 2, 8, 20, 44], labels=["1", "2", "3-8", "9-20", "21-44"])


def md(df, fmt="{:.4f}"):
    return df.to_markdown(floatfmt=".4f")


# ------------------------------------------------------------------ A 标签口径
def part_a(rep):
    Tr = load_truth("rand"); rm = pd.DataFrame(meta["rand_meta"]); b = band(rm["size"].values)
    c0, c1 = Tr[(0, "clean")], Tr[(1, "clean")]
    rows = []
    for lab in b.categories:
        k = np.asarray(b == lab)
        rows.append(dict(band=lab, n_sets=int(k.sum()),
                         mean_T=((c0[k] + c1[k]) / 2).mean(),
                         seed_noise=np.abs(c0[k] - c1[k]).mean() / 2,
                         leak69_minus_clean=((Tr[(0, "leak69")] - c0)[k]).mean(),
                         bestepoch_minus_clean=((Tr[(0, "oracle_best")] - c0)[k]).mean(),
                         ridge_minus_mlp=((Tr[(0, "ridge")] - (c0 + c1) / 2)[k]).mean(),
                         ridge_wins=((Tr[(0, "ridge")] > (c0 + c1) / 2)[k]).mean()))
    df = pd.DataFrame(rows).set_index("band"); df.to_csv(OUT / "A_label_protocol.csv")
    rep.append("## A. 真值标签口径（rand 集，4090 个集合×12 目标）\n\n" + md(df) + "\n")
    # 单调性：真值自身
    return Tr


# ------------------------------------------------------------------ B 44 维保真度
def part_b(rep, Tr):
    rm = pd.DataFrame(meta["rand_meta"]); b = band(rm["size"].values)
    T = (Tr[(0, "clean")] + Tr[(1, "clean")]) / 2
    ests = {m: load_est("rand", m) for m in MODELS}
    ests = {m: v for m, v in ests.items() if v is not None}
    ests["truth_seed1"] = Tr[(1, "clean")]; ests["ridge_attacker"] = Tr[(0, "ridge")]
    rows = []
    for m, E in ests.items():
        r = dict(model=m)
        for lab in b.categories:
            k = np.asarray(b == lab); r[f"MAE_{lab}"] = np.abs(E[k] - T[k]).mean()
        r["MAE_all"] = np.abs(E - T).mean(); r["bias_all"] = (E - T).mean()
        r["spearman_within_conf"] = np.mean([spearmanr(E[:, c], T[:, c])[0] for c in range(12)])
        hi = T > 0.7; r["under_on_T>0.7"] = (T - E)[hi].mean()
        r["danger@0.7"] = ((E <= 0.7) & hi).sum() / hi.sum()
        # 边际
        qs = rm[rm.kind == "marg_S"].index.values; qi = rm[rm.kind == "marg_Si"].index.values
        dT, dE = T[qi] - T[qs], E[qi] - E[qs]
        r["marg_MAE"] = np.abs(dE - dT).mean(); r["marg_corr"] = np.corrcoef(dE.ravel(), dT.ravel())[0, 1]
        r["mono_viol(<-0.02)"] = (dE < -0.02).mean()
        # 二阶交互
        def q(tag): return rm[rm.kind == "quad_" + tag].index.values
        iT = T[q("Sij")] - T[q("Si")] - T[q("Sj")] + T[q("S")]
        iE = E[q("Sij")] - E[q("Si")] - E[q("Sj")] + E[q("S")]
        r["inter_MAE"] = np.abs(iE - iT).mean(); r["inter_corr"] = np.corrcoef(iE.ravel(), iT.ravel())[0, 1]
        # 字段对协同 syn = v(ij) - max(v(i), v(j))
        pr = rm[rm.kind == "pair"]; si = rm[rm.kind == "single"].index.values
        synT = T[pr.index.values] - np.maximum(T[si[pr.i.values.astype(int)]], T[si[pr.j.values.astype(int)]])
        synE = E[pr.index.values] - np.maximum(E[si[pr.i.values.astype(int)]], E[si[pr.j.values.astype(int)]])
        r["pairsyn_spearman"] = spearmanr(synE.ravel(), synT.ravel())[0]
        top = synT.ravel() >= np.quantile(synT.ravel(), 0.99)
        r["pairsyn_top1%_under"] = (synT.ravel() - synE.ravel())[top].mean()
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model"); df.to_csv(OUT / "B_fidelity_rand.csv")
    tmono = ((Tr[(0, "clean")][rm[rm.kind == "marg_Si"].index.values] - Tr[(0, "clean")][rm[rm.kind == "marg_S"].index.values]) < -0.02).mean()
    rep.append("## B. 44 维未见集合保真度（真值=两种子 clean 平均）\n\n" + md(df.T) +
               f"\n\n单种子真值自身的单调违例率(<-0.02)：{tmono:.4f}\n")


# ------------------------------------------------------------------ C 精确博弈
def game_ops(V):
    phi = G.shapley(V); s2 = G.sii(V); st = G.stii(V); f1, f2 = G.faith2(V)
    keys = list(s2)
    return dict(phi=phi, sii=np.stack([s2[k] for k in keys]), stii=np.stack([st[k] for k in keys]),
                faith1=f1, faith2=np.stack([f2[k] for k in keys]),
                M0=G.maxmarg(V, 0), M1=G.maxmarg(V, 1), M2=G.maxmarg(V, 2), loo=G.loo(V))


def cmp(A, Bv, name):
    """A,B: (items, C)。逐目标 Spearman/Kendall 取均值。"""
    rho = np.nanmean([spearmanr(A[:, c], Bv[:, c])[0] for c in range(A.shape[1])])
    tau = np.nanmean([kendalltau(A[:, c], Bv[:, c])[0] for c in range(A.shape[1])])
    return {f"{name}_MAE": np.abs(A - Bv).mean(), f"{name}_max": np.abs(A - Bv).max(),
            f"{name}_rho": rho, f"{name}_tau": tau}


def part_c(rep, game):
    Tr = load_truth(game)
    if (1, "clean") not in Tr: rep.append(f"## C. {game}: 真值未完成\n"); return
    players = meta[game]; p = len(players)
    T = (Tr[(0, "clean")] + Tr[(1, "clean")]) / 2
    OT = game_ops(T); O0 = game_ops(Tr[(0, "clean")]); O1 = game_ops(Tr[(1, "clean")])
    ests = {m: load_est(game, m) for m in MODELS}; ests = {m: v for m, v in ests.items() if v is not None}
    ests["truth_seed0"] = Tr[(0, "clean")]; ests["leak69_seed0"] = Tr[(0, "leak69")]
    sage = ROOT / f"outputs/sage_{game}.npz"
    if sage.exists():
        z = np.load(sage)
        for s in range(3):
            if f"V_seed{s}" in z: ests[f"SAGE_fullmodel_s{s}"] = np.clip(z[f"V_seed{s}"], 0, 1)
    rows = []
    for m, E in ests.items():
        OE = game_ops(E); r = dict(model=m)
        eps = np.abs(E - T); r["eps_max"] = eps.max(); r["eps_mean"] = eps.mean()
        for k in ["phi", "sii", "stii", "faith2", "M1", "M2"]:
            r.update(cmp(OE[k], OT[k], k))
        top3 = [len(set(np.argsort(-OE["phi"][:, c])[:3]) & set(np.argsort(-OT["phi"][:, c])[:3])) / 3 for c in range(12)]
        r["phi_top3"] = np.mean(top3)
        # 误差界：|Δφ| ≤ 2ε_max（GPT），与实际对照
        r["bound_2eps"] = 2 * eps.max(axis=0).mean(); r["phi_err_max_per_conf"] = np.abs(OE["phi"] - OT["phi"]).max(0).mean()
        r["mono_viol"] = mono_viol(E)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model"); df.to_csv(OUT / f"C_{game}_fidelity.csv")
    rep.append(f"## C. 精确博弈 {game}（p={p}，2^{p} 联盟全枚举）\n\n" + md(df.T) + "\n")
    # 语义：各分级量在真值上的排序差异
    sem = []
    for c, cn in enumerate(CONF):
        for i, f in enumerate(players):
            sem.append(dict(conf=cn, field=f, single=OT["M0"][i, c], shapley=OT["phi"][i, c], M1=OT["M1"][i, c],
                            M2=OT["M2"][i, c], loo=OT["loo"][i, c], faith1=OT["faith1"][i, c]))
    sem = pd.DataFrame(sem); sem.to_csv(OUT / f"C_{game}_semantics.csv", index=False)
    cor = []
    for cn, d in sem.groupby("conf"):
        cor.append(dict(conf=cn, rho_shap_single=spearmanr(d.shapley, d.single)[0], rho_shap_M1=spearmanr(d.shapley, d.M1)[0],
                        rho_shap_loo=spearmanr(d.shapley, d.loo)[0], rho_M1_single=spearmanr(d.M1, d.single)[0],
                        shap_sum=d.shapley.sum(), vN=T[-1, CONF.index(cn)]))
    rep.append(f"### {game} 真值上各字段量的一致性\n\n" + md(pd.DataFrame(cor).set_index("conf")) + "\n")
    # 交互：真值噪声底
    rep.append(f"真值种子间：φ MAE {np.abs(O0['phi']-O1['phi']).mean()/2:.4f}（/2 后），SII MAE {np.abs(O0['sii']-O1['sii']).mean()/2:.4f}，"
               f"Faith2 MAE {np.abs(O0['faith2']-O1['faith2']).mean()/2:.4f}\n")
    np.savez(OUT / f"C_{game}_truth_ops.npz", **{k: v for k, v in OT.items()})


def mono_viol(V, thr=0.02):
    p = int(np.log2(len(V))); b = np.arange(2 ** p); c = 0; n = 0
    for i in range(p):
        S = b[(b >> i) & 1 == 0]; d = V[S | (1 << i)] - V[S]; c += (d < -thr).sum(); n += d.size
    return c / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="abc"); a = ap.parse_args()
    rep = ["# 98 号分析输出（自动生成）\n"]
    Tr = part_a(rep) if "a" in a.parts or "b" in a.parts else None
    if "b" in a.parts: part_b(rep, Tr)
    if "c" in a.parts:
        for gm in ["gameB", "gameA"]: part_c(rep, gm)
    (OUT / f"report_{a.parts}.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
