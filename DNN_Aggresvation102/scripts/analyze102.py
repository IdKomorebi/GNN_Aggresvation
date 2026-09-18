# -*- coding: utf-8 -*-
"""102 号分析（P0-2）：
A) 真值攻击器族诊断：单目标 DNN 专用重训 vs 多目标 DNN vs 树（抽样集合）；
B) single-target vs multi-target 通用模型：V-MAE、marginal-MAE、M-MAE(K=1,2)、Kendall、档位一致、τ-critical recall/false-safe、时延。
真值口径：FINAL_PROTOCOL §7（攻击器族取 max，val 选择；单调闭包 val 选子集）。
"""
import sys, glob, json, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent; R100 = REPO / "DNN_Aggresvation100"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of, attack_max
from analyze_est import marg_tables
A = ROOT / "outputs/analysis"; GR = [0.05, 0.2, 0.5]; TAUS = [0.5, 0.7, 0.9]
rep = ["# 102 号分析：single-target vs multi-target（P0-2）\n"]


def md(df, fl=".4f"): return df.to_markdown(index=False, floatfmt=fl)


def shards(pat, n, key, ncol=12):
    fs = sorted(glob.glob(pat))
    if not fs: return None
    need = int(fs[0].split("of")[-1].split(".")[0])
    if len(fs) != need: return None
    out = np.zeros((n, ncol), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


def truth_k3(ds, keys):
    """100 号 k4 真值中取 k3 子集 + 树模型，攻击器族取 max，再做单调闭包。"""
    meta4 = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys4 = meta4["keys"]
    n4 = len(keys4); c4 = np.zeros((n4, 12), np.float32); v4 = np.zeros((n4, 12), np.float32)
    for f in sorted(glob.glob(str(R100 / f"outputs/truth/{ds}_k4_seed0_s*of3.npz"))):
        z = np.load(f); c4[z["idx"]] = z["clean"]; v4[z["idx"]] = z["val_r2"]
    sel = np.array([len(k) <= 3 for k in keys4])
    dnn_c, dnn_v = np.clip(c4[sel], -1, 1), v4[sel]
    g = np.load(R100 / f"outputs/gbr/{ds}_k3.npz")
    att, att_v, _ = attack_max([dnn_c, np.clip(g["clean"], -1, 1)], [dnn_v, g["val_r2"]])
    return closure(att, keys, Vval=att_v), dnn_c, dnn_v, np.clip(g["clean"], -1, 1), g["val_r2"]


def mk_from_D(Dt, bsz, K):
    sel = bsz <= K; return Dt[:, sel].max(1), Dt[:, sel].argmax(1)


def critical(Vbar, keys, act, tau, K):
    """真值/估计的 τ-critical 布尔表 (n_act, C)。"""
    idx = index_of(keys); C = Vbar.shape[1]; out = np.zeros((len(act), C), bool)
    for p, i in enumerate(act):
        others = [a for a in act if a != i]
        hit = np.zeros(C, bool)
        for s in range(K + 1):
            for T in combinations(others, s):
                vT = Vbar[idx[T]] if T else np.zeros(C)
                hit |= (vT <= tau) & (Vbar[idx[tuple(sorted(T + (i,)))]] > tau)
        out[p] = hit
    return out


rows_a, rows_b, rows_c = [], [], []
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys, act = meta["keys_k3"], meta["active"]; n = len(keys)
    Vt, dnn_c, dnn_v, gbr_c, gbr_v = truth_k3(ds, keys)
    # ---------- A 攻击器族诊断（单目标 DNN 抽样）
    st_c = np.full((n, 12), np.nan, np.float32); st_v = np.full((n, 12), np.nan, np.float32)
    for t in range(12):
        fs = sorted(glob.glob(str(ROOT / f"outputs/truth/{ds}_k3_samp_single_t{t}_s*of3.npz")))
        for f in fs:
            z = np.load(f); st_c[z["idx"], t] = z["clean"][:, 0]; st_v[z["idx"], t] = z["val_r2"][:, 0]
    m = ~np.isnan(st_c)
    if m.any():
        sz = np.array([len(k) for k in keys])
        for s in [1, 2, 3]:
            k = m & (sz[:, None] == s)
            rows_a.append(dict(数据集=ds, 规模=s, 样本数=int(k.sum()), 单目标DNN=np.clip(st_c[k], 0, 1).mean(),
                               多目标DNN=np.clip(dnn_c[k], 0, 1).mean(), 树模型=np.clip(gbr_c[k], 0, 1).mean(),
                               单目标更强占比=float((st_c[k] > dnn_c[k]).mean()), 单目标被val选中占比=float((st_v[k] > dnn_v[k]).mean()),
                               单目标减多目标均值=float((np.clip(st_c[k], 0, 1) - np.clip(dnn_c[k], 0, 1)).mean())))
    # ---------- B 估计器对照
    Dt, bk = marg_tables(Vt, keys, act, kmax=2); bsz = np.array([len(T) for T in bk[0]])
    ests = {}
    s1 = np.zeros((n, 12), np.float32); ok = True
    for t in range(12):
        v = shards(str(ROOT / f"outputs/est/{ds}_k3_single_t{t}_s*of3.npz"), n, "v", 1)
        if v is None: ok = False; break
        s1[:, t] = v[:, 0]
    if ok: ests["single-target φ+x,x²"] = s1
    s0 = np.zeros((n, 12), np.float32); ok = True
    for t in range(12):
        v = shards(str(ROOT / f"outputs/est/{ds}_k3_single_nophi_t{t}_s*of3.npz"), n, "v", 1)
        if v is None: ok = False; break
        s0[:, t] = v[:, 0]
    if ok: ests["无 φ：x,x² ridge"] = s0
    mx = shards(str(ROOT / f"outputs/est/{ds}_k3_multix_s*of3.npz"), n, "v")
    if mx is not None: ests["multi-target φ+x,x²"] = mx
    meta4 = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys4 = meta4["keys"]
    e4 = shards(str(R100 / f"outputs/est/{ds}_k4_L0ensx_s*of*.npz"), len(keys4), "v")
    if e4 is not None: ests["multi-target 三种子集成+x,x²"] = e4[np.array([len(k) <= 3 for k in keys4])]
    for nm, E in ests.items():
        Eb = closure(np.clip(E, 0, 1), keys); De, _ = marg_tables(Eb, keys, act, kmax=2)
        r = dict(数据集=ds, 估计器=nm, V_MAE=np.abs(Eb - Vt).mean(), V偏差=(Eb - Vt).mean(),
                 边际MAE=np.abs(De - Dt).mean(), 边际q95=np.quantile(np.abs(De - Dt), 0.95))
        for K in [1, 2]:
            Me, _ = mk_from_D(De, bsz, K); Mt, At = mk_from_D(Dt, bsz, K)
            L = np.take_along_axis(Dt[:, bsz <= K], De[:, bsz <= K].argmax(1)[:, None, :], 1)[:, 0]
            r[f"M_MAE_K{K}"] = np.abs(Me - Mt).mean(); r[f"M偏差_K{K}"] = (Me - Mt).mean()
            r[f"Kendall_K{K}"] = np.nanmean([kendalltau(Me[:, c], Mt[:, c])[0] for c in range(12)])
            r[f"档位一致_K{K}"] = (np.digitize(Me, GR) == np.digitize(Mt, GR)).mean()
            r[f"见证召回_K{K}"] = (L >= Mt - 0.02).mean()
        for tau in [0.7]:
            for K in [1, 2]:
                ct, ce = critical(Vt, keys, act, tau, K), critical(Eb, keys, act, tau, K)
                tp = (ct & ce).sum(); r[f"τ{tau}critical_recall_K{K}"] = tp / max(ct.sum(), 1)
                r[f"τ{tau}critical_prec_K{K}"] = tp / max(ce.sum(), 1)
                r[f"τ{tau}false_safe_K{K}"] = (ct & ~ce).sum() / max(ct.sum(), 1)
        rows_b.append(r)
        np.savez(A / f"{ds}_est_{nm.split()[0].replace('：','_')}.npz", V=Eb)
    np.savez(A / f"{ds}_truth_k3.npz", V=Vt)
# 时延
ev = [json.loads(l) for l in open(ROOT / "outputs/events.jsonl", encoding="utf-8")] if (ROOT / "outputs/events.jsonl").exists() else []
for ds in ["pjm", "caiso"]:
    for stage, nm in [("EST1", "single-target（每目标一次查询）"), ("ESTM", "multi-target（一次查询给 12 目标）")]:
        es = [e for e in ev if e["stage"] == stage and e["note"].startswith(f"{ds}_k3") and "nophi" not in e["note"]]
        if es:
            per = float(np.mean([e["ms_per_set"] for e in es]))
            rows_c.append(dict(数据集=ds, 估计器=nm, 每次查询ms=per, 折算每集合12目标ms=per * (12 if stage == "EST1" else 1)))
    tr = [e for e in ev if e["stage"] == "SINGLE" and e["note"].startswith(ds)]
    if tr: rows_c.append(dict(数据集=ds, 估计器="single-target 预学习（12 目标合计）", 每次查询ms=np.nan, 折算每集合12目标ms=np.nan,
                              训练秒=float(np.sum([e["train_sec"] for e in tr]))))
for t, rws in [("A. 真值攻击器族诊断（抽样集合，规模分层）", rows_a), ("B. 估计器对照（k3 全部集合，真值=攻击器族取 max 后单调闭包）", rows_b), ("C. 查询时延与预学习成本", rows_c)]:
    if rws:
        df = pd.DataFrame(rws); df.to_csv(A / f"102_{t[0]}.csv", index=False); rep.append(f"## {t}\n\n" + md(df))
(A / "report102.md").write_text("\n\n".join(rep), encoding="utf-8"); print("\n\n".join(rep))
