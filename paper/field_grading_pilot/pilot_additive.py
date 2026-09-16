"""预实验：字段级可加模型在原始 R² 标度 vs 对数残差标度上的表现（只读 69 号真值，CPU）。"""
import json, csv, collections
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/outputs/"
subs = json.load(open(ROOT + "subsets.json"))
rows = [r for r in csv.DictReader(open(ROOT + "truth_long.csv")) if r["is_duplicate_entry"] == "False"]
fields = sorted({f for s in subs.values() for f in s["fields"]})
fidx = {f: i for i, f in enumerate(fields)}
n = len(fields)
TAU, EPS = 0.7, 1e-3

def g(v):  # 对数残差标度（高斯下即互信息，单位奈特）
    return -0.5 * np.log(1 - np.clip(v, 0, 1 - EPS))
def ginv(x):
    return 1 - np.exp(-2 * x)

by_conf = collections.defaultdict(list)
for r in rows:
    by_conf[r["conf"]].append((subs[r["sid"]]["fields"], r["group"], int(r["size"]), float(r["truth"])))

def X_of(sets):
    X = np.zeros((len(sets), n))
    for k, fs in enumerate(sets):
        X[k, [fidx[f] for f in fs]] = 1
    return X

def metrics(v, vh):
    v, vh = np.asarray(v), np.clip(np.asarray(vh), 0, 1)
    mae = np.mean(np.abs(v - vh))
    bias = np.mean(vh - v)
    rho = spearmanr(v, vh).correlation if len(v) > 2 else np.nan
    danger = np.mean((vh <= TAU) & (v > TAU))
    conserv = np.mean((vh > TAU) & (v <= TAU))
    return mae, bias, rho, danger, conserv

agg = collections.defaultdict(list)
pair_stats = collections.Counter()
wind = None
for conf, items in by_conf.items():
    single = {fs[0]: v for fs, grp, sz, v in items if grp == "single"}
    fit = [(fs, v) for fs, grp, sz, v in items if grp in ("single", "pair", "triple_rand")]
    ev_wide = [(fs, v, sz) for fs, grp, sz, v in items if grp == "wide_random" and sz >= 3]
    ev_top = [(fs, v, sz) for fs, grp, sz, v in items if grp == "triple_top"]
    Xf = X_of([fs for fs, _ in fit]); vf = np.array([v for _, v in fit])

    w_raw, _ = nnls(Xf, np.clip(vf, 0, 1))                 # 原始标度，c=0
    w_log, _ = nnls(Xf, g(vf))                              # 对数残差标度，c=0
    w_naive_raw = np.array([max(single[f], 0) for f in fields])
    w_naive_log = g(w_naive_raw)
    # 可加 + 两两修正（对数标度）：δ_ij = g(v_ij) − w_i − w_j
    delta = {}
    for fs, grp, sz, v in items:
        if grp == "pair":
            i, j = sorted(fidx[f] for f in fs)
            delta[(i, j)] = g(v) - w_log[i] - w_log[j]
            pair_stats["super" if delta[(i, j)] > 0.05 else ("sub" if delta[(i, j)] < -0.05 else "near")] += 1
            if set(fs) == {"gen_fuel_wind_mw", "gen_fuel_wind_pct"} and conf == "total_gen":
                wind = (single["gen_fuel_wind_mw"], single["gen_fuel_wind_pct"], v)

    def pred(ev, kind):
        out = []
        for fs, v, sz in ev:
            ids = [fidx[f] for f in fs]
            if kind == "naive_raw":  out.append(min(1, w_naive_raw[ids].sum()))
            if kind == "naive_log":  out.append(ginv(w_naive_log[ids].sum()))
            if kind == "fit_raw":    out.append(min(1, w_raw[ids].sum()))
            if kind == "fit_log":    out.append(ginv(w_log[ids].sum()))
            if kind == "fit_log_pair":
                s = w_log[ids].sum() + sum(delta.get(tuple(sorted((a, b))), 0) for a in ids for b in ids if a < b)
                out.append(ginv(max(s, 0)))
        return out

    for evname, ev in (("wide", ev_wide), ("top3", ev_top)):
        v = [x[1] for x in ev]
        for kind in ("naive_raw", "naive_log", "fit_raw", "fit_log", "fit_log_pair"):
            agg[(evname, kind)].append((len(v), metrics(v, pred(ev, kind))))
    # 拟合集内部解释力（对数标度 R²）
    res = g(vf) - Xf @ w_log
    agg[("fit_in", "log_R2")].append((len(vf), (1 - res.var() / g(vf).var(), 0, 0, 0, 0)))
    res_r = np.clip(vf, 0, 1) - Xf @ w_raw
    agg[("fit_in", "raw_R2")].append((len(vf), (1 - res_r.var() / np.clip(vf, 0, 1).var(), 0, 0, 0, 0)))

print("fields", n, "targets", len(by_conf))
print("wind pair (single_mw, single_pct, pair):", wind)
tot = sum(pair_stats.values()); print("pair delta in log scale: ", {k: f"{v}/{tot}={v/tot:.1%}" for k, v in pair_stats.items()})
print(f"{'eval':6}{'model':14}{'MAE':>8}{'bias':>8}{'Spear':>8}{'danger':>8}{'conserv':>8}")
for key in sorted(agg):
    lst = agg[key]; wts = np.array([m for m, _ in lst], float)
    arr = np.array([mt for _, mt in lst], float)
    mean = np.nansum(arr * wts[:, None], axis=0) / wts.sum()
    if key[0] == "fit_in":
        print(f"{key[0]:6}{key[1]:14}{mean[0]:8.3f}  (样本加权平均, 12 目标)")
    else:
        print(f"{key[0]:6}{key[1]:14}" + "".join(f"{x:8.3f}" for x in mean))
