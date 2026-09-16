"""预实验：字段级"有效信息几何"定义式 vs 对数可加(+协同取最大)。只读 69 号真值, CPU。
模型: 潜变量 z~N(0,I_d); 字段 i 观测 u_i'z+ε_i; 目标 Y=a'z+σε, ||a||≤1.
      v(S) = a'(I-(I+Σ_{i∈S}u_i u_i')^{-1})a   —— 空集为0、单调、≤1 由构造保证；共线→冗余、噪声抵消→协同。
"""
import json, csv, collections, sys
import numpy as np, torch
from scipy.optimize import nnls
from scipy.stats import spearmanr, kendalltau

torch.set_num_threads(8)
ROOT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/outputs/"
subs = json.load(open(ROOT + "subsets.json"))
rows = [r for r in csv.DictReader(open(ROOT + "truth_long.csv")) if r["is_duplicate_entry"] == "False"]
fields = sorted({f for s in subs.values() for f in s["fields"]})
fidx = {f: i for i, f in enumerate(fields)}; n = len(fields)
TAU, EPS = 0.7, 1e-3
g = lambda v: -0.5 * np.log(1 - np.clip(v, 0, 1 - EPS))
ginv = lambda x: 1 - np.exp(-2 * x)

by_conf = collections.defaultdict(list)
for r in rows:
    by_conf[r["conf"]].append((tuple(sorted(fidx[f] for f in subs[r["sid"]]["fields"])), r["group"], float(r["truth"])))

def X_of(sets):
    X = np.zeros((len(sets), n))
    for k, ids in enumerate(sets): X[k, list(ids)] = 1
    return X

def metrics(v, vh):
    v, vh = np.asarray(v), np.clip(np.asarray(vh), 0, 1)
    return (np.mean(np.abs(v - vh)), np.mean(vh - v), spearmanr(v, vh).correlation,
            np.mean((vh <= TAU) & (v > TAU)), np.mean((vh > TAU) & (v <= TAU)))

import os
LOSS=os.environ.get("LOSS","v"); WD=float(os.environ.get("WD","1e-5")); STEPS=int(os.environ.get("STEPS","2500"))
dev="cuda"
def fit_geom(sets, v, d, seed, steps=None):
    steps = steps or STEPS
    torch.manual_seed(seed)
    X = torch.tensor(X_of(sets), dtype=torch.float64, device=dev)
    y = torch.tensor(np.clip(v, 0, 1), dtype=torch.float64, device=dev)
    U = (0.3 * torch.randn(n, d, dtype=torch.float64, device=dev)).requires_grad_()
    ar = torch.randn(d, dtype=torch.float64, device=dev).requires_grad_(); s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([U, ar, s], lr=float(os.environ.get("LR","0.03")))
    def pred(Xb):
        a = ar / ar.norm() * torch.sigmoid(s * 3)
        P = torch.einsum("id,ie->ide", U, U)
        M = torch.einsum("bi,ide->bde", Xb, P) + torch.eye(d, dtype=torch.float64, device=dev)
        sol = torch.linalg.solve(M, a.expand(len(Xb), d).unsqueeze(-1)).squeeze(-1)
        return a @ a - sol @ a
    for t in range(steps):
        opt.zero_grad()
        p = pred(X)
        gl = lambda x: -0.5*torch.log(1-torch.clamp(x,0,1-1e-3))
        loss = (((p - y) ** 2).mean() if LOSS=="v" else ((gl(p)-gl(y))**2).mean() + ((p-y)**2).mean()) + WD * (U ** 2).sum()
        loss.backward(); opt.step()
    return (lambda sets2: pred(torch.tensor(X_of(sets2), dtype=torch.float64, device=dev)).detach().cpu().numpy()), loss.item()

def grade_k1(vfun):
    """最坏1字段背景下的边际泄露: M_i = max(v{i}, max_j v{i,j}-v{j})，以及取到最大值的伙伴 j。"""
    singles = [(i,) for i in range(n)]
    vs = vfun(singles)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    vp = vfun(pairs); Vp = np.zeros((n, n))
    for (i, j), x in zip(pairs, vp): Vp[i, j] = Vp[j, i] = x
    marg = Vp - vs[None, :]; np.fill_diagonal(marg, -1)
    return np.maximum(vs, marg.max(1)), marg.argmax(1), vs

LEVELS = [0.05, 0.2, 0.5]
lvl = lambda x: np.digitize(x, LEVELS)

PAIR_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DIMS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 16]
rng = np.random.default_rng(0)
agg = collections.defaultdict(list); gagg = collections.defaultdict(list); examples = []
for conf, items in sorted(by_conf.items()):
    truth = {ids: v for ids, grp, v in items}
    pairs = [x for x in items if x[1] == "pair"]
    keep = rng.random(len(pairs)) < PAIR_FRAC
    fit = [x for x in items if x[1] in ("single", "triple_rand")] + [p for p, k in zip(pairs, keep) if k]
    evsets = {"pair_held": [p for p, k in zip(pairs, keep) if not k],
              "top3": [x for x in items if x[1] == "triple_top"],
              "wide": [x for x in items if x[1] == "wide_random" and len(x[0]) >= 3]}
    fs = [x[0] for x in fit]; fv = np.array([x[2] for x in fit])
    w, _ = nnls(X_of(fs), g(fv))
    delta = {}
    for ids, grp, v in fit:
        if grp == "pair":
            dd = g(v) - w[list(ids)].sum()
            if dd > 0.05: delta[ids] = dd
    models = {
        "add_log": lambda S: np.array([ginv(w[list(ids)].sum()) for ids in S]),
        "add+maxpair": lambda S: np.array([ginv(w[list(ids)].sum() + max([delta.get((a, b), 0) for a in ids for b in ids if a < b] + [0])) for ids in S]),
    }
    for d in DIMS:
        best = min((fit_geom(fs, fv, d, sd) for sd in range(3)), key=lambda t: t[1])
        models[f"geom_d{d}"] = best[0]
    # 真值 k=1 分级（单字段+字段对真值穷举）
    tv = lambda S: np.array([truth[tuple(sorted(ids))] for ids in S])
    Gt, jt, vst = grade_k1(tv)
    for name, f in models.items():
        for en, ev in evsets.items():
            if ev: agg[(en, name)].append((len(ev), metrics([x[2] for x in ev], f([x[0] for x in ev]))))
        Gm, jm, _ = grade_k1(f)
        gagg[name].append((kendalltau(Gt, Gm).correlation, np.mean(lvl(Gt) == lvl(Gm)),
                           np.mean(np.abs(lvl(Gt) - lvl(Gm)) >= 2), np.mean(jm == jt)))
    gagg["naive_v{i}"].append((kendalltau(Gt, vst).correlation, np.mean(lvl(Gt) == lvl(vst)),
                               np.mean(np.abs(lvl(Gt) - lvl(vst)) >= 2), np.nan))
    up = np.sum(lvl(Gt) > lvl(vst)); examples.append((conf, up))
    if conf == "total_gen":
        f = models[f"geom_d{DIMS[-1]}"]
        for nm in ("gen_fuel_wind_mw", "gen_fuel_wind_pct"):
            i = fidx[nm]; Gm, jm, vsm = grade_k1(f)
            print(f"[example total_gen] {nm}: 真值 v_i={vst[i]:.3f} M_i={Gt[i]:.3f} 伙伴={fields[jt[i]]} | 几何 v_i={vsm[i]:.3f} M_i={Gm[i]:.3f} 伙伴={fields[jm[i]]}")

print(f"PAIR_FRAC={PAIR_FRAC}  fields={n} targets={len(by_conf)}")
print(f"{'eval':10}{'model':14}{'MAE':>8}{'bias':>8}{'Spear':>8}{'danger':>8}{'conserv':>8}")
for key in sorted(agg):
    lst = agg[key]; wts = np.array([m for m, _ in lst], float); arr = np.array([mt for _, mt in lst], float)
    print(f"{key[0]:10}{key[1]:14}" + "".join(f"{x:8.3f}" for x in np.nansum(arr * wts[:, None], 0) / wts.sum()))
print("\nk=1 最坏背景分级 vs 真值（12 目标平均）: Kendall τ | 4档一致率 | 差≥2档比例 | 伙伴命中率")
for name, lst in gagg.items():
    a = np.nanmean(np.array(lst, float), 0); print(f"  {name:14}" + "".join(f"{x:8.3f}" for x in a))
print("考虑协同后比单字段升档的字段数(真值,每目标):", examples)
