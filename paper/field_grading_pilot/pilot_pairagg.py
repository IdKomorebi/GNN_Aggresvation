exec(open(__file__.replace('pilot_pairagg.py','pilot_additive.py')).read().split("agg = collections.defaultdict(list)")[0])
agg = collections.defaultdict(list); size_bias = collections.defaultdict(list)
for conf, items in by_conf.items():
    fit = [(fs, v) for fs, grp, sz, v in items if grp in ("single", "pair", "triple_rand")]
    Xf = X_of([fs for fs, _ in fit]); vf = np.array([v for _, v in fit])
    w, _ = nnls(Xf, g(vf))
    delta = {}
    for fs, grp, sz, v in items:
        if grp == "pair":
            i, j = sorted(fidx[f] for f in fs); delta[(i, j)] = g(v) - w[i] - w[j]
    evs = {"wide": [(fs, v, sz) for fs, grp, sz, v in items if grp == "wide_random" and sz >= 3],
           "top3": [(fs, v, sz) for fs, grp, sz, v in items if grp == "triple_top"],
           "rand3": [(fs, v, sz) for fs, grp, sz, v in items if grp == "triple_rand"]}
    for evname, ev in evs.items():
        preds = collections.defaultdict(list); vs = []
        for fs, v, sz in ev:
            ids = [fidx[f] for f in fs]; base = w[ids].sum()
            ds = [delta[tuple(sorted((a, b)))] for a in ids for b in ids if a < b]
            pos = [d for d in ds if d > 0.05]
            preds["additive"].append(ginv(base))
            preds["+sum_all"].append(ginv(max(base + sum(ds), 0)))
            preds["+sum_syn>0.05"].append(ginv(base + sum(pos)))
            preds["+max_syn"].append(ginv(base + (max(pos) if pos else 0)))
            vs.append(v)
            if evname == "wide":
                b = "3-8" if sz <= 8 else ("9-20" if sz <= 20 else "21-43")
                size_bias[b].append(ginv(base) - v)
        for k, p in preds.items():
            agg[(evname, k)].append((len(vs), metrics(vs, p)))
print(f"{'eval':6}{'model':16}{'MAE':>8}{'bias':>8}{'Spear':>8}{'danger':>8}{'conserv':>8}")
for key in sorted(agg):
    lst = agg[key]; wts = np.array([m for m, _ in lst], float); arr = np.array([mt for _, mt in lst], float)
    mean = np.nansum(arr * wts[:, None], axis=0) / wts.sum()
    print(f"{key[0]:6}{key[1]:16}" + "".join(f"{x:8.3f}" for x in mean))
print("additive(log) bias by size on wide:", {k: (len(v), round(float(np.mean(v)), 3)) for k, v in size_bias.items()})
