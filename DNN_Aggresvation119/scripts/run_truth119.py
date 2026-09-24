# -*- coding: utf-8 -*-
"""119 号：带随机种子的三攻击器真值（与 111 号 run_stage 的口径完全相同，只多一个 --seed）。
  single / multi：pipe.truth_dnn(seed=...)；tree：梯度提升树两种配置 val 选择，random_state=seed（影响早停划分）。
用法：run_truth119.py --grp rts_seed1 --stage single --seed 1 --gpu 0"""
import os, sys, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
G = {}


def _init(D, seed):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1); G.update(D); G["seed"] = seed


def _tree_one(s):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    s = list(s); Xtr, Ytr, Xte, Yte, fi, vi = G["Xtr"], G["Ytr"], G["Xte"], G["Yte"], G["fit_idx"], G["val_idx"]
    out_t, out_v = [], []
    for c in range(Ytr.shape[1]):
        best = (-np.inf, None)
        for lr, leaf in [(0.05, 15), (0.1, 31)]:
            m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaf, max_iter=300, early_stopping=True,
                                              validation_fraction=0.15, random_state=G["seed"])
            m.fit(Xtr[fi][:, s], Ytr[fi, c]); v = r2_score(Ytr[vi, c], m.predict(Xtr[vi][:, s]))
            if v > best[0]:
                best = (v, m)
        out_v.append(best[0]); out_t.append(r2_score(Yte[:, c], best[1].predict(Xte[:, s])))
    return out_t, out_v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--grp", required=True); ap.add_argument("--stage", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--jobs", type=int, default=40)
    a = ap.parse_args(); O = os.path.join(ROOT, "groups", a.grp, "outputs")
    z = np.load(os.path.join(O, "D.npz")); D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
    masks = z["masks"]; keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; t0 = time.time()
    if a.stage in ("single", "multi"):
        clean, val = pipe.truth_dnn(D, masks, f"cuda:{a.gpu}", seed=a.seed, single=(a.stage == "single"))
    else:
        with Pool(a.jobs, initializer=_init, initargs=(D, a.seed)) as pool:
            res = pool.map(_tree_one, keys, chunksize=8)
        clean = np.array([r[0] for r in res], np.float32); val = np.array([r[1] for r in res], np.float32)
    np.savez(os.path.join(O, f"truth_{a.stage}.npz"), clean=clean, val=val)
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {a.grp} {a.stage} seed={a.seed}：{len(keys)} 个集合，用时 {time.time() - t0:.0f}s", flush=True)
