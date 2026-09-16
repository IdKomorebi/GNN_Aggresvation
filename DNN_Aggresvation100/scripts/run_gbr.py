# -*- coding: utf-8 -*-
"""树模型攻击器：sklearn HistGradientBoostingRegressor（LightGBM 同类直方图梯度提升树），逐 (集合, 目标) 专用训练。

口径与 DNN 真值一致：fit 上训练，val 上选迭代轮数（staged_predict），测试集只报告。
超参固定（不在测试集调）：learning_rate 0.05, max_iter 600, max_leaf_nodes 31, min_samples_leaf 20, l2 1.0。
CPU 多进程（每进程单线程），默认 110 进程，给服务器留余量。
用法：run_gbr.py --ds pjm --set k3 [--limit N]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from common100 import load

G = {}


def init(ds):
    G["D"] = load(ds)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def work(args):
    row, cols = args
    from sklearn.ensemble import HistGradientBoostingRegressor
    D = G["D"]; X = D["Xtr"][:, cols]; Xte = D["Xte"][:, cols]
    Xf, Xv = X[D["fit_idx"]], X[D["val_idx"]]
    out_te, out_val = np.zeros(12, np.float32), np.zeros(12, np.float32)
    for c in range(D["Ytr"].shape[1]):
        yf, yv, yte = D["Ytr"][D["fit_idx"], c], D["Ytr"][D["val_idx"], c], D["Yte"][:, c]
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xf, yf)
        best, bi = -1e9, 0
        for it, pv in enumerate(m.staged_predict(Xv)):
            if it % 10 == 9:
                s = r2(yv, pv)
                if s > best: best, bi = s, it
        for it, pt in enumerate(m.staged_predict(Xte)):
            if it == bi: out_te[c] = r2(yte, pt); break
        out_val[c] = best
    return row, out_te, out_val


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--set", default="k3")
    ap.add_argument("--procs", type=int, default=110); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    M = np.load(ROOT / f"outputs/sets/{a.ds}_{a.set}.npy")
    rows = np.arange(len(M))[: a.limit or None]
    tasks = [(int(r), np.where(M[r])[0]) for r in rows]
    te = np.zeros((len(M), 12), np.float32); va = np.zeros((len(M), 12), np.float32); t0 = time.time()
    with Pool(a.procs, initializer=init, initargs=(a.ds,)) as pool:
        for k, (r, o, v) in enumerate(pool.imap_unordered(work, tasks, chunksize=4)):
            te[r], va[r] = o, v
            if k % 2000 == 0: print(f"[gbr {a.ds}_{a.set}] {k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    tag = f"_limit{a.limit}" if a.limit else ""
    np.savez(ROOT / f"outputs/gbr/{a.ds}_{a.set}{tag}.npz", clean=te, val_r2=va, rows=rows)
    print(f"done {len(tasks)} sets {time.time()-t0:.0f}s ({(time.time()-t0)/len(tasks)*a.procs:.2f} s/集合·进程)", flush=True)
