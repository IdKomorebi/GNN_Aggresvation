# -*- coding: utf-8 -*-
"""123 号 步骤 3（CPU 版）：单目标 DNN 真值，按 32 个集合分块、多进程并行（每进程 3 线程）。口径与 111 号完全相同。"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
OUTD = os.path.join(ROOT, "outputs", os.environ.get("TAG123", "wide"))


def work(lohi):
    import torch
    torch.set_num_threads(3)
    sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa
    z = np.load(os.path.join(OUTD, "D.npz")); D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
    lo, hi = lohi; c, v = pipe.truth_dnn(D, z["masks"][lo:hi], "cpu", seed=0, single=True, chunk=hi - lo)
    return lo, c, v


if __name__ == "__main__":
    N = len(np.load(os.path.join(OUTD, "D.npz"))["keys"]); jobs = [(lo, min(lo + 32, N)) for lo in range(0, N, 32)]
    t0 = time.time(); res = []
    with Pool(10) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs)):
            res.append(r); print(f"[{time.strftime('%H:%M:%S')}] {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res.sort(key=lambda x: x[0])
    np.savez(os.path.join(OUTD, "truth_single.npz"), clean=np.concatenate([r[1] for r in res]), val=np.concatenate([r[2] for r in res]))
    print("完成", time.time() - t0)
