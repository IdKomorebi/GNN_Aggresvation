# -*- coding: utf-8 -*-
"""123 号：梯度提升树真值（与 111 号 run_stage tree 口径相同），按 TAG123 选择 wide / narrow。"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
OUTD = os.path.join(ROOT, "outputs", os.environ.get("TAG123", "wide"))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
if __name__ == "__main__":
    z = np.load(os.path.join(OUTD, "D.npz")); D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
    keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; t0 = time.time()
    c, v = pipe.truth_tree(D, keys, n_jobs=24); np.savez(os.path.join(OUTD, "truth_tree.npz"), clean=c, val=v); print("完成", time.time() - t0)
