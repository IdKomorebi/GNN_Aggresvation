# -*- coding: utf-8 -*-
"""123 号 步骤 3：单目标 DNN 真值（与 111 号口径相同；GPU 显存被他人占满，改用小块 chunk，必要时用 CPU）。
用法：truth123.py --dev cuda:1 --chunk 256"""
import os, sys, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--dev", default="cuda:1"); ap.add_argument("--chunk", type=int, default=256); a = ap.parse_args()
z = np.load(os.path.join(ROOT, "outputs", "D.npz")); D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
t0 = time.time(); clean, val = pipe.truth_dnn(D, z["masks"], a.dev, seed=0, single=True, chunk=a.chunk)
np.savez(os.path.join(ROOT, "outputs", "truth_single.npz"), clean=clean, val=val); print(f"完成 {len(clean)} 个集合，{time.time() - t0:.0f}s")
