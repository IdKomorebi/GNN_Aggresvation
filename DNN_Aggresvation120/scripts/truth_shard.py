# -*- coding: utf-8 -*-
"""120 号：单目标 DNN 真值的分片运行（与 111 号 run_stage 的 single 口径完全相同，只是按集合分片以便多卡并行）。
用法：truth_shard.py --shard 0 --nshard 3 --gpu 0；全部分片完成后 --merge 合并为 outputs/truth_single.npz"""
import os, sys, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=3)
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
O = os.path.join(ROOT, "outputs"); z = np.load(os.path.join(O, "D.npz")); N = len(z["keys"])
cuts = np.linspace(0, N, a.nshard + 1).astype(int)
if a.merge:
    parts = [np.load(os.path.join(O, f"truth_single_shard{s}.npz")) for s in range(a.nshard)]
    np.savez(os.path.join(O, "truth_single.npz"), clean=np.concatenate([p["clean"] for p in parts]), val=np.concatenate([p["val"] for p in parts]))
    print("合并完成", N); sys.exit(0)
D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}; lo, hi = cuts[a.shard], cuts[a.shard + 1]; t0 = time.time()
clean, val = pipe.truth_dnn(D, z["masks"][lo:hi], f"cuda:{a.gpu}", seed=0, single=True)
np.savez(os.path.join(O, f"truth_single_shard{a.shard}.npz"), clean=clean, val=val)
print(f"[{time.strftime('%m-%d %H:%M:%S')}] 分片 {a.shard}: 集合 {lo}–{hi}，用时 {time.time() - t0:.0f}s", flush=True)
