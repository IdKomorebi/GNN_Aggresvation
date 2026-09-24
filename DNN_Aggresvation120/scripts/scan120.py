# -*- coding: utf-8 -*-
"""120 号（只汇报、不进论文）：大字段空间上的扫描。
在 120 号数据上训练本组主干（与 111/116 号相同的 pipe.train_oracle），用新主口径的读出设置估计全部规模 ≤3 的集合。
为控制成本只用单种子（相当于 118 号的"本文·单种子"，精度与三种子相近）。可分片：--shard i --nshard n。
用法：scan120.py --stage train --gpu 0；scan120.py --stage est --shard 0 --nshard 3 --gpu 0；scan120.py --stage merge --nshard 3"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation116", "src")); import bb116  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=3); a = ap.parse_args()
O = os.path.join(ROOT, "outputs"); z = np.load(os.path.join(O, "D.npz")); D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]; t0 = time.time()
if a.stage == "train":
    m, inf = pipe.train_oracle(D, 0, f"cuda:{a.gpu}"); torch.save(m.state_dict(), os.path.join(O, "oracle_seed0.pt")); print("主干", inf, flush=True)
elif a.stage == "est":
    N = len(z["keys"]); cuts = np.linspace(0, N, a.nshard + 1).astype(int); lo, hi = cuts[a.shard], cuts[a.shard + 1]
    models = bb116.load_models([os.path.join(O, "oracle_seed0.pt")], p, C, f"cuda:{a.gpu}")
    E, _, sec = bb116.estimate_E(D["Xtr"], D["Ytr"], D["Xte"], D["Yte"], models, z["masks"][lo:hi], list(range(p)), f"cuda:{a.gpu}",
                                 log=lambda s: print(s, flush=True))
    np.savez(os.path.join(O, f"est1_shard{a.shard}.npz"), E=E, sec=sec); print(f"分片 {a.shard}：{hi - lo} 个集合，{sec * 1000:.1f} ms/集合", flush=True)
elif a.stage == "merge":
    parts = [np.load(os.path.join(O, f"est1_shard{s}.npz")) for s in range(a.nshard)]
    np.savez(os.path.join(O, "est1.npz"), E=np.concatenate([q["E"] for q in parts]), sec=np.mean([float(q["sec"]) for q in parts]))
    print("合并完成")
print(f"用时 {time.time() - t0:.0f}s")
