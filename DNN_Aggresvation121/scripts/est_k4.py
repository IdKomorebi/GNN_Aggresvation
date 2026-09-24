# -*- coding: utf-8 -*-
"""121 号（只汇报、不进论文）步骤 1：新主口径 E 在规模为 4 的集合上的估计（再定级需要 V(b∪T∪i)，|T|≤2）。
复用各号已训练的 3 种子本组主干；读出设置与 116 号完全相同。用法：est_k4.py --grp pjm_load|rts --gpu 1"""
import os, sys, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation116", "src")); import bb116  # noqa: E402
SRC = {"pjm_load": os.path.join(REPO, "DNN_Aggresvation116/groups/pjm_load/outputs"), "rts": os.path.join(REPO, "DNN_Aggresvation111/outputs")}
ap = argparse.ArgumentParser(); ap.add_argument("--grp", required=True); ap.add_argument("--gpu", type=int, default=1); a = ap.parse_args()
O = SRC[a.grp]; z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
s4 = np.array([len(k) == 4 for k in keys]); p, C = z["Xtr"].shape[1], z["Ytr"].shape[1]
models = bb116.load_models([os.path.join(O, f"oracle_seed{s}.pt") for s in range(3)], p, C, f"cuda:{a.gpu}")
E, E1, sec = bb116.estimate_E(z["Xtr"], z["Ytr"], z["Xte"], z["Yte"], models, z["masks"][s4], list(range(p)), f"cuda:{a.gpu}",
                              log=lambda m: print(m, flush=True))
np.savez(os.path.join(ROOT, "outputs", f"est4_{a.grp}.npz"), sel4=s4, E=E, E1=E1, sec_per_set=sec)
print(f"{a.grp}: {int(s4.sum())} 个规模 4 集合，{sec * 1000:.0f} ms/集合")
