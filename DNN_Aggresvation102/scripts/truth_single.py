# -*- coding: utf-8 -*-
"""真值攻击器族扩展：单目标 DNN 专用重训（协议同 FINAL_PROTOCOL §7 攻击器 1）。
用法：truth_single.py --ds pjm --set k3 --target 3 --shard 0 --nshard 3 [--limit N]"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation98/src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load
from batch_truth import train_batched
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--set", required=True)
ap.add_argument("--target", type=int, required=True); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--tag", default="")
a = ap.parse_args(); dev = torch.device("cuda")
D = load(a.ds); D = {**D, "Ytr": D["Ytr"][:, [a.target]], "Yte": D["Yte"][:, [a.target]]}
M = np.load(REPO / f"DNN_Aggresvation100/outputs/sets/{a.ds}_{a.set}.npy").astype(np.float32)
idx = np.arange(len(M))[a.shard::a.nshard]
if a.limit: idx = np.random.RandomState(7).permutation(idx)[:a.limit]
f = ROOT / f"outputs/truth/{a.ds}_{a.set}{a.tag}_single_t{a.target}_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)
t0 = time.time(); cl, vl = [], []
for s in range(0, len(idx), a.chunk):
    R = train_batched(M[idx[s:s + a.chunk]], D, seed=a.seed * 100000 + int(idx[s]), device=dev)
    cl.append(R["clean"]); vl.append(R["val_r2"])
np.savez(f, idx=idx, clean=np.concatenate(cl), val_r2=np.concatenate(vl))
log("TRUTH1", "DONE", f"{a.ds}_{a.set}{a.tag} single t{a.target} sh{a.shard}/{a.nshard} n={len(idx)} {time.time()-t0:.0f}s", sec=time.time() - t0)
print(f"done t{a.target} n={len(idx)} {time.time()-t0:.0f}s", flush=True)
