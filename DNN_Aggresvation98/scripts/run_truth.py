# -*- coding: utf-8 -*-
"""分片跑并行专用重训真值。用法：run_truth.py --set gameA --seed 0 --shard 0 --nshard 3"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_pjm
from batch_truth import train_batched, ridge_truth
from runlog import log
ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args()
D = load_pjm(); dev = torch.device("cuda")
M = np.load(ROOT / f"outputs/masks/{a.set}.npy")
idx = np.arange(len(M))[a.shard::a.nshard]
od = ROOT / "outputs/truth"; od.mkdir(parents=True, exist_ok=True)
f = od / f"{a.set}_seed{a.seed}_s{a.shard}of{a.nshard}.npz"
log("TRUTH", "START", f"{a.set} seed{a.seed} shard{a.shard}/{a.nshard} n={len(idx)}")
t0 = time.time(); acc = {k: [] for k in ["clean", "leak69", "oracle_best", "val_r2", "ridge"]}
for s in range(0, len(idx), a.chunk):
    ii = idx[s:s + a.chunk]; m = M[ii]
    m_nz = m.copy(); empty = m.sum(1) == 0; m_nz[empty, 0] = 1        # 空集占位，结果强制为 0
    R = train_batched(m_nz, D, seed=a.seed * 1000 + s, device=dev)
    for k in ["clean", "leak69", "oracle_best", "val_r2"]:
        v = R[k]; v[empty] = 0.0; acc[k].append(v)
    L = ridge_truth(m_nz, D, dev); L[empty] = 0.0; acc["ridge"].append(L)
    print(f"[{a.set} s{a.seed} sh{a.shard}] {s+len(ii)}/{len(idx)} ep={R['epochs']} {time.time()-t0:.0f}s", flush=True)
np.savez(f, idx=idx, **{k: np.concatenate(v) for k, v in acc.items()})
log("TRUTH", "DONE", f"{a.set} seed{a.seed} shard{a.shard}/{a.nshard} n={len(idx)} {time.time()-t0:.0f}s -> {f.name}")
