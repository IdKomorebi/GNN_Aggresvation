# -*- coding: utf-8 -*-
"""并行专用重训真值（引擎同 98 号 src/batch_truth.py，val 选轮次、测试集只报告）。
用法：run_truth.py --ds pjm --set k4 --seed 0 --shard 0 --nshard 3"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "DNN_Aggresvation98/src")); sys.path.insert(0, str(ROOT / "src"))  # 本号 src 在最前：避免导入 98 号同名 runlog
from common100 import load
from batch_truth import train_batched, ridge_truth
from runlog import log
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--set", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args(); dev = torch.device("cuda")
D = load(a.ds); M = np.load(ROOT / f"outputs/sets/{a.ds}_{a.set}.npy").astype(np.float32)
idx = np.arange(len(M))[a.shard::a.nshard]
f = ROOT / f"outputs/truth/{a.ds}_{a.set}_seed{a.seed}_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)
log("TRUTH", "START", f"{a.ds}_{a.set} seed{a.seed} shard{a.shard}/{a.nshard} n={len(idx)}")
t0 = time.time(); acc = {k: [] for k in ["clean", "val_r2", "ridge"]}
for s in range(0, len(idx), a.chunk):
    m = M[idx[s:s + a.chunk]]
    R = train_batched(m, D, seed=a.seed * 100000 + int(idx[s]), device=dev)
    acc["clean"].append(R["clean"]); acc["val_r2"].append(R["val_r2"]); acc["ridge"].append(ridge_truth(m, D, dev))
    print(f"[{a.ds}_{a.set} s{a.seed} sh{a.shard}] {s+len(m)}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
np.savez(f, idx=idx, **{k: np.concatenate(v) for k, v in acc.items()})
log("TRUTH", "DONE", f"{a.ds}_{a.set} seed{a.seed} shard{a.shard}/{a.nshard} n={len(idx)} {time.time()-t0:.0f}s", sec=time.time() - t0)
