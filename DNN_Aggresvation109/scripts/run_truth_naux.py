# -*- coding: utf-8 -*-
"""109 号：攻击者辅助标签量受限时的并行专用重训真值。
引擎同 98 号 batch_truth（val 选轮次、测试集只报告），攻击器为多目标 DNN（单一攻击器口径，
比例之间可比；与 103 号三攻击器正式真值口径不同，分析时会单独标注）。
用法：run_truth_naux.py --ds pjm --frac 0.25 --shard 0 --nshard 3"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation98/src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src"))
sys.path.insert(0, str(ROOT / "src"))                      # 本号 src 最前：runlog 与 naux
from common100 import load
from batch_truth import train_batched
from naux import subsample
from runlog import log
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--frac", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args(); dev = torch.device("cuda")
D = subsample(load(a.ds), a.frac)
M = np.load(REPO / f"DNN_Aggresvation100/outputs/sets/{a.ds}_k3.npy").astype(np.float32)
idx = np.arange(len(M))[a.shard::a.nshard]
tag = f"{a.ds}_f{int(a.frac*100):03d}"
f = ROOT / f"outputs/truth/{tag}_seed{a.seed}_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)
log("TRUTH", "START", f"{tag} n_aux={D['n_aux']} shard{a.shard}/{a.nshard} n={len(idx)}")
t0 = time.time(); acc = {k: [] for k in ["clean", "val_r2"]}
for s in range(0, len(idx), a.chunk):
    m = M[idx[s:s + a.chunk]]
    R = train_batched(m, D, seed=a.seed * 100000 + int(idx[s]), device=dev)
    acc["clean"].append(R["clean"]); acc["val_r2"].append(R["val_r2"])
    print(f"[{tag} sh{a.shard}] {s+len(m)}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
np.savez(f, idx=idx, n_aux=D["n_aux"], **{k: np.concatenate(v) for k, v in acc.items()})
log("TRUTH", "DONE", f"{tag} n_aux={D['n_aux']} shard{a.shard}/{a.nshard} {time.time()-t0:.0f}s", sec=time.time() - t0)
