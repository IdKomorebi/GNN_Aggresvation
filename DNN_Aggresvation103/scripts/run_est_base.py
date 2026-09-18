# -*- coding: utf-8 -*-
"""RQ-A2 基线估计器（不含学出的 φ）：raw ridge / raw+x² ridge / 随机傅里叶特征 ridge（可选含 raw）。
α 在 train 内部 fit/val 选，test 只报告。用法：run_est_base.py --ds pjm --feat raw --shard 0 --nshard 3"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation91/src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load
from featridge import ridge_r2, fit_val_idx
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--set", default="k3")
ap.add_argument("--feat", required=True, choices=["raw", "raw2", "rff", "rffx"])
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--dim", type=int, default=768)
a = ap.parse_args(); dev = torch.device("cuda")
D = load(a.ds); nG = len(D["general"])
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev).double()
Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev).double()
fit_idx, val_idx = fit_val_idx(len(Xtr), dev)
gen = torch.Generator(device=dev).manual_seed(103)
W = torch.randn(a.dim, nG, generator=gen, device=dev); bb = torch.rand(a.dim, generator=gen, device=dev) * 6.2832
M = np.load(REPO / f"DNN_Aggresvation100/outputs/sets/{a.ds}_{a.set}.npy")
idx = np.arange(len(M))[a.shard::a.nshard]
dim = {"raw": nG, "raw2": 2 * nG, "rff": a.dim, "rffx": a.dim + 2 * nG}[a.feat]
chunk = max(4, int(6e8 / (len(Xtr) * dim * 8)))
f = ROOT / f"outputs/est/{a.ds}_{a.set}_{a.feat}_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)


def feats(X, m):
    xm = X.unsqueeze(0) * m.unsqueeze(1); blocks = []
    if a.feat in ("raw", "raw2", "rffx"): blocks.append(xm)
    if a.feat in ("raw2", "rffx"): blocks.append(xm ** 2)
    if a.feat in ("rff", "rffx"):
        sc = 1 / m.sum(1).clamp_min(1).sqrt()
        blocks.append(torch.cos(torch.einsum("bnk,dk->bnd", xm, W) * sc[:, None, None] + bb) * 1.4142)
    return torch.cat(blocks, 2).double()


t0 = time.time(); outs = []
with torch.no_grad():
    for s in range(0, len(idx), chunk):
        m = torch.as_tensor(M[idx[s:s + chunk]], device=dev, dtype=torch.float32)
        outs.append(ridge_r2(feats(Xtr, m), Ytr, feats(Xte, m), Yte, fit_idx, val_idx).float().cpu().numpy())
np.savez(f, idx=idx, v=np.concatenate(outs)); ms = (time.time() - t0) / len(idx) * 1e3
log("ESTBASE", "DONE", f"{a.ds}_{a.set} {a.feat} sh{a.shard}/{a.nshard} n={len(idx)} {ms:.2f} ms/集合", ms_per_set=ms)
print(f"done {a.feat} {ms:.2f} ms/set", flush=True)
