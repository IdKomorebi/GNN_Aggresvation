# -*- coding: utf-8 -*-
"""单目标通用模型估计：Z=[φ_y(x,m), X_S, X_S²] 闭式 ridge（α 在 train 内部 fit/val 选），报告 test R²。
用法：run_est_single.py --ds pjm --set k3 --target 3 --shard 0 --nshard 3 [--seed 0]"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation91/src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load
from featridge import FrozenPhi, ridge_r2, fit_val_idx
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--set", required=True)
ap.add_argument("--target", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--nophi", action="store_true")
a = ap.parse_args(); dev = torch.device("cuda")
D = load(a.ds); nG = len(D["general"])
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"][:, [a.target]], device=dev)
Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"][:, [a.target]], device=dev)
fit_idx, val_idx = fit_val_idx(len(Xtr), dev)
sys.path.insert(0, str(REPO / "DNN_Aggresvation69"))
from src.oracle import MLPOracle
ck = torch.load(ROOT / f"outputs/oracle/{a.ds}_single_t{a.target}_seed{a.seed}.pt", map_location=dev, weights_only=False)
mdl = MLPOracle(nG, 1).to(dev); mdl.load_state_dict(ck["state"]); phi = FrozenPhi(mdl.eval(), "last")
M = np.load(REPO / f"DNN_Aggresvation100/outputs/sets/{a.ds}_{a.set}.npy")
idx = np.arange(len(M))[a.shard::a.nshard]
chunk = max(4, int(6e8 / (len(Xtr) * 338 * 8)))
tag = "single_nophi" if a.nophi else "single"
f = ROOT / f"outputs/est/{a.ds}_{a.set}_{tag}_t{a.target}_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)
t0 = time.time(); outs = []
with torch.no_grad():
    for s in range(0, len(idx), chunk):
        m = torch.as_tensor(M[idx[s:s + chunk]], device=dev, dtype=torch.float32)
        xm = Xtr.unsqueeze(0) * m.unsqueeze(1); xe = Xte.unsqueeze(0) * m.unsqueeze(1)
        Ztr = torch.cat([xm, xm ** 2] + ([] if a.nophi else [phi(Xtr, m)]), 2).double()
        Zte = torch.cat([xe, xe ** 2] + ([] if a.nophi else [phi(Xte, m)]), 2).double()
        outs.append(ridge_r2(Ztr, Ytr.double(), Zte, Yte.double(), fit_idx, val_idx).float().cpu().numpy())
V = np.concatenate(outs); np.savez(f, idx=idx, v=V); ms = (time.time() - t0) / len(idx) * 1e3
log("EST1", "DONE", f"{a.ds}_{a.set} {tag} t{a.target} sh{a.shard}/{a.nshard} n={len(idx)} {ms:.2f} ms/集合", ms_per_set=ms)
print(f"done t{a.target} {ms:.2f} ms/set", flush=True)
