# -*- coding: utf-8 -*-
"""多目标 backbone 对照：Z=[φ_multi(x,m), X_S, X_S²] 闭式 ridge，一次给 12 个目标（与单目标版公平对照，同为单种子 φ）。
用法：run_est_multi.py --ds pjm --set k3 --shard 0 --nshard 3"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation91/src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load
from featridge import FrozenPhi, ridge_r2, fit_val_idx, load_oracle
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--set", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args(); dev = torch.device("cuda")
D = load(a.ds); nG, nC = len(D["general"]), len(D["conf"])
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev)
fit_idx, val_idx = fit_val_idx(len(Xtr), dev)
ck = (REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt") if a.ds == "pjm" else (REPO / "DNN_Aggresvation95_caiso/outputs/oracle_uniform_seed0.pt")
phi = FrozenPhi(load_oracle(ck, nG, nC, dev), "last")
M = np.load(REPO / f"DNN_Aggresvation100/outputs/sets/{a.ds}_{a.set}.npy")
idx = np.arange(len(M))[a.shard::a.nshard]
chunk = max(4, int(6e8 / (len(Xtr) * 338 * 8)))
f = ROOT / f"outputs/est/{a.ds}_{a.set}_multix_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)
t0 = time.time(); outs = []
with torch.no_grad():
    for s in range(0, len(idx), chunk):
        m = torch.as_tensor(M[idx[s:s + chunk]], device=dev, dtype=torch.float32)
        xm = Xtr.unsqueeze(0) * m.unsqueeze(1); xe = Xte.unsqueeze(0) * m.unsqueeze(1)
        Ztr = torch.cat([xm, xm ** 2, phi(Xtr, m)], 2).double(); Zte = torch.cat([xe, xe ** 2, phi(Xte, m)], 2).double()
        outs.append(ridge_r2(Ztr, Ytr.double(), Zte, Yte.double(), fit_idx, val_idx).float().cpu().numpy())
np.savez(f, idx=idx, v=np.concatenate(outs)); ms = (time.time() - t0) / len(idx) * 1e3
log("ESTM", "DONE", f"{a.ds}_{a.set} multix sh{a.shard}/{a.nshard} n={len(idx)} {ms:.2f} ms/集合", ms_per_set=ms)
print(f"done multix {ms:.2f} ms/set", flush=True)
