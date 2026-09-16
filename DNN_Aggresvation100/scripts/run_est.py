# -*- coding: utf-8 -*-
"""通用模型估计：对集合文件逐集合给出 12 目标的测试 R²（ridge 读出的 α 在 train 内部 fit/val 选）。
模型：direct（共享头）、L0（冻结 uniform 主干 + 闭式读出）、L0ensx（uniform 三种子特征 ⊕ x ⊕ x²，主口径）、L1x（仅 PJM main：97 号 r93 ⊕ x ⊕ x²）。
用法：run_est.py --ds pjm --set k4 --model L0ensx --shard 0 --nshard 3 [--split half --rep 0 --half 0]"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation91/src")); sys.path.insert(0, str(ROOT / "src"))  # 本号 src 在最前：避免导入 91 号同名 runlog
from common100 import load_any
from featridge import FrozenPhi, ridge_r2, fit_val_idx, load_oracle
from runlog import log
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--set", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--split", default="main"); ap.add_argument("--rep", type=int, default=0); ap.add_argument("--half", type=int, default=0)
ap.add_argument("--setfile", default=""); ap.add_argument("--outroot", default="")
a = ap.parse_args(); dev = torch.device("cuda")
OUTR = Path(a.outroot) if a.outroot else ROOT


def ck(tag):
    if a.split != "main":
        return OUTR / f"outputs/oracle_{a.ds}_rep{a.rep}h{a.half}_uniform_seed{tag[-1]}.pt"
    if tag == "r93":
        return REPO / "DNN_Aggresvation97/outputs/oracle_l1_r93_seed0.pt"
    s = int(tag[-1])
    if a.ds == "pjm": return REPO / f"DNN_Aggresvation75/outputs/oracle_uniform_seed{s}.pt"
    return REPO / "DNN_Aggresvation95_caiso/outputs/oracle_uniform_seed0.pt" if s == 0 else ROOT / f"outputs/oracle_caiso_main_uniform_seed{s}.pt"


D = load_any(a.ds, a.split, a.rep, a.half); nG, nC = len(D["general"]), len(D["conf"])
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev)
fit_idx, val_idx = fit_val_idx(len(Xtr), dev)
phis = {"L0": ["u0"], "L0ensx": ["u0", "u1", "u2"], "L1x": ["r93"]}.get(a.model, [])
phis = [FrozenPhi(load_oracle(ck(t), nG, nC, dev), "last") for t in phis]
direct = load_oracle(ck("u0"), nG, nC, dev) if a.model == "direct" else None
rawx = a.model in ("L0ensx", "L1x")
dim = {"L0": 256, "L0ensx": 856, "L1x": 344}.get(a.model, 1)
chunk = max(4, int(6e8 / (len(Xtr) * dim * 8)))
setfile = a.setfile or str(ROOT / f"outputs/sets/{a.ds}_{a.set}.npy")
M = np.load(setfile); idx = np.arange(len(M))[a.shard::a.nshard]


def feats(X, m):
    xm = X.unsqueeze(0) * m.unsqueeze(1); blocks = [xm, xm ** 2] if rawx else []
    return torch.cat(blocks + [phi(X, m) for phi in phis], 2)


@torch.no_grad()
def run(m):
    if direct is not None:
        sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
        return torch.stack([1 - ((direct(Xte, m[b:b + 1].expand(len(Xte), -1)) - Yte) ** 2).sum(0) / sst for b in range(len(m))])
    return ridge_r2(feats(Xtr, m).double(), Ytr.double(), feats(Xte, m).double(), Yte.double(), fit_idx, val_idx)


tag = a.set if a.split == "main" else f"{a.set}_rep{a.rep}h{a.half}"
f = OUTR / f"outputs/est/{a.ds}_{tag}_{a.model}_s{a.shard}of{a.nshard}.npz"
if f.exists(): print("exists", f.name); sys.exit(0)
t0 = time.time(); outs = []
for s in range(0, len(idx), chunk):
    m = torch.as_tensor(M[idx[s:s + chunk]], device=dev, dtype=torch.float32); outs.append(run(m).float().cpu().numpy())
    if (s // chunk) % 200 == 0: print(f"[{a.ds}_{tag} {a.model} sh{a.shard}] {s+len(m)}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
np.savez(f, idx=idx, v=np.concatenate(outs)); ms = (time.time() - t0) / len(idx) * 1e3
log("EST", "DONE", f"{a.ds}_{tag} {a.model} sh{a.shard}/{a.nshard} n={len(idx)} {ms:.2f} ms/集合", ms_per_set=ms); print(f"done {ms:.2f} ms/set")
