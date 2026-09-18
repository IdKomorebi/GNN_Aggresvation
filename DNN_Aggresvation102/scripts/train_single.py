# -*- coding: utf-8 -*-
"""P0-2：单目标（target-specific）随机掩码 backbone 预学习。
协议同 FINAL_PROTOCOL §8（EPOCHS 400 / PATIENCE 60 / BATCH 256 / Adam 1e-3 wd 5e-4 / train 内 15% val / 8 组固定验证掩码 / uniform 两段式掩码）。
与多目标版本唯一差别：输出维度 1（只预测一个敏感目标）。
用法：train_single.py --ds pjm --target 3 --seed 0"""
import sys, time, argparse
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load, R69
sys.path.insert(0, str(R69))
from src.oracle import MLPOracle, sample_mask
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--target", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = torch.device("cuda")
torch.manual_seed(42); np.random.seed(42)
D = load(a.ds)
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"][:, [a.target]], device=dev)
nG, nC = Xtr.shape[1], 1
n = len(Xtr); nv = max(int(n * 0.15), 1); perm = np.random.RandomState(a.seed).permutation(n)
va, tr = torch.as_tensor(perm[:nv], device=dev), torch.as_tensor(perm[nv:], device=dev)
torch.manual_seed(a.seed); np.random.seed(a.seed)
model = MLPOracle(nG, nC).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
mrng = np.random.RandomState(1234 + a.seed); vr = np.random.RandomState(999)
vms = [torch.as_tensor(sample_mask(nv, nG, vr), device=dev) for _ in range(8)]
best, bst, pat, t0 = 1e9, None, 0, time.time()
for ep in range(400):
    model.train(); order = tr[torch.randperm(len(tr), device=dev)]
    for b in range(0, len(tr), 256):
        ix = order[b:b + 256]; m = torch.as_tensor(sample_mask(len(ix), nG, mrng), device=dev)
        opt.zero_grad(); ((model(Xtr[ix], m) - Ytr[ix]) ** 2).mean().backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = float(np.mean([((model(Xtr[va], mm) - Ytr[va]) ** 2).mean().item() for mm in vms]))
    if v < best - 1e-6: best, bst, pat = v, deepcopy(model.state_dict()), 0
    else:
        pat += 1
        if pat >= 60: break
out = ROOT / f"outputs/oracle/{a.ds}_single_t{a.target}_seed{a.seed}.pt"
torch.save({"state": bst, "val": best, "epochs_run": ep + 1, "train_sec": time.time() - t0, "target": a.target}, out)
msg = f"{a.ds} single t{a.target} seed{a.seed} ep={ep+1} val={best:.5f} {time.time()-t0:.0f}s"
log("SINGLE", "DONE", msg, train_sec=time.time() - t0); print(msg, flush=True)
