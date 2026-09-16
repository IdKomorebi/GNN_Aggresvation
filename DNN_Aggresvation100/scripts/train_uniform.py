# -*- coding: utf-8 -*-
"""uniform（两段式）掩码 MLP 通用模型训练。协议逐项对齐 75/95 号（EPOCHS 400 / PATIENCE 60 / BATCH 256 /
Adam 1e-3 wd 5e-4 / train 内按 seed 置换取 15% 验证 / 8 组固定验证掩码 / 69 号 sample_mask）。
--split main 时应与 75 号(PJM)/95 号(CAISO) 同种子权重逐位相同（本脚本首跑时自检）。"""
import sys, time, argparse
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from common100 import load_any, R69
sys.path.insert(0, str(R69))
from src.oracle import MLPOracle, sample_mask
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="main"); ap.add_argument("--outroot", default=""); ap.add_argument("--rep", type=int, default=0); ap.add_argument("--half", type=int, default=0)
a = ap.parse_args(); dev = torch.device("cuda")
torch.manual_seed(42); np.random.seed(42)
D = load_any(a.ds, a.split, a.rep, a.half)
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev); nG, nC = Xtr.shape[1], Ytr.shape[1]
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
tag = "main" if a.split == "main" else f"rep{a.rep}h{a.half}"
OUTR = Path(a.outroot) if a.outroot else ROOT
out = OUTR / f"outputs/oracle_{a.ds}_{tag}_uniform_seed{a.seed}.pt"
torch.save({"state": bst, "val": best, "epochs_run": ep + 1, "train_sec": time.time() - t0}, out)
msg = f"{a.ds} {tag} seed{a.seed} ep={ep+1} val={best:.4f} {time.time()-t0:.0f}s"
if a.split == "main":
    ref = {"pjm": REPO / f"DNN_Aggresvation75/outputs/oracle_uniform_seed{a.seed}.pt",
           "caiso": REPO / f"DNN_Aggresvation95_caiso/outputs/oracle_uniform_seed{a.seed}.pt"}[a.ds]
    if ref.exists():
        rs = torch.load(ref, map_location=dev, weights_only=False)["state"]
        msg += f" | 与历史权重最大差 {max(float((rs[k]-bst[k]).abs().max()) for k in rs):.2e}"
log("ORACLE", "DONE", msg, train_sec=time.time() - t0); print(msg)
