# -*- coding: utf-8 -*-
"""GPT 方案 8.2：给通用 oracle 加单调性软约束，检验是否改善保真度 / Shapley。

协议逐项对齐 75 号 uniform（EPOCHS 400 / PATIENCE 60 / BATCH 256 / Adam 1e-3 wd 5e-4 /
train 内 15% val / 同一掩码采样器），唯一变量是附加损失：

  每步另取 G=8 组、每组 32 行，组内共用掩码 S 与 S+i（i∉S 随机），
  L_mono = mean_g relu( MSE_g(S+i) − MSE_g(S) )
  L = L_value(逐行随机掩码 MSE，与 75 号相同) + λ·L_mono

注意本项目的 oracle 不是 mask→R² 回归器，而是"掩码输入的预测器"，
V(S) 由其在数据上的误差定义，所以单调约束只能写在组级 MSE 上。λ=0 为同脚本对照。
"""
import argparse, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_pjm
sys.path.insert(0, str(REPO / "DNN_Aggresvation69"))
from src.oracle import MLPOracle, sample_mask
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = torch.device("cuda")
D = load_pjm(); nG, nC = 44, 12
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
n = len(Xtr); nv = max(int(n * 0.15), 1); perm = np.random.RandomState(a.seed).permutation(n)
va, tr = torch.as_tensor(perm[:nv], device=dev), torch.as_tensor(perm[nv:], device=dev)
torch.manual_seed(a.seed); np.random.seed(a.seed)
model = MLPOracle(nG, nC).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
mrng = np.random.RandomState(1234 + a.seed); vr = np.random.RandomState(999)
vms = [torch.as_tensor(sample_mask(nv, nG, vr), device=dev) for _ in range(8)]
G, R = 8, 32
best, bst, pat, t0 = 1e9, None, 0, time.time(); viol_hist = []
for ep in range(400):
    model.train(); order = tr[torch.randperm(len(tr), device=dev)]
    for b in range(0, len(tr), 256):
        ix = order[b:b + 256]
        m = torch.as_tensor(sample_mask(len(ix), nG, mrng), device=dev)
        loss = ((model(Xtr[ix], m) - Ytr[ix]) ** 2).mean()
        if a.lam > 0:
            gi = tr[torch.randint(len(tr), (G * R,), device=dev)]
            S = np.zeros((G, nG), np.float32)
            for q in range(G):
                k = mrng.randint(0, nG); S[q, mrng.choice(nG, k, replace=False)] = 1
            Si = S.copy()
            for q in range(G): Si[q, mrng.choice(np.where(S[q] == 0)[0])] = 1
            mS = torch.as_tensor(np.repeat(S, R, 0), device=dev); mSi = torch.as_tensor(np.repeat(Si, R, 0), device=dev)
            eS = ((model(Xtr[gi], mS) - Ytr[gi]) ** 2).mean(1).view(G, R).mean(1)
            eSi = ((model(Xtr[gi], mSi) - Ytr[gi]) ** 2).mean(1).view(G, R).mean(1)
            lm = torch.relu(eSi - eS).mean(); loss = loss + a.lam * lm
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = float(np.mean([((model(Xtr[va], mm) - Ytr[va]) ** 2).mean().item() for mm in vms]))
    if v < best - 1e-6: best, bst, pat = v, deepcopy(model.state_dict()), 0
    else:
        pat += 1
        if pat >= 60: break
out = ROOT / f"outputs/oracle_mono{a.lam:g}_seed{a.seed}.pt"
torch.save({"state": bst, "lam": a.lam, "val": best, "epochs_run": ep + 1}, out)
log("MONO", "DONE", f"lam={a.lam} seed{a.seed} ep={ep+1} val={best:.4f} {time.time()-t0:.0f}s -> {out.name}")
print("done", out.name, best, ep + 1)
