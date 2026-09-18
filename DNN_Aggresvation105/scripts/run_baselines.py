# -*- coding: utf-8 -*-
"""RQ-B1 领域/归因 baseline：Pearson、Spearman、NMI、单字段 R²(M^0)、全量模型 SAGE、全量模型置换重要性。
任务：用各字段分数检测"τ-critical 字段"（正式真值定义，K=2、τ=0.7），比较 PR-AUC / recall / false-safe。
SAGE：多目标全量 DNN（协议同攻击器族的多目标 DNN）+ 边际填补 + 置换采样 Shapley（200 次置换）。
"""
import sys, time, argparse
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation100/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--nperm", type=int, default=200)
a = ap.parse_args(); dev = torch.device("cuda")
D = load(a.ds); act = D["active"]; gen, conf = D["general"], D["conf"]; p = len(act)
Xtr, Ytr, Xte, Yte = D["Xtr"][:, act], D["Ytr"], D["Xte"][:, act], D["Yte"]
rows = []
for c in range(12):
    for j, i in enumerate(act):
        pe = abs(pearsonr(Xtr[:, j], Ytr[:, c])[0]); sp = abs(spearmanr(Xtr[:, j], Ytr[:, c])[0])
        rows.append(dict(数据集=a.ds, 目标=conf[c], 字段=gen[i], pearson=pe, spearman=sp))
mi = np.stack([mutual_info_regression(Xtr, Ytr[:, c], random_state=0) for c in range(12)])
R = pd.DataFrame(rows); R["nmi"] = mi.ravel()
# 全量模型：多目标 DNN（与攻击器族口径一致）
Xtr_t = torch.as_tensor(Xtr, device=dev); Ytr_t = torch.as_tensor(Ytr, device=dev)
Xte_t = torch.as_tensor(Xte, device=dev); Yte_t = torch.as_tensor(Yte, device=dev)
fi = torch.as_tensor(D["fit_idx"], device=dev); vi = torch.as_tensor(D["val_idx"], device=dev)
sst = ((Yte_t - Yte_t.mean(0)) ** 2).sum(0)
torch.manual_seed(0)
f = nn.Sequential(nn.Linear(p, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 12)).to(dev)
opt = torch.optim.Adam(f.parameters(), 1e-3, weight_decay=5e-4); best, bs, pat = 1e9, None, 0
for ep in range(400):
    f.train(); o = fi[torch.randperm(len(fi), device=dev)]
    for s in range(0, len(o), 128):
        ix = o[s:s + 128]; opt.zero_grad(); ((f(Xtr_t[ix]) - Ytr_t[ix]) ** 2).mean().backward(); opt.step()
    f.eval()
    with torch.no_grad(): v = float(((f(Xtr_t[vi]) - Ytr_t[vi]) ** 2).mean())
    if v < best: best, bs, pat = v, deepcopy(f.state_dict()), 0
    else:
        pat += 1
        if pat >= 60: break
f.load_state_dict(bs); f.eval()
with torch.no_grad(): full_r2 = (1 - ((f(Xte_t) - Yte_t) ** 2).sum(0) / sst).cpu().numpy()
log("BASE", "DONE", f"{a.ds} 全量模型 ep={ep+1} 平均R²={full_r2.mean():.4f}")
# SAGE（边际填补 + 置换采样）与置换重要性
gen_t = torch.Generator(device=dev).manual_seed(105)
bg = Xtr_t[fi][torch.randint(len(fi), (4, len(Xte_t)), generator=gen_t, device=dev)]   # 4 次填补


@torch.no_grad()
def v_of(mask):
    m = torch.as_tensor(mask, device=dev, dtype=torch.float32)[None, None, :]
    pred = f(m * Xte_t[None] + (1 - m) * bg).mean(0)
    return (1 - ((pred - Yte_t) ** 2).sum(0) / sst).cpu().numpy()


t0 = time.time(); sage = np.zeros((p, 12)); rng = np.random.RandomState(5)
for it in range(a.nperm):
    order = rng.permutation(p); mask = np.zeros(p, np.float32); prev = v_of(mask)
    for j in order:
        mask[j] = 1; cur = v_of(mask); sage[j] += cur - prev; prev = cur
    if it % 50 == 0: print(f"[sage {a.ds}] {it}/{a.nperm} {time.time()-t0:.0f}s", flush=True)
sage /= a.nperm
perm_imp = np.zeros((p, 12))
with torch.no_grad():
    for j in range(p):
        xs = Xte_t.clone(); xs[:, j] = xs[torch.randperm(len(xs), device=dev), j]
        perm_imp[j] = (full_r2 - (1 - ((f(xs) - Yte_t) ** 2).sum(0) / sst).cpu().numpy())
R["sage"] = sage.T.ravel(); R["perm_importance"] = perm_imp.T.ravel()
R.to_csv(ROOT / f"outputs/analysis/{a.ds}_field_baselines.csv", index=False)
log("BASE", "DONE", f"{a.ds} SAGE+置换重要性完成 {time.time()-t0:.0f}s")
print("saved", len(R), "rows", flush=True)
