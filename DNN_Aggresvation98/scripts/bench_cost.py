# -*- coding: utf-8 -*-
"""E：独占单卡计时 + 大集合真值充分性核对。

1) 并行专用重训：B=256/1024/2048 个随机规模集合，秒/集合；
2) 逐个专用重训（历史做法：一个集合一个 DNN，300 epoch，每 epoch 验证）：秒/集合；
3) 60 号 50 个宽集合（规模 3–43）：本引擎 leak69 口径 vs 69 号历史 DNN 真值（400 epoch/patience120）。
估计器的 ms/集合由 eval_models.py --limit 512 在独占卡上另测。
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_pjm
from batch_truth import train_batched
from runlog import log
D = load_pjm(); dev = torch.device("cuda"); g = D["general"]; rng = np.random.RandomState(5)
res = {}
for B in [256, 1024, 2048]:
    M = np.zeros((B, 44), np.float32)
    for b in range(B): M[b, rng.choice(44, rng.randint(1, 45), replace=False)] = 1
    torch.cuda.synchronize(); t = time.time(); train_batched(M, D, 0, dev); torch.cuda.synchronize()
    res[f"batched_B{B}_s_per_set"] = (time.time() - t) / B
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
fi = torch.as_tensor(D["fit_idx"], device=dev); vi = torch.as_tensor(D["val_idx"], device=dev)
ts = []
for rep in range(3):
    sel = rng.choice(44, 22, replace=False); X = Xtr[:, sel]
    f = nn.Sequential(nn.Linear(22, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 12)).to(dev)
    opt = torch.optim.Adam(f.parameters(), 1e-3, weight_decay=5e-4); torch.cuda.synchronize(); t = time.time()
    for ep in range(300):
        f.train(); o = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(o), 128):
            ix = o[s:s + 128]; opt.zero_grad(); ((f(X[ix]) - Ytr[ix]) ** 2).mean().backward(); opt.step()
        f.eval()
        with torch.no_grad(): float(((f(X[vi]) - Ytr[vi]) ** 2).mean())
    torch.cuda.synchronize(); ts.append(time.time() - t)
res["sequential_s_per_set"] = float(np.mean(ts))
tl = pd.read_csv(ROOT.parent / "DNN_Aggresvation69/outputs/truth_long.csv")
subs = json.load(open(ROOT.parent / "DNN_Aggresvation69/outputs/subsets.json"))
w = tl[tl.source.str.contains("60")].drop_duplicates(["canonical_sid", "conf"])
sids = list(w.canonical_sid.unique()); M = np.zeros((len(sids), 44), np.float32)
for b, s in enumerate(sids): M[b, [g.index(x) for x in subs[s]["fields"]]] = 1
R = train_batched(M, D, 0, dev)
H = w.pivot(index="canonical_sid", columns="conf", values="dnn").loc[sids, D["conf"]].values
res["wide_n"] = len(sids); res["wide_size_range"] = [int(M.sum(1).min()), int(M.sum(1).max())]
for k in ["leak69", "clean"]:
    v = np.clip(R[k], 0, 1); res[f"wide_{k}_minus_hist69"] = float((v - H).mean()); res[f"wide_{k}_MAE_hist69"] = float(np.abs(v - H).mean())
json.dump(res, open(ROOT / "outputs/analysis/E_bench_cost.json", "w"), indent=2)
log("BENCH", "DONE", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False))
print(json.dumps(res, indent=2))
