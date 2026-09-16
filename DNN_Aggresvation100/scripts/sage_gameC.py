# -*- coding: utf-8 -*-
"""gameC 全量模型 SAGE（边际填补，K=8 次平均预测）：在 14 个玩家上训练全量 DNN（val 早停），穷举 2^14 联盟，三个训练种子。
与 98 号 scripts/sage_model.py 同一口径，仅数据集与玩家不同。"""
import sys, json, time
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from common100 import load
dev = torch.device("cuda"); D = load("caiso"); P = json.load(open(ROOT / "outputs/sets/caiso_gameC_meta.json"))["idx"]; p = len(P)
Xtr = torch.as_tensor(D["Xtr"][:, P], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
Xte = torch.as_tensor(D["Xte"][:, P], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev)
fi, vi = torch.as_tensor(D["fit_idx"], device=dev), torch.as_tensor(D["val_idx"], device=dev); sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
out = {}
for seed in [0, 1, 2]:
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(p, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 12)).to(dev)
    opt = torch.optim.Adam(f.parameters(), 1e-3, weight_decay=5e-4); best, bs, pat = 1e9, None, 0
    for ep in range(400):
        f.train(); o = fi[torch.randperm(len(fi), device=dev)]
        for s in range(0, len(o), 128):
            ix = o[s:s + 128]; opt.zero_grad(); ((f(Xtr[ix]) - Ytr[ix]) ** 2).mean().backward(); opt.step()
        f.eval()
        with torch.no_grad(): v = float(((f(Xtr[vi]) - Ytr[vi]) ** 2).mean())
        if v < best: best, bs, pat = v, deepcopy(f.state_dict()), 0
        else:
            pat += 1
            if pat >= 60: break
    f.load_state_dict(bs); f.eval()
    gen = torch.Generator(device=dev).manual_seed(100 + seed)
    bg = Xtr[fi][torch.randint(len(fi), (8, len(Xte)), generator=gen, device=dev)]
    bits = ((torch.arange(2 ** p, device=dev)[:, None] >> torch.arange(p, device=dev)) & 1).float()
    V = torch.zeros(2 ** p, 12, device=dev)
    with torch.no_grad():
        for s in range(0, 2 ** p, 16):
            m = bits[s:s + 16][:, None, None, :]
            V[s:s + 16] = 1 - ((f(m * Xte[None, None] + (1 - m) * bg[None]).mean(1) - Yte) ** 2).sum(1) / sst
    out[f"V_seed{seed}"] = V.cpu().numpy(); print("seed", seed, "ep", ep + 1, flush=True)
np.savez(ROOT / "outputs/analysis/sage_caiso_gameC.npz", **out)
