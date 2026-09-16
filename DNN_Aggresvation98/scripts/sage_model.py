# -*- coding: utf-8 -*-
"""GPT 实验 4：全量模型的"模型依赖型"全局 Shapley（SAGE，边际填补）vs 总体推断能力 Shapley。

对博弈 A/B：在 p 个玩家上训练一个全量 DNN（与真值同结构，val 早停），
v_f(S) = 用 f 预测、S 外字段以随机训练行替换（K=8 次平均预测）后的测试 R²，穷举全部 2^p 联盟，
再取精确 Shapley。三个训练种子，观察模型依赖随训练随机性的漂移。
另给置换重要性（全量模型上逐字段打乱）作为最常见的模型依赖基线。
"""
import sys, argparse, time
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_pjm, GAME_A, game_b
from runlog import log
ap = argparse.ArgumentParser(); ap.add_argument("--game", default="gameA"); ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args(); dev = torch.device("cuda")
D = load_pjm(); g = D["general"]
P = [g.index(f) for f in (GAME_A if a.game == "gameA" else game_b(g))]; p = len(P)
Xtr = torch.as_tensor(D["Xtr"][:, P], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
Xte = torch.as_tensor(D["Xte"][:, P], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev)
fi, vi = torch.as_tensor(D["fit_idx"], device=dev), torch.as_tensor(D["val_idx"], device=dev)
sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
out = {}
for seed in map(int, a.seeds.split(",")):
    torch.manual_seed(seed); t0 = time.time()
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
    K = 8; gen = torch.Generator(device=dev).manual_seed(100 + seed)
    bg = Xtr[fi][torch.randint(len(fi), (K, len(Xte)), generator=gen, device=dev)]   # (K,n,p)
    b = torch.arange(2 ** p, device=dev); bits = ((b[:, None] >> torch.arange(p, device=dev)) & 1).float()
    V = torch.zeros(2 ** p, 12, device=dev)
    with torch.no_grad():
        for s in range(0, 2 ** p, 64):
            m = bits[s:s + 64][:, None, None, :]                                  # (B,1,1,p)
            xin = m * Xte[None, None] + (1 - m) * bg[None]                        # (B,K,n,p)
            pred = f(xin).mean(1)                                                 # (B,n,12)
            V[s:s + 64] = 1 - ((pred - Yte) ** 2).sum(1) / sst
        pi = []
        full = 1 - ((f(Xte) - Yte) ** 2).sum(0) / sst
        for j in range(p):
            xs = Xte.clone(); xs[:, j] = xs[torch.randperm(len(xs), device=dev), j]
            pi.append((full - (1 - ((f(xs) - Yte) ** 2).sum(0) / sst)).cpu().numpy())
    out[f"V_seed{seed}"] = V.cpu().numpy(); out[f"perm_seed{seed}"] = np.stack(pi)
    log("SAGE", "DONE", f"{a.game} seed{seed} ep={ep+1} fullR2={float(full.mean()):.4f} {time.time()-t0:.0f}s")
np.savez(ROOT / f"outputs/sage_{a.game}.npz", **out)
