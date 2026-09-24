# -*- coding: utf-8 -*-
"""116 号：三种主干的训练与新主口径估计（E 读出）。

三种主干（读出方式完全相同，只有主干从哪来不同）：
  本组主干   —— 只在本组候选字段上、以本组目标为输出做随机掩码预训练（= 114/115 号的做法）；
  全列主干   —— 在数据集全部 53 列（41 个去别名后的一般列 + 12 个机密列）上做"掩码重建"预训练：
               随机遮住一部分列，用其余可见列预测被遮住的列（损失只算被遮住的列）。
               审计方持有的全部运行数据都进预训练，之后任何列都可以被指定为目标或候选，只需换读出；
  留目标主干 —— 与全列主干相同，但把本组目标列从数据中整列删除：检验"对预训练时从未见过的目标"是否仍然有效。
数据边界：预训练只用训练行（早停用 train 内部的 val）；读出的 λ 在训练集内部 5 折选；测试行只用于报告 R²。
读出时掩码只打开本组候选列，目标列永远不可见。

新主口径 E（115 号定稿）：[x⊙m, (x⊙m)², φ_s(x⊙m, m)] 上的子集闭式岭读出，特征标准差下限 1e-3、
测试特征截断到训练集取值范围、5 折选 λ；三个种子各自读出，预测取平均。
"""
from __future__ import annotations

import importlib.util
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


orc = _load("orc69", REPO / "DNN_Aggresvation69/src/oracle.py")
fr = _load("fr91", REPO / "DNN_Aggresvation91/src/featridge.py")
ro = _load("ro115", REPO / "DNN_Aggresvation115/src/readout.py")
FIX = dict(sd_floor=1e-3, clip_te=True)


def train_mae(X: np.ndarray, fit_idx, val_idx, seed: int, device, epochs=400, patience=60, bs=256):
    """掩码重建预训练：输入 [x⊙m, m]，输出全部列；损失只算被遮住的列。X 为训练行 (n,p)。"""
    dev = torch.device(device); Xt = torch.as_tensor(X, device=dev); p = X.shape[1]
    fi = torch.as_tensor(fit_idx, device=dev); vi = torch.as_tensor(val_idx, device=dev)
    torch.manual_seed(seed); np.random.seed(seed)
    model = orc.MLPOracle(p, p).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    mrng, vr = np.random.RandomState(1234 + seed), np.random.RandomState(999)
    vms = [torch.as_tensor(orc.sample_mask(len(vi), p, vr), device=dev) for _ in range(8)]

    def loss(x, m):
        h = 1 - m
        return (((model(x, m) - x) ** 2) * h).sum() / h.sum().clamp_min(1.0)

    best, bst, pat, t0 = 1e9, None, 0, time.time()
    for ep in range(epochs):
        model.train(); order = fi[torch.randperm(len(fi), device=dev)]
        for b in range(0, len(order), bs):
            ix = order[b:b + bs]; m = torch.as_tensor(orc.sample_mask(len(ix), p, mrng), device=dev)
            opt.zero_grad(); loss(Xt[ix], m).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss(Xt[vi], mm).item() for mm in vms]))
        if v < best - 1e-6:
            best, bst, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(bst)
    return model.eval(), dict(val=best, epochs=ep + 1, sec=time.time() - t0)


def load_models(paths, n_in, n_out, device):
    out = []
    for f in paths:
        m = orc.MLPOracle(n_in, n_out).to(device)
        ck = torch.load(f, map_location=device, weights_only=False)
        m.load_state_dict(ck["state"] if isinstance(ck, dict) and "state" in ck else ck); out.append(m.eval())
    return out


@torch.no_grad()
def estimate_E(Xtr, Ytr, Xte, Yte, models, masks, col_map, device, B=8, seeds_avg=True, raw=True, log=None):
    """masks:(N,p_cand) 本组候选上的掩码；col_map：候选下标 → 主干输入列下标；Xtr/Xte 为主干输入空间的矩阵。
    返回 (N,C) 测试 R²（E：三种子预测平均）以及单种子（第一个种子）的 R²。"""
    dev = torch.device(device)
    Xtr = torch.as_tensor(Xtr, device=dev); Xte = torch.as_tensor(Xte, device=dev)
    Ytr = torch.as_tensor(Ytr, device=dev, dtype=torch.float64); Yte = torch.as_tensor(Yte, device=dev, dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
    phis = [fr.FrozenPhi(m, "last") for m in models]
    width = Xtr.shape[1]; col_map = list(col_map)
    Mfull = np.zeros((len(masks), width), np.float32); Mfull[:, col_map] = masks
    Mfull = torch.as_tensor(Mfull, device=dev)
    outE, out1, t0 = [], [], time.time()
    for s in range(0, len(Mfull), B):
        m = Mfull[s:s + B]; preds = []
        for ph in phis:
            def feats(X):
                blocks = []
                if raw:
                    xm = (X.unsqueeze(0) * m.unsqueeze(1))[:, :, col_map]
                    blocks += [xm, xm ** 2]
                return torch.cat(blocks + [ph(X, m)], 2).double()
            preds.append(ro.ridge_predict(feats(Xtr), Ytr, feats(Xte), fi, vi, cv="kfold", folds=folds, **FIX))
        outE.append(ro.r2_from_pred(sum(preds) / len(preds), Yte)); out1.append(ro.r2_from_pred(preds[0], Yte))
        if log and s % (B * 100) == 0:
            log(f"{s + len(m)}/{len(Mfull)}  {time.time() - t0:.0f}s")
    E = torch.cat(outE).cpu().numpy().astype(np.float32); E1 = torch.cat(out1).cpu().numpy().astype(np.float32)
    return E, E1, (time.time() - t0) / len(Mfull)


def full_matrix(ds, drop=()):
    """数据集全部 53 列（去别名后的一般列 + 机密列），训练/测试行与各组 D.npz 完全一致（与 run_bb.full_matrix 相同）。"""
    import sys as _s
    _s.path.insert(0, str(REPO / "DNN_Aggresvation100" / "src"))
    from common100 import load
    D0 = load(ds); gen, conf = D0["general"], D0["conf"]
    names = [gen[i] for i in D0["active"]] + conf
    Xtr = np.concatenate([D0["Xtr"][:, D0["active"]], D0["Ytr"]], 1).astype(np.float32)
    Xte = np.concatenate([D0["Xte"][:, D0["active"]], D0["Yte"]], 1).astype(np.float32)
    keep = [k for k, c in enumerate(names) if c not in drop]
    return Xtr[:, keep], Xte[:, keep], [names[k] for k in keep], D0["fit_idx"], D0["val_idx"]
