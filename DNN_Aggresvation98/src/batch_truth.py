# -*- coding: utf-8 -*-
"""并行专用重训：一次在 GPU 上训练 B 个**互相独立**的 DNN，每个只看自己的字段集合。

等价性：第 b 个网络的输入是 x⊙m_b。被遮蔽列恒为 0，其第一层权重既不影响输出，
也不向其它参数传梯度，所以函数上等价于"只用 S 的列、从头训练一个 DNN"。
初始化按 |S| 作 fan_in（与 nn.Linear(|S|,H) 相同分布），Adam 逐元素、dropout 逐网络独立，
因此 B 个网络彼此之间没有任何耦合——与逐个重训只差在 minibatch 顺序共用。

结构与 60/67/68 号真值 DNN 相同：Linear(|S|,128)-ReLU-Drop0.15-Linear(128,128)-ReLU-Drop0.15-Linear(128,12)，
Adam lr=1e-3, wd=5e-4, batch 128。

**口径修正（本号要点）**：
  clean  —— 逐 (集合, 目标) 在 val 上选最佳 epoch，报告该 epoch 的测试 R²（测试集不参与任何选择）；
  leak69 —— 复刻 60/67/68 号：按 12 目标平均**测试** MSE 选 epoch（历史真值口径）；
  oracle_best —— 逐目标取测试 R² 最大的 epoch（选择偏差上界）。
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def _uniform(shape, bound, gen, device):
    return (torch.rand(shape, generator=gen, device=device) * 2 - 1) * bound


def train_batched(masks: np.ndarray, D: dict, seed: int, device,
                  hidden: int = 128, epochs: int = 300, patience: int = 60,
                  bs: int = 128, lr: float = 1e-3, wd: float = 5e-4, p_drop: float = 0.15):
    Xtr = torch.as_tensor(D["Xtr"], device=device)
    Ytr = torch.as_tensor(D["Ytr"], device=device)
    Xfit, Yfit = Xtr[D["fit_idx"]], Ytr[D["fit_idx"]]
    Xval, Yval = Xtr[D["val_idx"]], Ytr[D["val_idx"]]
    Xte = torch.as_tensor(D["Xte"], device=device)
    Yte = torch.as_tensor(D["Yte"], device=device)
    M = torch.as_tensor(masks, device=device)
    B, nG = M.shape
    nC = Yfit.shape[1]
    gen = torch.Generator(device=device).manual_seed(seed)

    fan = M.sum(1).clamp_min(1)                                  # |S|
    b1 = (1 / fan.sqrt())[:, None, None]
    W1 = (_uniform((B, hidden, nG), 1, gen, device) * b1).requires_grad_()
    c1 = (_uniform((B, 1, hidden), 1, gen, device) * b1).requires_grad_()
    bh = 1 / math.sqrt(hidden)
    W2 = _uniform((B, hidden, hidden), bh, gen, device).requires_grad_()
    c2 = _uniform((B, 1, hidden), bh, gen, device).requires_grad_()
    W3 = _uniform((B, nC, hidden), bh, gen, device).requires_grad_()
    c3 = _uniform((B, 1, nC), bh, gen, device).requires_grad_()
    params = [W1, c1, W2, c2, W3, c3]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def fwd(X, train):  # X:(n,nG) → (B,n,nC)
        h = X.unsqueeze(0) * M.unsqueeze(1)
        h = torch.relu(torch.baddbmm(c1, h, W1.transpose(1, 2)))
        if train:
            h = F.dropout(h, p_drop, True)
        h = torch.relu(torch.baddbmm(c2, h, W2.transpose(1, 2)))
        if train:
            h = F.dropout(h, p_drop, True)
        return torch.baddbmm(c3, h, W3.transpose(1, 2))

    @torch.no_grad()
    def sse(X, Y, chunk=4096):
        out = torch.zeros(B, nC, device=device)
        for s in range(0, len(X), chunk):
            out += ((fwd(X[s:s + chunk], False) - Y[s:s + chunk].unsqueeze(0)) ** 2).sum(1)
        return out

    sst_val = ((Yval - Yval.mean(0)) ** 2).sum(0)
    sst_te = ((Yte - Yte.mean(0)) ** 2).sum(0)

    best_val = torch.full((B, nC), float("inf"), device=device)
    clean = torch.zeros(B, nC, device=device)
    best_te_mean = torch.full((B,), float("inf"), device=device)
    leak69 = torch.zeros(B, nC, device=device)
    oracle_best = torch.full((B, nC), -float("inf"), device=device)
    best_val_mean = torch.full((B,), float("inf"), device=device)
    stall = torch.zeros(B, dtype=torch.long, device=device)
    perm_gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    n = len(Xfit)
    ep_run = 0
    for ep in range(epochs):
        ep_run = ep + 1
        order = torch.randperm(n, generator=perm_gen).to(device)
        for s in range(0, n, bs):
            ix = order[s:s + bs]
            loss = ((fwd(Xfit[ix], True) - Yfit[ix].unsqueeze(0)) ** 2).mean((1, 2)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sv, st = sse(Xval, Yval), sse(Xte, Yte)
        r2te = 1 - st / sst_te
        upd = sv < best_val
        best_val = torch.where(upd, sv, best_val)
        clean = torch.where(upd, r2te, clean)
        mt = st.mean(1)
        u69 = mt < best_te_mean
        best_te_mean = torch.where(u69, mt, best_te_mean)
        leak69 = torch.where(u69[:, None], r2te, leak69)
        oracle_best = torch.maximum(oracle_best, r2te)
        vm = sv.mean(1)
        improved = vm < best_val_mean - 1e-7
        best_val_mean = torch.where(improved, vm, best_val_mean)
        stall = torch.where(improved, torch.zeros_like(stall), stall + 1)
        if bool((stall >= patience).all()):
            break
    val_r2 = 1 - best_val / sst_val
    return {k: v.detach().cpu().numpy() for k, v in
            dict(clean=clean, leak69=leak69, oracle_best=oracle_best, val_r2=val_r2).items()} | {"epochs": ep_run}


@torch.no_grad()
def ridge_truth(masks: np.ndarray, D: dict, device, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100), chunk=512):
    """线性攻击者：原始 X_S 上的岭回归，α 在 val 上逐 (集合,目标) 选，fit 上拟合、测试报告。"""
    Xtr = torch.as_tensor(D["Xtr"], device=device, dtype=torch.float64)
    Ytr = torch.as_tensor(D["Ytr"], device=device, dtype=torch.float64)
    Xte = torch.as_tensor(D["Xte"], device=device, dtype=torch.float64)
    Yte = torch.as_tensor(D["Yte"], device=device, dtype=torch.float64)
    fi, vi = D["fit_idx"], D["val_idx"]
    Xf, Yf, Xv, Yv = Xtr[fi], Ytr[fi], Xtr[vi], Ytr[vi]
    ym = Yf.mean(0)
    G = Xf.T @ Xf; H = Xf.T @ (Yf - ym)
    Gv = Xv.T @ Xv; Hv = Xv.T @ (Yv - ym); rv = ((Yv - ym) ** 2).sum(0)
    Ge = Xte.T @ Xte; He = Xte.T @ (Yte - ym); re = ((Yte - ym) ** 2).sum(0)
    sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
    nG = Xf.shape[1]
    outs = []
    for s in range(0, len(masks), chunk):
        m = torch.as_tensor(masks[s:s + chunk], device=device, dtype=torch.float64)
        Bm = len(m)
        MM = m[:, :, None] * m[:, None, :]
        eye = torch.eye(nG, device=device, dtype=torch.float64)
        off = (1 - m)[:, :, None] * eye                       # 被遮蔽维度置单位对角，解为 0
        best = torch.full((Bm, Yf.shape[1]), float("inf"), device=device, dtype=torch.float64)
        te_at = torch.zeros_like(best)
        for a in alphas:
            A = G * MM + a * eye * m[:, :, None] + off
            beta = torch.linalg.solve(A, H.unsqueeze(0) * m[:, :, None])   # (B,nG,nC)
            sv = rv - 2 * (beta * Hv).sum(1) + (beta * (Gv @ beta)).sum(1)
            se = re - 2 * (beta * He).sum(1) + (beta * (Ge @ beta)).sum(1)
            upd = sv < best
            best = torch.where(upd, sv, best)
            te_at = torch.where(upd, se, te_at)
        outs.append((1 - te_at / sst).float().cpu().numpy())
    return np.concatenate(outs)
