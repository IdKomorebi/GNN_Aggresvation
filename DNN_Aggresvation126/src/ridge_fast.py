# -*- coding: utf-8 -*-
"""126 号：闭式岭读出的等价快速实现（与 122 号 readout122.ridge_predict 数学上完全相同，只改计算顺序）。
  1. 全体训练行的 Gram 矩阵 TᵀT、TᵀY、列和只算一次；k 折中训练部分的量 = 全体 − 验证折（减法，不重算）；
  2. 6 个 λ 合并成一次批量线性求解。
口径不变：特征用全体训练行统计量标准化、sd 下限、测试特征截断、5 折选 λ（逐集合逐目标）、全体训练行重拟合、预测截断。"""
import numpy as np
import torch

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def ridge_predict(F_tr, Y_tr, F_te, fit_idx=None, val_idx=None, sd_floor=1e-3, cv="kfold", folds=None,
                  clip_te=True, clip_y=False, alphas=None, **_):
    assert cv == "kfold"
    mu = F_tr.mean(1, keepdim=True); sd = F_tr.std(1, keepdim=True).clamp_min(sd_floor)
    T, E = (F_tr - mu) / sd, (F_te - mu) / sd
    if clip_te:
        E = torch.minimum(torch.maximum(E, T.amin(1, keepdim=True)), T.amax(1, keepdim=True))
    AL = torch.as_tensor(ALPHAS if alphas is None else alphas, device=T.device, dtype=T.dtype); nA = len(AL)
    B, n, d = T.shape; C = Y_tr.shape[1]; eye = torch.eye(d, device=T.device, dtype=T.dtype)
    G = T.transpose(1, 2) @ T; TY = torch.einsum("bnp,nc->bpc", T, Y_tr); T1 = T.sum(1); Ys = Y_tr.sum(0)

    def solve_all(Gm, Hm):   # (B,d,d),(B,d,C) → (nA,B,d,C)
        A = Gm.unsqueeze(0) + AL.view(nA, 1, 1, 1) * eye
        return torch.linalg.solve(A, Hm.unsqueeze(0).expand(nA, -1, -1, -1))
    S = torch.zeros(nA, B, C, device=T.device, dtype=T.dtype)
    for f in folds:
        f = torch.as_tensor(f, device=T.device); Tv = T[:, f]; Yv = Y_tr[f]; nf = n - len(f)
        ym = (Ys - Yv.sum(0)) / nf
        Gf = G - Tv.transpose(1, 2) @ Tv
        Hf = (TY - torch.einsum("bnp,nc->bpc", Tv, Yv)) - (T1 - Tv.sum(1)).unsqueeze(2) * ym.view(1, 1, C)
        beta = solve_all(Gf, Hf)                                           # (nA,B,d,C)
        S += (((Yv - ym).view(1, 1, len(f), C) - Tv.unsqueeze(0) @ beta) ** 2).sum(2)
    best = S.argmin(0)                                                     # (B,C)
    ym = Ys / n; H = TY - T1.unsqueeze(2) * ym.view(1, 1, C)
    beta = solve_all(G, H)                                                 # (nA,B,d,C)
    bsel = torch.gather(beta, 0, best.view(1, B, 1, C).expand(1, B, d, C))[0]
    pred = E @ bsel + ym
    if clip_y:
        pred = torch.minimum(torch.maximum(pred, Y_tr.amin(0)), Y_tr.amax(0))
    return pred
