# -*- coding: utf-8 -*-
"""115 号：可配置的子集闭式岭读出（返回测试预测，支持预测层面的多种子平均与交叉拟合）。

与 91 号 featridge.ridge_r2 的口径一致（train 内 fit/val 选 λ、全 train 拟合、test 只报告），区别只在可选项：
  sd_floor   —— 特征标准化的标准差下限。91 号为 1e-8：训练集上近常数的特征（如死神经元）在测试集稍有变化
                就会被放大到天文数字，预测爆掉、R² 被截断为 0。本号检验这一数值问题。
  drop_const —— 训练集上标准差低于阈值的特征直接置零（不参与回归）。
  cv         —— "holdout"（91 号口径：一次 fit/val 划分）或 "kfold"（5 折，降低 λ 选择噪声）。
  clip_te    —— 测试特征截断到训练集取值范围（只用训练集信息，不看测试标签）。
"""
from __future__ import annotations
import numpy as np
import torch

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _solve(G, H, a, eye):
    return torch.linalg.solve(G + a * eye, H)


@torch.no_grad()
def ridge_predict(F_tr, Y_tr, F_te, fit_idx, val_idx, sd_floor=1e-8, drop_const=0.0, cv="holdout", folds=None,
                  clip_te=False):
    """F_tr:(B,n,d) Y_tr:(n,C) F_te:(B,m,d) → 测试预测 (B,m,C)（已加回均值）。"""
    mu = F_tr.mean(1, keepdim=True); sd = F_tr.std(1, keepdim=True)
    keep = (sd > drop_const).to(F_tr.dtype) if drop_const > 0 else torch.ones_like(sd)
    sd = sd.clamp_min(sd_floor)
    T, E = (F_tr - mu) / sd * keep, (F_te - mu) / sd * keep
    if clip_te:   # 测试特征截断到训练集取值范围，防止尖峰输入把线性读出外推爆掉
        E = torch.minimum(torch.maximum(E, T.amin(1, keepdim=True)), T.amax(1, keepdim=True))
    B, n, d = T.shape; C = Y_tr.shape[1]
    eye = torch.eye(d, device=T.device, dtype=T.dtype).expand(B, d, d)

    def sse_for(tr_idx, ev_idx):
        Tf, Tv = T[:, tr_idx], T[:, ev_idx]; Yf, Yv = Y_tr[tr_idx], Y_tr[ev_idx]; ym = Yf.mean(0)
        G = Tf.transpose(1, 2) @ Tf; H = torch.einsum("bnp,nc->bpc", Tf, Yf - ym)
        out = []
        for a in ALPHAS:
            beta = _solve(G, H, a, eye)
            out.append((((Yv - ym).unsqueeze(0) - Tv @ beta) ** 2).sum(1))       # (B,C)
        return torch.stack(out, 0)                                               # (nA,B,C)

    if cv == "holdout":
        S = sse_for(fit_idx, val_idx)
    else:
        S = sum(sse_for(np.setdiff1d(np.arange(n), f), f) for f in folds)
    best = S.argmin(0)                                                           # (B,C)
    ym = Y_tr.mean(0); G = T.transpose(1, 2) @ T; H = torch.einsum("bnp,nc->bpc", T, Y_tr - ym)
    pred = torch.zeros(B, E.shape[1], C, device=T.device, dtype=T.dtype)
    for ai, a in enumerate(ALPHAS):
        p = E @ _solve(G, H, a, eye) + ym
        pred = torch.where((best == ai).unsqueeze(1), p, pred)
    return pred


def r2_from_pred(pred, Y_te):
    sst = ((Y_te - Y_te.mean(0)) ** 2).sum(0) + 1e-12
    return (1 - ((pred - Y_te.unsqueeze(0)) ** 2).sum(1) / sst).clamp_min(0.0)
