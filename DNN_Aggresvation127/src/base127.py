# -*- coding: utf-8 -*-
"""127 号：基于"全模型"（全部候选字段可见训练）的免重训基线。
  train_full    多输出 MLP（2×128 ReLU，dropout 0.15），Adam 1e-3 / wd 5e-4，批 128，≤300 轮，验证早停 60 轮
  dropout_est   Dropout / 均值代替：被移除特征置为训练均值（标准化后为 0），全模型直接预测（多种子预测平均）
  lazy_est      LazyVI（Gao et al., ICML 2022）：f(x̃;θ0+Δ) ≈ f(x̃;θ0) + J(x̃)Δ，岭惩罚 λ‖Δ‖²，核形式闭式解；
                λ 由留一交叉验证（K 的特征分解）选取；逐输出（目标）单独求解
  ws_est        热启动 + 早停（Sun & Raskutti）：从全模型出发，在 x̃ 上继续训练，最多 100 步（批 256），每 10 步验证，取验证最优
数据边界：只用训练行（早停、λ 选择在训练行内部）；测试行只用于报告 R²。"""
import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev


class MLP(nn.Module):
    def __init__(self, p, C, h=128, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, C))

    def forward(self, x):
        return self.net(x)


def r2(pred, Y):
    Y = torch.as_tensor(Y, device=pred.device, dtype=torch.float64); pred = pred.double()
    sst = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return (1 - ((pred - Y) ** 2).sum(-2) / sst).clamp_min(0.0)


def train_full(D, seed, dev, epochs=300, patience=60, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.as_tensor(D["Xtr"], device=dev); Y = torch.as_tensor(D["Ytr"], device=dev)
    fit = torch.as_tensor(D["fit_idx"], device=dev); val = torch.as_tensor(D["val_idx"], device=dev)
    m = MLP(X.shape[1], Y.shape[1]).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
    best, bst, pat, t0 = 1e9, None, 0, time.time()
    for ep in range(epochs):
        m.train(); order = fit[torch.randperm(len(fit), device=dev)]
        for b in range(0, len(order), bs):
            ix = order[b:b + bs]; opt.zero_grad(); ((m(X[ix]) - Y[ix]) ** 2).mean().backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = ((m(X[val]) - Y[val]) ** 2).mean().item()
        if v < best - 1e-7:
            best, bst, pat = v, deepcopy(m.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                break
    m.load_state_dict(bst); return m.eval(), dict(epochs=ep + 1, sec=time.time() - t0, val=best)


@torch.no_grad()
def dropout_est(D, models, masks, dev, B=64):
    Xte = torch.as_tensor(D["Xte"], device=dev); out = []; M = torch.as_tensor(masks, device=dev)
    for s in range(0, len(M), B):
        m = M[s:s + B]; xt = (Xte.unsqueeze(0) * m.unsqueeze(1))                 # (B,n,p)
        pred = sum(md(xt) for md in models) / len(models)                          # (B,n,C)
        out.append(r2(pred, D["Yte"]))
    return torch.cat(out).cpu().numpy().astype(np.float32)


def _jac(model, X):
    """每行对全部参数的雅可比：(n, C, P)。"""
    params = {k: v.detach() for k, v in model.named_parameters()}
    def f(pr, x):
        return functional_call(model, pr, (x.unsqueeze(0),))[0]
    J = vmap(jacrev(f), in_dims=(None, 0))(params, X)                              # dict: (n,C,*shape)
    return torch.cat([j.reshape(X.shape[0], j.shape[1], -1) for j in J.values()], 2)


LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@torch.no_grad()
def lazy_est(D, model, masks, dev, chunk=512, dtype=torch.float32):
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev)
    Ytr = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64); C = Ytr.shape[1]; out = []
    model = model.eval()
    for mk in masks:
        m = torch.as_tensor(mk, device=dev)[None]; xtr, xte = Xtr * m, Xte * m
        f0tr, f0te = model(xtr).double(), model(xte).double()
        with torch.enable_grad():
            Jtr = torch.cat([_jac(model, xtr[i:i + chunk]) for i in range(0, len(xtr), chunk)])      # (n,C,P)
            Jte = torch.cat([_jac(model, xte[i:i + chunk]) for i in range(0, len(xte), chunk)])
        pred = torch.zeros(len(xte), C, device=dev, dtype=torch.float64)
        for c in range(C):
            A = Jtr[:, c].to(dtype); K = A @ A.T; Kte = Jte[:, c].to(dtype) @ A.T     # 消费级 GPU 双精度很慢：核与特征分解用单精度
            s, Q = torch.linalg.eigh(K); s = s.clamp_min(0); r = (Ytr[:, c] - f0tr[:, c]).to(dtype); Qr = Q.T @ r
            scale = s.mean(); best = (np.inf, None)
            for lam in LAMS:
                w = s / (s + lam * scale); yhat = Q @ (w * Qr); h = (Q ** 2) @ w
                loo = (((r - yhat) / (1 - h).clamp_min(1e-6)) ** 2).mean().item()
                if loo < best[0]:
                    best = (loo, lam)
            alpha = Q @ (Qr / (s + best[1] * scale))
            pred[:, c] = f0te[:, c] + (Kte @ alpha).double()
        out.append(r2(pred[None], D["Yte"])[0])
    return torch.stack(out).cpu().numpy().astype(np.float32)


def ws_est(D, model, masks, dev, steps=100, bs=256):
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev); Y = torch.as_tensor(D["Ytr"], device=dev)
    fit = torch.as_tensor(D["fit_idx"], device=dev); val = torch.as_tensor(D["val_idx"], device=dev); out = []
    for j, mk in enumerate(masks):
        m = torch.as_tensor(mk, device=dev)[None]; md = deepcopy(model).train(); opt = torch.optim.Adam(md.parameters(), lr=1e-3, weight_decay=5e-4)
        g = torch.Generator(device=dev).manual_seed(j)
        def vloss():
            md.eval()
            with torch.no_grad():
                v = ((md(Xtr[val] * m) - Y[val]) ** 2).mean().item()
            md.train(); return v
        best, bst = vloss(), deepcopy(md.state_dict())
        for st in range(steps):
            ix = fit[torch.randint(len(fit), (bs,), device=dev, generator=g)]
            opt.zero_grad(); ((md(Xtr[ix] * m) - Y[ix]) ** 2).mean().backward(); opt.step()
            if (st + 1) % 10 == 0:
                v = vloss()
                if v < best:
                    best, bst = v, deepcopy(md.state_dict())
        md.load_state_dict(bst); md.eval()
        with torch.no_grad():
            out.append(r2(md(Xte * m)[None], D["Yte"])[0])
    return torch.stack(out).cpu().numpy().astype(np.float32)
