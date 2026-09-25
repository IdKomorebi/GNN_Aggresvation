# -*- coding: utf-8 -*-
"""122 号：随机掩码预训练为什么几乎不起作用？——隔离实验的公共件。

主干（MLPOracle，输入 [x⊙m, m]，3×256 ReLU）的训练方式：
  uniform        先抽可见数 k~U{1..p}，再均匀抽 k 个字段；输出本组目标（= 现行做法，111/116 号的本组主干）
  small          70% 的样本抽 k~U{1..4}（与查询规模对齐），30% 按 uniform
  recon          uniform 掩码；输出 = 本组目标 + 全部候选列，目标损失 + 被遮住候选列的重建损失（组内多任务，防表征塌缩）
  small_recon    small 掩码 + recon 输出
  random         不训练（随机初始化）
读出与 118 号单种子口径相同：[x⊙m, (x⊙m)², φ]，特征标准差下限 1e-3、测试特征截断、5 折选 λ。
另可把多组特征拼接（例如 [φ_trained, φ_random]）。
数据边界：训练与读出只用训练行（早停与 λ 用训练行内部划分）；测试行只用于报告 R²。"""
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
ro = _load("ro122", Path(__file__).resolve().parent / "readout122.py")
FIX = dict(sd_floor=1e-3, clip_te=True)


def sample(kind, b, p, rng):
    if kind.startswith("small"):
        m = orc.sample_mask(b, p, rng); sm = rng.rand(b) < 0.7
        for i in np.where(sm)[0]:
            k = rng.randint(1, min(4, p) + 1); m[i] = 0; m[i, rng.choice(p, k, replace=False)] = 1
        return m
    return orc.sample_mask(b, p, rng)


def train(D, kind, seed, device, epochs=400, patience=60, bs=256, hidden=256):
    """返回训练好的 MLPOracle（random 直接返回随机初始化）。"""
    dev = torch.device(device); p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]
    recon = kind.endswith("recon"); out = C + p if recon else C
    torch.manual_seed(seed); np.random.seed(seed)
    model = orc.MLPOracle(p, out, hidden=hidden).to(dev)
    if kind == "random":
        return model.eval(), dict(epochs=0, sec=0.0)
    Xt = torch.as_tensor(D["Xtr"], device=dev); Yt = torch.as_tensor(D["Ytr"], device=dev)
    fit = torch.as_tensor(D["fit_idx"], device=dev); val = torch.as_tensor(D["val_idx"], device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    mrng, vr = np.random.RandomState(1234 + seed), np.random.RandomState(999)
    vms = [torch.as_tensor(orc.sample_mask(len(val), p, vr), device=dev) for _ in range(8)]

    def loss(ix, m):
        o = model(Xt[ix], m); lt = ((o[:, :C] - Yt[ix]) ** 2).mean()
        if not recon:
            return lt
        h = 1 - m
        return lt + (((o[:, C:] - Xt[ix]) ** 2) * h).sum() / h.sum().clamp_min(1.0)

    def vloss():   # 早停只看目标损失（与现行做法可比）
        with torch.no_grad():
            return float(np.mean([((model(Xt[val], mm)[:, :C] - Yt[val]) ** 2).mean().item() for mm in vms]))

    best, bst, pat, t0 = 1e9, None, 0, time.time()
    for ep in range(epochs):
        model.train(); order = fit[torch.randperm(len(fit), device=dev)]
        for b in range(0, len(order), bs):
            ix = order[b:b + bs]; m = torch.as_tensor(sample(kind, len(ix), p, mrng), device=dev)
            opt.zero_grad(); loss(ix, m).backward(); opt.step()
        model.eval(); v = vloss()
        if v < best - 1e-6:
            best, bst, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(bst)
    return model.eval(), dict(epochs=ep + 1, sec=time.time() - t0, val=best)


@torch.no_grad()
def readout(D, phis, masks, device, B=4, raw=True, log=None):
    """phis：FrozenPhi 列表（特征拼接）；masks：(N,p)。返回 (N,C) 测试 R²（单次读出）。"""
    dev = torch.device(device)
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev)
    Ytr = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64); Yte = torch.as_tensor(D["Yte"], device=dev, dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
    Mt = torch.as_tensor(np.asarray(masks, np.float32), device=dev); out, t0 = [], time.time()

    def feats(X, m):
        xm = X.unsqueeze(0) * m.unsqueeze(1)
        return torch.cat(([xm, xm ** 2] if raw else []) + [ph(X, m) for ph in phis], 2).double()
    for s in range(0, len(Mt), B):
        m = Mt[s:s + B]
        pred = ro.ridge_predict(feats(Xtr, m), Ytr, feats(Xte, m), fi, vi, cv="kfold", folds=folds, **FIX)
        out.append(ro.r2_from_pred(pred, Yte))
        if log and s % (B * 250) == 0:
            log(f"{s + len(m)}/{len(Mt)} {time.time() - t0:.0f}s")
    return torch.cat(out).cpu().numpy().astype(np.float32), (time.time() - t0) / max(len(Mt), 1)


@torch.no_grad()
def diagnose(D, model, masks, device, kind="last"):
    """表征诊断：失效单元比例、有效维度（参与比）、以及特征对目标的线性可解释度（全训练行岭回归 R²，衡量"对齐"）。"""
    dev = torch.device(device); X = torch.as_tensor(D["Xtr"], device=dev); Y = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64)
    ph = fr.FrozenPhi(model, kind); dead, pr = [], []
    for m in masks:
        F = ph(X, torch.as_tensor(m[None], device=dev, dtype=torch.float32))[0].double()
        dead.append(float((F.std(0) < 1e-6).float().mean()))
        ev = torch.linalg.eigvalsh(torch.cov(F.T)).clamp_min(0); pr.append(float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)))
    return float(np.mean(dead)), float(np.mean(pr))


@torch.no_grad()
def readout_avg(D, phi_groups, masks, device, raw=True):
    """三种子预测平均（与新主口径 E 相同）：phi_groups 为若干组 FrozenPhi 列表，每组单独读出，预测取平均后算 R²。"""
    dev = torch.device(device)
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev)
    Ytr = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64); Yte = torch.as_tensor(D["Yte"], device=dev, dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
    Mt = torch.as_tensor(np.asarray(masks, np.float32), device=dev); out, t0 = [], time.time()
    for s in range(len(Mt)):
        m = Mt[s:s + 1]; preds = []
        for phis in phi_groups:
            def feats(X):
                xm = X.unsqueeze(0) * m.unsqueeze(1)
                return torch.cat(([xm, xm ** 2] if raw else []) + [ph(X, m) for ph in phis], 2).double()
            preds.append(ro.ridge_predict(feats(Xtr), Ytr, feats(Xte), fi, vi, cv="kfold", folds=folds, **FIX))
        out.append(ro.r2_from_pred(sum(preds) / len(preds), Yte))
    return torch.cat(out).cpu().numpy().astype(np.float32), (time.time() - t0) / max(len(Mt), 1)
