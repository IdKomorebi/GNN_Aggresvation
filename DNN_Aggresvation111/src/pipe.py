# -*- coding: utf-8 -*-
"""111–113 号共享流程：数据准备 → 专用重训真值 → 通用推断模型 → 字段风险表。

一份数据只需给出 (DataFrame, 候选字段列表, 敏感目标列表)，其余步骤全部相同，
保证不同数据集的结果口径一致。复用的历史实现按**文件路径**导入，避免各号 src/runlog 同名冲突：
  98 号 batch_truth.train_batched —— 并行专用重训（DNN 攻击者）
  100 号 mkfull.attack_max / closure —— 攻击器族取 max（val 选择）+ 单调闭包（val 选子集、test 报告）
  69 号 oracle.MLPOracle / sample_mask —— 随机掩码预训练的通用推断模型
  91 号 featridge.FrozenPhi / ridge_r2 / fit_val_idx —— 冻结表征 + 子集专用闭式读出

数据使用边界（与 FINAL_PROTOCOL 一致）：test 只用于最终报告 R²，不参与训练、轮次/攻击器/子集选择、
读出 β 求解或 λ 选择。
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPLIT_SEED, VAL_SEED = 42, 20260915


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mods():
    return dict(bt=_load("bt98", REPO / "DNN_Aggresvation98/src/batch_truth.py"),
                mk=_load("mk100", REPO / "DNN_Aggresvation100/src/mkfull.py"),
                orc=_load("orc69", REPO / "DNN_Aggresvation69/src/oracle.py"),
                fr=_load("fr91", REPO / "DNN_Aggresvation91/src/featridge.py"))


# ============================================================ 数据
def prep(df, cand: list[str], targ: list[str]) -> dict:
    """随机 70/30 划分（seed 42），train 内 15% 为 val（seed 20260915），仅用 train 统计量标准化。"""
    d = df[cand + targ].apply(lambda s: s.astype(float)).dropna().reset_index(drop=True)
    X, Y = d[cand].values.astype(np.float32), d[targ].values.astype(np.float32)
    n = len(d); perm = np.random.RandomState(SPLIT_SEED).permutation(n); nte = round(0.3 * n)
    te, tr = perm[:nte], perm[nte:]
    mx, sx = X[tr].mean(0), X[tr].std(0); sx[sx < 1e-10] = 1
    my, sy = Y[tr].mean(0), Y[tr].std(0); sy[sy < 1e-10] = 1
    X, Y = (X - mx) / sx, (Y - my) / sy
    p2 = np.random.RandomState(VAL_SEED).permutation(len(tr)); nv = round(0.15 * len(tr))
    return dict(Xtr=X[tr], Ytr=Y[tr], Xte=X[te], Yte=Y[te], fit_idx=p2[nv:], val_idx=p2[:nv],
                cand=list(cand), targ=list(targ), n=n)


def enumerate_sets(p: int, kmax: int):
    keys = [s for k in range(1, kmax + 1) for s in itertools.combinations(range(p), k)]
    M = np.zeros((len(keys), p), np.float32)
    for r, s in enumerate(keys):
        M[r, list(s)] = 1
    return M, keys


# ============================================================ 真值：三攻击器
def truth_dnn(D: dict, masks: np.ndarray, device: str, seed: int = 0, single: bool = False, chunk: int = 1024):
    """single=False：多目标 DNN（一个网络同时输出全部目标）；single=True：逐目标单输出 DNN。"""
    import torch
    bt = _load("bt98", REPO / "DNN_Aggresvation98/src/batch_truth.py")
    dev = torch.device(device)
    C = D["Ytr"].shape[1]
    clean = np.zeros((len(masks), C), np.float32); val = np.zeros_like(clean)
    targets = [list(range(C))] if not single else [[c] for c in range(C)]
    for cols in targets:
        Dc = dict(D); Dc["Ytr"] = D["Ytr"][:, cols]; Dc["Yte"] = D["Yte"][:, cols]
        for s in range(0, len(masks), chunk):
            R = bt.train_batched(masks[s:s + chunk], Dc, seed=seed * 100000 + s, device=dev)
            clean[s:s + chunk, cols] = R["clean"]; val[s:s + chunk, cols] = R["val_r2"]
    return clean, val


_G = {}


def _tree_init(D):
    _G.update(D)


def _tree_one(s):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    s = list(s); Xtr, Ytr, Xte, Yte = _G["Xtr"], _G["Ytr"], _G["Xte"], _G["Yte"]
    fi, vi = _G["fit_idx"], _G["val_idx"]
    out_t, out_v = [], []
    for c in range(Ytr.shape[1]):
        best = (-np.inf, None)
        for lr, leaf in [(0.05, 15), (0.1, 31)]:
            m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaf, max_iter=300,
                                              early_stopping=True, validation_fraction=0.15, random_state=0)
            m.fit(Xtr[fi][:, s], Ytr[fi, c])
            v = r2_score(Ytr[vi, c], m.predict(Xtr[vi][:, s]))
            if v > best[0]:
                best = (v, m)
        out_v.append(best[0]); out_t.append(r2_score(Yte[:, c], best[1].predict(Xte[:, s])))
    return out_t, out_v


def truth_tree(D: dict, keys, n_jobs: int = 40):
    """梯度提升树攻击者（CPU）。调用前须已设 OMP_NUM_THREADS=1。"""
    from multiprocessing import Pool
    with Pool(n_jobs, initializer=_tree_init, initargs=(D,)) as pool:
        res = pool.map(_tree_one, keys, chunksize=8)
    return (np.array([r[0] for r in res], np.float32), np.array([r[1] for r in res], np.float32))


def official_truth(parts: list[tuple[np.ndarray, np.ndarray]], keys):
    """攻击器族取 max（逐 (集合,目标) 在 val 上选攻击器），再做单调闭包。返回 V(N,C) 与各攻击器被选中比例。"""
    mk = _load("mk100", REPO / "DNN_Aggresvation100/src/mkfull.py")
    V, Vv, pick = mk.attack_max([np.clip(p[0], 0, 1) for p in parts], [p[1] for p in parts])
    Vbar = mk.closure(V, keys, Vval=Vv)
    share = [float((pick == a).mean()) for a in range(len(parts))]
    return Vbar.astype(np.float32), share


# ============================================================ 通用推断模型
def train_oracle(D: dict, seed: int, device: str, epochs: int = 400, patience: int = 60, bs: int = 256):
    """随机掩码预训练（69 号 sample_mask：先抽可见数 k~U{1..p}，再均匀抽 k 个字段）。
    早停只用 train 内部的 val 划分。"""
    import torch
    orc = _load("orc69", REPO / "DNN_Aggresvation69/src/oracle.py")
    dev = torch.device(device)
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
    p, C = Xtr.shape[1], Ytr.shape[1]
    fi = torch.as_tensor(D["fit_idx"], device=dev); vi = torch.as_tensor(D["val_idx"], device=dev)
    torch.manual_seed(seed); np.random.seed(seed)
    model = orc.MLPOracle(p, C).to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    mrng, vr = np.random.RandomState(1234 + seed), np.random.RandomState(999)
    vms = [torch.as_tensor(orc.sample_mask(len(vi), p, vr), device=dev) for _ in range(8)]
    best, bst, pat, t0 = 1e9, None, 0, time.time()
    for ep in range(epochs):
        model.train(); order = fi[torch.randperm(len(fi), device=dev)]
        for b in range(0, len(order), bs):
            ix = order[b:b + bs]; m = torch.as_tensor(orc.sample_mask(len(ix), p, mrng), device=dev)
            opt.zero_grad(); ((model(Xtr[ix], m) - Ytr[ix]) ** 2).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([((model(Xtr[vi], mm) - Ytr[vi]) ** 2).mean().item() for mm in vms]))
        if v < best - 1e-6:
            best, bst, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= patience:
                break
    model.load_state_dict(bst)
    return model.eval(), dict(val=best, epochs=ep + 1, sec=time.time() - t0)


def estimate(D: dict, models: list, masks: np.ndarray, device: str, raw: bool = True):
    """冻结 φ（各种子最后一层 ReLU）⊕ [x⊙m, (x⊙m)²] 上的子集专用闭式岭读出 → 每集合每目标的 test R²。
    λ 在 train 内部 fit/val 上逐 (集合,目标) 选择；test 只用于报告。"""
    import torch
    fr = _load("fr91", REPO / "DNN_Aggresvation91/src/featridge.py")
    dev = torch.device(device)
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
    Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev)
    fit_idx, val_idx = fr.fit_val_idx(len(Xtr), dev)
    phis = [fr.FrozenPhi(m, "last") for m in models]
    dim = (2 * Xtr.shape[1] if raw else 0) + 256 * len(phis)
    chunk = max(2, int(6e8 / (len(Xtr) * dim * 8)))

    def feats(X, m):
        xm = X.unsqueeze(0) * m.unsqueeze(1)
        blocks = [xm, xm ** 2] if raw else []
        return torch.cat(blocks + [ph(X, m) for ph in phis], 2)

    out = np.zeros((len(masks), Ytr.shape[1]), np.float32); t0 = time.time()
    for s in range(0, len(masks), chunk):
        m = torch.as_tensor(masks[s:s + chunk], device=dev)
        r2 = fr.ridge_r2(feats(Xtr, m).double(), Ytr.double(), feats(Xte, m).double(), Yte.double(), fit_idx, val_idx)
        out[s:s + chunk] = r2.float().cpu().numpy()
    return out, (time.time() - t0) / len(masks)


def closure_max(V: np.ndarray, keys):
    """估计器没有独立 val：闭包取子集上的最大值。"""
    mk = _load("mk100", REPO / "DNN_Aggresvation100/src/mkfull.py")
    return mk.closure(np.clip(V, 0, 1), keys).astype(np.float32)


# ============================================================ 字段风险表
def marg_tables(V: np.ndarray, keys, fields: list[int], kmax_bg: int):
    """对每个字段 i（在候选子集 fields 内）枚举背景 T⊆fields\\{i}、|T|≤kmax_bg，返回边际 Δ(p_sub, nT, C)、
    背景键 bk[i]、背景规模。V[()] 视为 0。keys 中须含全部所需集合。"""
    idx = {k: r for r, k in enumerate(keys)}; C = V.shape[1]
    Ds, bks = [], []
    for i in fields:
        oth = [j for j in fields if j != i]
        Ts = [T for k in range(kmax_bg + 1) for T in itertools.combinations(oth, k)]
        vT = np.stack([V[idx[T]] if T else np.zeros(C, np.float32) for T in Ts])
        vTi = np.stack([V[idx[tuple(sorted(T + (i,)))]] for T in Ts])
        Ds.append(vTi - vT); bks.append(Ts)
    bsz = np.array([len(T) for T in bks[0]])
    return np.stack(Ds), bks, bsz


def m_table(V: np.ndarray, keys, fields: list[int], kmax_bg: int):
    """M^(K)(p_sub, K+1, C) 与见证背景（字段下标元组）。"""
    Dm, bks, bsz = marg_tables(V, keys, fields, kmax_bg)
    C = V.shape[1]; P = len(fields)
    M = np.zeros((P, kmax_bg + 1, C), np.float32); W = [[[None] * C for _ in range(kmax_bg + 1)] for _ in range(P)]
    for K in range(kmax_bg + 1):
        sel = np.where(bsz <= K)[0]
        arg = Dm[:, sel, :].argmax(1)
        for a in range(P):
            for c in range(C):
                j = sel[arg[a, c]]; M[a, K, c] = Dm[a, j, c]; W[a][K][c] = bks[a][j]
    return M, W, Dm, bks, bsz


def gain_matrix(V: np.ndarray, keys, fields: list[int]):
    """G[a, b, c] = V({i,j}) − V({j})（字段 i=fields[a] 在背景 j=fields[b] 下的增益）；对角为 V({i})。"""
    idx = {k: r for r, k in enumerate(keys)}; P = len(fields); C = V.shape[1]
    G = np.full((P, P, C), np.nan, np.float32)
    for a, i in enumerate(fields):
        for b, j in enumerate(fields):
            G[a, b] = V[idx[(i,)]] if i == j else V[idx[tuple(sorted((i, j)))]] - V[idx[(j,)]]
    return G


def mus_list(V: np.ndarray, keys, fields: list[int], c: int, tau: float, kmax: int):
    """候选子集 fields 内、规模 ≤kmax 的 τ-最小不安全集合。"""
    fs = set(fields); idx = {k: r for r, k in enumerate(keys)}; out = []
    for k in keys:
        if len(k) > kmax or not set(k) <= fs or V[idx[k], c] <= tau:
            continue
        if all(V[idx[s], c] <= tau for r in range(1, len(k)) for s in itertools.combinations(k, r)):
            out.append(k)
    return out


def certify(Dt: np.ndarray, De: np.ndarray, bsz: np.ndarray, K: int, top: int = 3):
    """用估计器挑出的前 top 个背景，在真值边际上认证：返回 (精确 M, 估计 M, top-1 下界, top-k 下界)。"""
    sel = np.where(bsz <= K)[0]; dt, de = Dt[:, sel], De[:, sel]
    Mt, Me = dt.max(1), de.max(1)
    order = np.argsort(-de, axis=1)
    L1 = np.take_along_axis(dt, order[:, :1], 1)[:, 0]
    Lk = np.take_along_axis(dt, order[:, :top], 1).max(1)
    return Mt, Me, L1, Lk


def r2_single_linear(D: dict):
    """单字段线性相关的 r²（在 train 上计算）——现行"相关性"口径。"""
    X, Y = D["Xtr"], D["Ytr"]
    return np.array([[np.corrcoef(X[:, i], Y[:, c])[0, 1] ** 2 for c in range(Y.shape[1])]
                     for i in range(X.shape[1])], np.float32)
