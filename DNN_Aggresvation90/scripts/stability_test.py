# -*- coding: utf-8 -*-
"""90 号 验证二：t 检验 vs R² 差，在真实数据上的【分片稳定性】。

89 号（Codex）的决定性发现：syn=R²差 在 audit 强尾部的 search→audit Spearman
是**负的**（order-4 −0.03、order-5 −0.12）——统计量在最该管用的地方失效。
本脚本用同一口径检验新统计量是否修好这一点。

做法：把训练数据分成两个不相交的半份 A/B（此处检验的是**估计量本身的稳定性**，
与 88 号的 search/audit 评估分片是不同层面的事）：
  · 在 A 上算 t 统计量与 syn；在 B 上独立再算一次；
  · 比较两份的 Spearman —— 全体、以及"在 B 上排前 K 的尾部"。
尾部相关才是发现任务真正依赖的量。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from fstat import highorder_t  # noqa: E402
from runlog import log  # noqa: E402

ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
N_SETS = 3000
ORDERS = (3, 4)


def poly2(Z):
    cols = [Z, Z ** 2]
    if Z.shape[1] > 1:
        cols.append(np.column_stack([Z[:, a] * Z[:, b]
                                     for a, b in combinations(range(Z.shape[1]), 2)]))
    return np.column_stack(cols)


def ridge_r2_multi(Ztr, Ytr, Zte, Yte):
    """逐 conf 的独立集 R²（向量）。"""
    Xtr, Xte = poly2(Ztr), poly2(Zte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    n = len(Xtr)
    rs = np.random.RandomState(0).permutation(n)
    nv = max(int(n * 0.15), 1)
    vi, fi = rs[:nv], rs[nv:]
    p = Xtr.shape[1]
    best_sse = np.full(Ytr.shape[1], np.inf)
    best_beta = None
    for a in ALPHAS:
        b = np.linalg.solve(Xtr[fi].T @ Xtr[fi] + a * np.eye(p), Xtr[fi].T @ (Ytr[fi] - ym))
        sse = np.sum((Ytr[vi] - ym - Xtr[vi] @ b) ** 2, axis=0)
        if best_beta is None:
            best_beta = np.zeros((p, Ytr.shape[1]))
        upd = sse < best_sse
        if upd.any():
            bfull = np.linalg.solve(Xtr.T @ Xtr + a * np.eye(p), Xtr.T @ (Ytr - ym))
            best_beta[:, upd] = bfull[:, upd]
            best_sse[upd] = sse[upd]
    pred = Xte @ best_beta + ym
    sse = np.sum((Yte - pred) ** 2, axis=0)
    sst = np.sum((Yte - Yte.mean(0)) ** 2, axis=0) + 1e-12
    return np.clip(1 - sse / sst, 0.0, None)


def main() -> None:
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    X = np.vstack([data["train_data"][:, gi], data["test_data"][:, gi]]).astype(np.float64)
    Y = np.vstack([data["train_data"][:, ci], data["test_data"][:, ci]]).astype(np.float64)
    n_gen = X.shape[1]

    rng = np.random.RandomState(900)
    perm = rng.permutation(len(X))
    half = len(perm) // 2
    IA, IB = perm[:half], perm[half:]
    log("STAB", "SPLIT", note=f"A={len(IA)} B={len(IB)} 两个不相交半份")

    def prep(idx):
        Xi = X[idx]
        mu, sd = Xi.mean(0), Xi.std(0)
        sd[sd < 1e-9] = 1.0
        return (Xi - mu) / sd, Y[idx]

    ZA, YA = prep(IA)
    ZB, YB = prep(IB)
    # 每半份内部再切 train/eval，供 R² 差用
    def inner(Z, Yy, seed):
        r = np.random.RandomState(seed).permutation(len(Z))
        k = int(len(r) * 0.7)
        return Z[r[:k]], Yy[r[:k]], Z[r[k:]], Yy[r[k:]]

    ZAtr, YAtr, ZAte, YAte = inner(ZA, YA, 1)
    ZBtr, YBtr, ZBte, YBte = inner(ZB, YB, 2)

    out = []
    for m in ORDERS:
        sets = set()
        while len(sets) < N_SETS:
            sets.add(tuple(sorted(rng.choice(n_gen, m, replace=False))))
        sets = sorted(sets)
        recs = []
        for i, S in enumerate(sets):
            cols = list(S)
            tA, _, _ = highorder_t(ZA[:, cols], YA)
            tB, _, _ = highorder_t(ZB[:, cols], YB)
            vA = ridge_r2_multi(ZAtr[:, cols], YAtr, ZAte[:, cols], YAte)
            vB = ridge_r2_multi(ZBtr[:, cols], YBtr, ZBte[:, cols], YBte)
            subA = np.max([ridge_r2_multi(ZAtr[:, list(T)], YAtr, ZAte[:, list(T)], YAte)
                           for T in combinations(S, m - 1)], axis=0)
            subB = np.max([ridge_r2_multi(ZBtr[:, list(T)], YBtr, ZBte[:, list(T)], YBte)
                           for T in combinations(S, m - 1)], axis=0)
            recs.append(dict(S=S,
                             tA=float(np.max(np.abs(tA))), tB=float(np.max(np.abs(tB))),
                             synA=float(np.max(vA - subA)), synB=float(np.max(vB - subB))))
            if (i + 1) % 500 == 0:
                log("STAB", "PROG", note=f"order{m} {i+1}/{len(sets)}")
        d = pd.DataFrame(recs)
        d.to_csv(ROOT / f"outputs/stability_order{m}.csv", index=False)

        row = dict(order=m, n=len(d))
        row["spearman_t_all"] = round(float(spearmanr(d.tA, d.tB).correlation), 4)
        row["spearman_syn_all"] = round(float(spearmanr(d.synA, d.synB).correlation), 4)
        for K in (100, 300):
            topB_t = d.nlargest(K, "tB")
            topB_s = d.nlargest(K, "synB")
            row[f"spearman_t_topB{K}"] = round(float(spearmanr(topB_t.tA, topB_t.tB).correlation), 4)
            row[f"spearman_syn_topB{K}"] = round(float(spearmanr(topB_s.synA, topB_s.synB).correlation), 4)
            # 尾部重合率：A 的 topK 与 B 的 topK 交集
            row[f"overlap_t_top{K}"] = round(len(set(map(str, d.nlargest(K, "tA").S)) &
                                                 set(map(str, topB_t.S))) / K, 4)
            row[f"overlap_syn_top{K}"] = round(len(set(map(str, d.nlargest(K, "synA").S)) &
                                                   set(map(str, topB_s.S))) / K, 4)
        out.append(row)
        log("STAB", "ROW", note=f"order{m} t_all={row['spearman_t_all']} syn_all={row['spearman_syn_all']} "
                                f"t_top100={row['spearman_t_topB100']} syn_top100={row['spearman_syn_topB100']}")

    res = pd.DataFrame(out)
    res.to_csv(ROOT / "outputs/stability_summary.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
