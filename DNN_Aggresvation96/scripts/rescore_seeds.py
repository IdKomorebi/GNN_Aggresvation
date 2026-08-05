# -*- coding: utf-8 -*-
"""96 号：用独立训练的 seed1/seed2 给 seed0 的高阶候选重新打分，检验跨种子秩一致性
过滤（94 号在 k=4 上的发现）能否延伸到 k=6/7/8。

与 94 号的口径差异（须诚实标注）：94 号的 cross_rank 是候选在 seed1/seed2 **全扫**
中的全局排名；本号 k>=6 无法为每个 seed 都做全扫（成本 3 倍），故改用
**池内相对排名**——候选在 seed0 自选 top-N 池内、按 seed_i 打分的排名。
这测的是"多个独立种子是否对同一批候选的强弱排序一致"，是原思想的弱化但可比版本。
"""
from __future__ import annotations

import argparse
import ast
import glob
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation69"))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, ridge_r2)
from runlog import log  # noqa: E402

BIN = np.array([[comb(n, k) if k <= n else 0 for k in range(12)]
                for n in range(65)], dtype=np.int64)


def drop_one(arr):
    B, k = arr.shape
    out = np.empty((B, k, k - 1), dtype=arr.dtype)
    for p in range(k):
        out[:, p] = np.delete(arr, p, axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True)
    ap.add_argument("--source", required=True, help="seed0 扫描产出前缀")
    ap.add_argument("--n_pool", type=int, default=3000, help="取 seed0 top-N 作为池")
    ap.add_argument("--chunk", type=int, default=48)
    args = ap.parse_args()
    m = args.order

    fs = sorted(glob.glob(str(ROOT / f"outputs/{args.source}_s*of*.parquet")))
    assert fs, args.source
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["Stup"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    pool = d.nlargest(args.n_pool, "syn").reset_index(drop=True)
    arr_all = np.array([list(t) for t in pool.Stup], dtype=np.int64)
    print(f"池：seed0 top-{len(pool)}（syn {pool.syn.min():.4f}–{pool.syn.max():.4f}）")

    dev = torch.device("cuda")
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi, ci = np.asarray(data["general_indices"]), np.asarray(data["confidential_indices"])
    Xtr_np = data["train_data"][:, gi].astype(np.float64)
    Ytr_np = data["train_data"][:, ci].astype(np.float64)
    Xhe_np = data["test_data"][:, gi].astype(np.float64)
    Yhe_np = data["test_data"][:, ci].astype(np.float64)
    n_gen, n_conf = Xtr_np.shape[1], Ytr_np.shape[1]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(Xhe_np))
    a_idx = perm[len(perm) // 2:]
    mu, sd = Xtr_np.mean(0), Xtr_np.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=dev)
    Xev = torch.tensor(Xhe_np[a_idx], dtype=torch.float32, device=dev)
    Ztr = torch.tensor((Xtr_np - mu) / sd, dtype=torch.float64, device=dev)
    Zev = torch.tensor((Xhe_np[a_idx] - mu) / sd, dtype=torch.float64, device=dev)
    Ytr = torch.tensor(Ytr_np, dtype=torch.float64, device=dev)
    Yev = torch.tensor(Yhe_np[a_idx], dtype=torch.float64, device=dev)
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)

    out = pool[["indices", "syn"]].rename(columns={"syn": "syn_s0"})
    for sd_i in (1, 2):
        ck = REPO / f"DNN_Aggresvation93/outputs/oracle_l1_aug8_seed{sd_i}.pt"
        phi = FrozenPhi(load_oracle(ck, n_gen, n_conf, dev), "last")

        def ev(sets_np):
            res = []
            for s in range(0, len(sets_np), args.chunk):
                sb = torch.as_tensor(sets_np[s:s + args.chunk], dtype=torch.long, device=dev)
                F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, "last")
                F_ev = build_features(phi, Xev, Zev, sb, n_gen, "last")
                res.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
            return np.concatenate(res, 0)

        subs = drop_one(arr_all).reshape(-1, m - 1)
        r = BIN[subs, np.arange(1, m)].sum(1)
        uniq, inv = np.unique(r, return_inverse=True)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        v_sub = ev(subs[first])[inv].reshape(len(arr_all), m, n_conf)
        v_set = ev(arr_all)
        syn = (v_set - v_sub.max(1)).max(1)
        out[f"syn_s{sd_i}"] = syn
        out[f"rank_s{sd_i}"] = (-syn).argsort().argsort() + 1
        print(f"  seed{sd_i} 重打分完成：max={syn.max():.4f}")
        log("RESCORE", "DONE", note=f"o{m} seed{sd_i} max={syn.max():.4f}")

    out["cross_rank"] = np.sqrt(out.rank_s1 * out.rank_s2)
    p = ROOT / f"outputs/rescore_o{m}.csv"
    out.to_csv(p, index=False)
    print(f"→ {p.name}（{len(out)} 行）")


if __name__ == "__main__":
    main()
