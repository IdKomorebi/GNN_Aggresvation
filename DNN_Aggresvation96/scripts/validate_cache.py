# -*- coding: utf-8 -*-
"""校验 order-5 全量子集缓存：随机抽 N 个五阶子集，把缓存值与现场重算值逐点比对。

build_subs 走的是 np.fromiter + colex 秩散射的路径，与冒烟测试（sample 模式）
不同，故必须单独验一次；缓存一旦错位，k=6 的全部 syn 都会系统性错误。
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation69"))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, ridge_r2)

N_CHECK = 240
CKPT = REPO / "DNN_Aggresvation93/outputs/oracle_l1_aug8_seed0.pt"
CACHE = ROOT / "outputs/subs_o5_oracle_l1_aug8_seed0_last.npy"

BIN = np.array([[comb(n, k) if k <= n else 0 for k in range(12)]
                for n in range(65)], dtype=np.int64)


def colex_ranks(arr):
    k = arr.shape[1]
    return BIN[arr, np.arange(1, k + 1)].sum(1)


def main() -> None:
    v_sub = np.load(CACHE)
    print(f"缓存 {CACHE.name}: {v_sub.shape}（应为 ({comb(44,5)}, 12)）")
    assert v_sub.shape[0] == comb(44, 5)
    print(f"  全零行数: {int((v_sub == 0).all(1).sum())}（应为 0 或极少）")

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
    phi = FrozenPhi(load_oracle(CKPT, n_gen, n_conf, dev), "last")

    rng = np.random.RandomState(20260727)
    sets = np.sort(np.array([rng.choice(n_gen, 5, replace=False)
                             for _ in range(N_CHECK)]), axis=1)
    out = []
    for s in range(0, len(sets), 48):
        sb = torch.as_tensor(sets[s:s + 48], dtype=torch.long, device=dev)
        F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, "last")
        F_ev = build_features(phi, Xev, Zev, sb, n_gen, "last")
        out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
    fresh = np.concatenate(out, 0)
    cached = v_sub[colex_ranks(sets)]
    d = np.abs(fresh - cached)
    print(f"  逐点最大绝对差: {d.max():.3e}   中位: {np.median(d):.3e}")
    print("  ✅ 缓存校验通过" if d.max() < 1e-5 else "  ❌ 缓存错位，禁止用于 k=6 扫描")
    assert d.max() < 1e-5


if __name__ == "__main__":
    main()
