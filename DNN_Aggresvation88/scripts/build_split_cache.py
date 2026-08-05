# -*- coding: utf-8 -*-
"""88 号 严谨性修正 #1：把"测试集参与搜索"改成三分割（GPT 指出，已接受）。

问题：85/86/87 的结构化查询直接返回 **test R²**，而 beam 用它做了几十万次自适应搜索。
反复在同一测试集上做自适应选择会使标准测试集保证失效（adaptive data analysis）。

修正：把原 test（1967 行）再切成两半——
    · search 集：beam/peel/阈值/宽度全部在这里决定，可任意反复使用；
    · audit 集：**只在最终报数时用一次**。
训练与 alpha 选择仍在 train 内部（fit/val），与原口径一致。

产出 `outputs/poly2_moments_split.npz`，字段与 85 号同名，另加 `_audit` 一套。
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from runlog import log  # noqa: E402

SPLIT_SEED = 880725   # search/audit 二分的固定种子


def dictionary(z: np.ndarray) -> np.ndarray:
    pairs = list(combinations(range(z.shape[1]), 2))
    products = np.column_stack([z[:, a] * z[:, b] for a, b in pairs])
    return np.column_stack([z, z**2, products]).astype(np.float64)


def moments(x, y):
    return x.T @ x, x.T @ y, np.sum(y**2, axis=0)


def main() -> None:
    t0 = time.perf_counter()
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])

    train_z = data["train_data"][:, gi].astype(np.float64)
    train_y = data["train_data"][:, ci].astype(np.float64)
    held_z = data["test_data"][:, gi].astype(np.float64)
    held_y = data["test_data"][:, ci].astype(np.float64)

    # ---- 原 test 一分为二：search / audit ----
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(held_z))
    half = len(perm) // 2
    s_idx, a_idx = perm[:half], perm[half:]
    log("SPLIT", "INFO", note=f"train={len(train_z)} search={len(s_idx)} audit={len(a_idx)}")

    phi_train = dictionary(train_z)
    phi_search = dictionary(held_z[s_idx])
    phi_audit = dictionary(held_z[a_idx])

    # 训练内部 fit/val（用于 alpha 选择），与 85 号同口径
    p2 = np.random.RandomState(20260724).permutation(len(train_z))
    n_val = round(len(p2) * 0.15)
    val_idx, fit_idx = p2[:n_val], p2[n_val:]

    fit_mean, fit_std = phi_train[fit_idx].mean(0), phi_train[fit_idx].std(0)
    fit_std[fit_std < 1e-8] = 1.0
    x_fit = (phi_train[fit_idx] - fit_mean) / fit_std
    x_val = (phi_train[val_idx] - fit_mean) / fit_std
    y_fit_mean = train_y[fit_idx].mean(0)
    g_fit, h_fit, _ = moments(x_fit, train_y[fit_idx] - y_fit_mean)
    g_val, h_val, rtr_val = moments(x_val, train_y[val_idx] - y_fit_mean)

    all_mean, all_std = phi_train.mean(0), phi_train.std(0)
    all_std[all_std < 1e-8] = 1.0
    x_all = (phi_train - all_mean) / all_std
    y_all_mean = train_y.mean(0)
    g_all, h_all, _ = moments(x_all, train_y - y_all_mean)

    out = dict(g_fit=g_fit, h_fit=h_fit, g_val=g_val, h_val=h_val, rtr_val=rtr_val,
               g_all=g_all, h_all=h_all)
    for tag, phi, yy in (("search", phi_search, held_y[s_idx]), ("audit", phi_audit, held_y[a_idx])):
        x = (phi - all_mean) / all_std
        g, h, rtr = moments(x, yy - y_all_mean)
        out[f"g_{tag}"] = g
        out[f"h_{tag}"] = h
        out[f"rtr_{tag}"] = rtr
        out[f"total_{tag}"] = ((yy - yy.mean(0)) ** 2).sum(0) + 1e-12

    np.savez(ROOT / "outputs/poly2_moments_split.npz", **out)
    log("SPLIT", "DONE", note=f"三分割缓存写出，用时 {time.perf_counter()-t0:.0f}s")
    print(f"写出 outputs/poly2_moments_split.npz  "
          f"(train={len(train_z)}, search={len(s_idx)}, audit={len(a_idx)}) "
          f"[{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
