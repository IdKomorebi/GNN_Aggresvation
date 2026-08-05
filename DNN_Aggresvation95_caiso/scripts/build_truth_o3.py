#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""95_caiso：无偏随机三元组池重训（对齐 77 号 h2_unbiased_pool 的 triple_rand 组）。

固定 RandomState(950726) 从 C(44,3) 均匀抽 N_TRIPLE 个三元组，逐个重训专用 DNN
（88 号修正协议，worker 与 build_truth_o2 相同）。syn3 的三个二阶子集 v 来自
order-2 真值表（同协议同 seed），由 finalize_truth.py 汇总时计算。

用法：CUDA_VISIBLE_DEVICES=1 python scripts/build_truth_o3.py --shard 0 --nshard 4
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import SPLIT_SEED  # noqa: E402
from runlog import log  # noqa: E402

from build_truth_o2 import DNN, train_generic, per_conf_r2  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIPLE = 400
POOL_SEED = 950726


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/caiso_2025_hourly_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    tr, te = di["train_data"], di["test_data"]
    n_gen = len(gi)
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(te))
    a_idx = perm[len(perm) // 2:]

    all_triples = list(combinations(range(n_gen), 3))
    pick = np.random.RandomState(POOL_SEED).choice(len(all_triples), N_TRIPLE,
                                                   replace=False)
    sets = [all_triples[i] for i in sorted(pick)]
    mine = sets[args.shard::args.nshard]

    out_csv = ROOT / f"outputs/truth_o3_sets_s{args.shard}of{args.nshard}.csv"
    done = set()
    for f in glob.glob(str(ROOT / "outputs/truth_o3_sets_s*of*.csv")):
        try:
            done |= set(pd.read_csv(f)["S"].unique().tolist())
        except Exception:
            pass
    log("TRUTH3", "START", note=f"shard{args.shard}/{args.nshard}: {len(mine)} 三元组，"
                                f"已完成 {len(done)}")

    Ytr_t = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Yte_full = te[:, ci]
    Yte_audit = te[a_idx][:, ci]
    conf_names = list(di["confidential"])

    t0 = time.perf_counter()
    n_new = 0
    for S in mine:
        if str(S) in done:
            continue
        cols = gi[list(S)]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        Xtr_t = torch.as_tensor(tr[:, cols], dtype=torch.float32, device=DEV)
        Xte_t = torch.as_tensor(te[:, cols], dtype=torch.float32, device=DEV)
        model = train_generic(DNN(len(S), len(ci)).to(DEV), Xtr_t, Ytr_t)
        with torch.no_grad():
            pred = model(Xte_t).cpu().numpy()
        va = per_conf_r2(pred[a_idx], Yte_audit)
        vf = per_conf_r2(pred, Yte_full)
        rows = [dict(S=str(S), size=3, conf=conf_names[k],
                     v_audit=float(va[k]), v_full=float(vf[k]))
                for k in range(len(ci))]
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(),
                                  index=False)
        n_new += 1
        if n_new % 25 == 0:
            el = time.perf_counter() - t0
            log("TRUTH3", "PROGRESS",
                note=f"shard{args.shard} {n_new} 新完成 {el:.0f}s ({el/n_new:.1f}s/集合)")

    log("TRUTH3", "DONE", note=f"shard{args.shard} 新完成 {n_new}，"
                               f"用时 {time.perf_counter()-t0:.0f}s")
    print(f"shard{args.shard} done: {n_new} new triples", flush=True)


if __name__ == "__main__":
    main()
