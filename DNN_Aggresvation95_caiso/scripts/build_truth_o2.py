#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""95_caiso：构建 order-2 重训真值表（44 单字段 + 946 对 = 990 次专用 DNN 重训）。

协议 = 88 号修正版（certify_highorder 同款 worker）：专用 DNN(128,128,dropout0.15)、
Adam 1e-3/5e-4、EPOCHS 400/PATIENCE 120、早停用训练集内部 15% val（seed 20260725）、
测试集完全不参与训练。注意：PJM 的 68 号真值用的是旧 worker（测试集早停，绝对值
偏乐观）；CAISO 真值从第一天就是修正协议，绝对值可信。

产物：
  outputs/truth_o2_sets_s{shard}of{nshard}.csv  逐 (set, conf) 的 v_audit / v_full
  （聚合脚本 finalize_truth_o2.py 由此生成 synergy2_perconf.csv 供 eval_l1_order2 用）

断点续跑：逐集合追加写；同 shard 布局重启自动跳过。
用法：CUDA_VISIBLE_DEVICES=1 python scripts/build_truth_o2.py --shard 0 --nshard 4
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import SPLIT_SEED  # noqa: E402
from runlog import log  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 120


class DNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_generic(model, Xtr, Ytr):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    g = torch.Generator()
    g.manual_seed(20260725)
    perm = torch.randperm(len(Xtr), generator=g).to(DEV)
    n_val = max(int(len(Xtr) * 0.15), 1)
    vi, ti = perm[:n_val], perm[n_val:]
    Xv, Yv, Xt, Yt = Xtr[vi], Ytr[vi], Xtr[ti], Ytr[ti]
    best, bs, pat, n = 1e9, None, 0, len(Xt)
    for _ in range(EPOCHS):
        model.train()
        pm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            ix = pm[k:k + 128]
            opt.zero_grad()
            ((model(Xt[ix]) - Yt[ix]) ** 2).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Xv) - Yv) ** 2).mean())
        if v < best:
            best, bs, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(bs)
    model.eval()
    return model


def per_conf_r2(pred, true):
    ss = ((true - pred) ** 2).sum(0)
    st = ((true - true.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="重训随机性")
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

    sets = [(i,) for i in range(n_gen)] + list(combinations(range(n_gen), 2))
    mine = sets[args.shard::args.nshard]

    out_csv = ROOT / f"outputs/truth_o2_sets_s{args.shard}of{args.nshard}.csv"
    done = set()
    for f in glob.glob(str(ROOT / "outputs/truth_o2_sets_s*of*.csv")):
        try:
            done |= set(pd.read_csv(f)["S"].unique().tolist())
        except Exception:
            pass
    log("TRUTH2", "START", note=f"shard{args.shard}/{args.nshard}: {len(mine)} 集合，"
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
        rows = [dict(S=str(S), size=len(S), conf=conf_names[k],
                     v_audit=float(va[k]), v_full=float(vf[k]))
                for k in range(len(ci))]
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(),
                                  index=False)
        n_new += 1
        if n_new % 25 == 0:
            el = time.perf_counter() - t0
            log("TRUTH2", "PROGRESS",
                note=f"shard{args.shard} {n_new} 新完成 {el:.0f}s ({el/n_new:.1f}s/集合)")
            print(f"  [{n_new}] {S} {el:.0f}s", flush=True)

    log("TRUTH2", "DONE", note=f"shard{args.shard} 新完成 {n_new}，"
                               f"用时 {time.perf_counter()-t0:.0f}s")
    print(f"shard{args.shard} done: {n_new} new sets", flush=True)


if __name__ == "__main__":
    main()
