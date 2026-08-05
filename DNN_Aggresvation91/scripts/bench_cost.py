# -*- coding: utf-8 -*-
"""91 号 E5：成本基准——"闭式读出比微调更高效"要有硬数字，不能只说理。

在**同一张卡、同一批集合**上测三种处理的墙钟吞吐：
  oracle    共享读出（一次前向，当前方案）
  L0        冻结特征 + 闭式重解读出（本号方案）
  ft-K      微调 K 步（现行的逐集合细化）

诚实预期：纯 FLOPs 上闭式并不比 25 步微调省很多（一次前向 vs ~25 次前反向），
真正的差距来自 **L0 可跨集合批量、无优化器状态、无串行依赖**，
而微调必须一个集合一个优化循环。所以这里测的是**实际吞吐**，不是理论 FLOPs。
"""
from __future__ import annotations

import argparse
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
from src.oracle import MLPOracle  # noqa: E402
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, ridge_r2)
from runlog import log  # noqa: E402

CKPT = REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt"
BATCH, LR, WD = 256, 1e-3, 5e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sets", type=int, default=200)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kind", default="last")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--ft_k", type=int, default=25)
    args = ap.parse_args()

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
    Ytr_f = torch.tensor(Ytr_np, dtype=torch.float32, device=dev)
    Ztr = torch.tensor((Xtr_np - mu) / sd, dtype=torch.float64, device=dev)
    Zev = torch.tensor((Xhe_np[a_idx] - mu) / sd, dtype=torch.float64, device=dev)
    Ytr = torch.tensor(Ytr_np, dtype=torch.float64, device=dev)
    Yev = torch.tensor(Yhe_np[a_idx], dtype=torch.float64, device=dev)
    sst = ((Yev - Yev.mean(0)) ** 2).sum(0) + 1e-12
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)

    rng = np.random.RandomState(91)
    sets = [tuple(sorted(rng.choice(n_gen, args.order, replace=False)))
            for _ in range(args.n_sets)]
    base = load_oracle(CKPT, n_gen, n_conf, dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    state = ck["state"] if "state" in ck else ck
    rows = []

    def timeit(name, fn, warm=1):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        rows.append(dict(method=name, n_sets=args.n_sets, sec=round(el, 3),
                         ms_per_set=round(el / args.n_sets * 1000, 3)))
        print(f"  {name:14s} {el:7.2f}s  {el/args.n_sets*1000:8.2f} ms/set", flush=True)

    # ---- 1. oracle 共享读出 ----
    @torch.no_grad()
    def run_oracle():
        for S in sets:
            m = torch.zeros(Xev.shape[0], n_gen, device=dev)
            m[:, list(S)] = 1.0
            pred = base(Xev, m).double()
            _ = (1 - ((pred - Yev) ** 2).sum(0) / sst)
    timeit("oracle", run_oracle)

    # ---- 2. L0 闭式读出（可批量）----
    phi = FrozenPhi(base, "cat3" if args.kind.startswith("cat3") else "last")

    def run_l0():
        for s in range(0, len(sets), args.chunk):
            sb = torch.as_tensor(sets[s:s + args.chunk], device=dev)
            F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, args.kind)
            F_ev = build_features(phi, Xev, Zev, sb, n_gen, args.kind)
            _ = ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i)
    timeit(f"L0[{args.kind}]", run_l0)

    # ---- 3. 微调 K 步（必须逐集合，无法跨集合批量）----
    def run_ft():
        for S in sets:
            torch.manual_seed(0)
            model = MLPOracle(n_gen, n_conf)
            model.load_state_dict(state)
            model = model.to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            mtr = torch.zeros(len(Xtr), n_gen, device=dev)
            mtr[:, list(S)] = 1.0
            r = np.random.RandomState(0)
            step = 0
            while step < args.ft_k:
                order = r.permutation(len(Xtr))
                for b in range(0, len(order), BATCH):
                    ix = torch.as_tensor(order[b:b + BATCH], device=dev)
                    model.train()
                    opt.zero_grad()
                    ((model(Xtr[ix], mtr[ix]) - Ytr_f[ix]) ** 2).mean().backward()
                    opt.step()
                    step += 1
                    if step >= args.ft_k:
                        break
            model.eval()
            with torch.no_grad():
                m = torch.zeros(Xev.shape[0], n_gen, device=dev)
                m[:, list(S)] = 1.0
                _ = (1 - ((model(Xev, m).double() - Yev) ** 2).sum(0) / sst)
    timeit(f"ft{args.ft_k}", run_ft, warm=0)

    df = pd.DataFrame(rows)
    base_ms = df.loc[df.method.str.startswith("ft"), "ms_per_set"].iloc[0]
    df["vs_ft_speedup"] = (base_ms / df.ms_per_set).round(1)
    df.to_csv(ROOT / f"outputs/bench_cost_o{args.order}.csv", index=False)
    print(f"\n=== E5 成本基准（order-{args.order}，{args.n_sets} 个集合，单卡）===")
    print(df.to_string(index=False))
    log("E5", "DONE", note=f"order{args.order} " +
        " | ".join(f"{r.method}={r.ms_per_set}ms" for r in df.itertuples()))


if __name__ == "__main__":
    main()
