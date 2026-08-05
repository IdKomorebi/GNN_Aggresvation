# -*- coding: utf-8 -*-
"""93 号 延伸：★让 L1/L0 **自己**选 order-5 的 top-K，再送去认证。

为什么必须这样比
----------------
`l0_on_certified.py` 只能在 **full 字典选出的** 60 个集合上比各估计器，
那批集合带 full 字典的选择偏差——L0/L1 在上面表现好，可能只是"没跟着犯同一个错"。
真正的问题是：**换成学出来的特征去搜，选出的 top-K 经认证后是不是真的更强？**

所以本脚本用指定估计器**全枚举** order-5（C(44,5)=1,086,008）+ 其全部四阶子集
（C(44,4)=135,751），算 syn5，输出 top-K 供 `certify_highorder.py` 认证。

口径与 90 号一致：train 拟合、audit 评估、同一 alpha 网格、SPLIT_SEED=880725。
成本：φ 前向约 2.5 ms/集合（91 号 E5 实测）⟹ 122 万个集合单卡约 50 分钟，4 卡约 13 分钟。
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
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, ridge_r2)
from runlog import log  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="cat3")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--subs_only", action="store_true",
                    help="只算 m-1 阶子集并存盘。分片跑五元组前先跑一次，"
                         "避免每个分片重复算 13.6 万个子集（省约 17 分钟）")
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

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
    phi = FrozenPhi(load_oracle(ckpt, n_gen, n_conf, dev),
                    "cat3" if args.kind.startswith("cat3") else "last")

    def ev(sl):
        out = []
        for s in range(0, len(sl), args.chunk):
            sb = torch.as_tensor(sl[s:s + args.chunk], device=dev)
            F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, args.kind)
            F_ev = build_features(phi, Xev, Zev, sb, n_gen, args.kind)
            out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
        return np.concatenate(out, 0)

    m = args.order
    t0 = time.perf_counter()
    subs = list(combinations(range(n_gen), m - 1))
    sub_pos = {s: i for i, s in enumerate(subs)}
    cache = ROOT / f"outputs/subs_o{m-1}_{ckpt.stem}_{args.kind}{args.tag}.npy"
    log("SCAN5", "START", note=f"{ckpt.stem}/{args.kind} shard{args.shard}/{args.nshard}: "
                               f"{len(subs)} 个 {m-1} 阶子集，缓存{'命中' if cache.exists() else '待建'}")
    if cache.exists() and not args.subs_only:
        v_sub = np.load(cache)
        assert v_sub.shape[0] == len(subs), f"缓存与子集数不符：{v_sub.shape[0]} vs {len(subs)}"
        log("SCAN5", "SUBS", note=f"读缓存 {cache.name}")
    else:
        v_sub = ev(subs)
        np.save(cache, v_sub)
        log("SCAN5", "SUBS", note=f"{len(subs)} 个子集算完并存盘 {time.perf_counter()-t0:.0f}s")
    if args.subs_only:
        print(f"[subs_only] {len(subs)} 个 {m-1} 阶子集已存 → {cache.name}", flush=True)
        return

    all_sets = list(combinations(range(n_gen), m))
    mine = all_sets[args.shard::args.nshard]
    v_set = ev(mine)
    log("SCAN5", "SETS", note=f"{len(mine)} 个 {m} 阶完成 {time.perf_counter()-t0:.0f}s")

    syn = np.empty(len(mine))
    conf = np.empty(len(mine), dtype=int)
    for i, S in enumerate(mine):
        best = v_sub[[sub_pos[t] for t in combinations(S, m - 1)]].max(0)
        inc = v_set[i] - best
        c = int(np.argmax(inc))
        syn[i], conf[i] = inc[c], c
    df = pd.DataFrame(dict(indices=[str(s) for s in mine], syn=syn, conf=conf,
                           v_set=v_set.max(axis=1)))
    out = ROOT / (f"outputs/scan_o{m}_{ckpt.stem}_{args.kind}{args.tag}"
                  f"_s{args.shard}of{args.nshard}.parquet")
    df.to_parquet(out, index=False)
    el = time.perf_counter() - t0
    print(f"[o{m}/{ckpt.stem}/{args.kind}/s{args.shard}] {len(mine)} sets {el:.0f}s "
          f"max_syn={syn.max():.4f} n>0.2={(syn>0.2).sum()} → {out.name}", flush=True)
    log("SCAN5", "DONE", note=f"shard{args.shard} {el:.0f}s max_syn={syn.max():.4f} "
                              f"n>0.2={(syn>0.2).sum()}")


if __name__ == "__main__":
    main()
