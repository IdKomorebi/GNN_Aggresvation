# -*- coding: utf-8 -*-
"""96 号：k=6,7,8 高阶协同的**发现准确率**验证——扫描阶段。

问题
----
94 号把认证过的衰减律定到 k=5，并把 k>=6 标为"全枚举不可行"。重新核算后这一判断
对 k=6 过于保守：C(44,6)=7,059,052，按实测 ~2.5-3.5 ms/集合，双卡约 3 小时可全枚举。
真正不可行的是 k=7（3830 万）与 k=8（1.77 亿），改用**均匀随机抽样池**。

召回率在 k>=6 上无法测（无法枚举真值），但**精确率、校准偏差与判别力可以测**：
让估计器自选 top-K，连同同池随机对照一起送重训认证。

实现要点
--------
· 组合数系统（colex rank）把 (k-1) 阶子集映射为整数下标，全程 numpy 向量化，
  避免 4000 万次 Python 字典查找；
· full 模式先建 order-(k-1) 全量子集缓存（逐 conf 的 v），再枚举 k 阶；
· sample 模式对每个候选连同其 k 个 (k-1) 阶子集一起算（去重后批量求解）；
· 口径与 93/94 号逐条一致：train 拟合、audit 评估、同一 alpha 网格、SPLIT_SEED。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations, islice
from math import comb
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

NG_MAX = 64
BIN = np.array([[comb(n, k) if k <= n else 0 for k in range(12)]
                for n in range(NG_MAX + 1)], dtype=np.int64)


def colex_ranks(arr: np.ndarray) -> np.ndarray:
    """(B,k) 升序子集 → colex 秩（0..C(n,k)-1），向量化。"""
    k = arr.shape[1]
    return BIN[arr, np.arange(1, k + 1)].sum(1)


def drop_one(arr: np.ndarray) -> np.ndarray:
    """(B,k) → (B,k,k-1)：逐位删除一个元素得全部 (k-1) 阶子集（保持升序）。"""
    B, k = arr.shape
    out = np.empty((B, k, k - 1), dtype=arr.dtype)
    for p in range(k):
        out[:, p] = np.delete(arr, p, axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="last")
    ap.add_argument("--order", type=int, required=True)
    ap.add_argument("--mode", choices=["full", "sample", "build_subs"], required=True)
    ap.add_argument("--n_sample", type=int, default=200000)
    ap.add_argument("--sample_seed", type=int, default=960727)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_top", type=int, default=5000, help="落盘的强候选数")
    ap.add_argument("--n_weak", type=int, default=5000, help="落盘的弱集合数（认证对照池）")
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    m = args.order

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

    def ev(sets_np: np.ndarray) -> np.ndarray:
        """(B,k) → (B,nC) 的 audit R²。"""
        out = []
        for s in range(0, len(sets_np), args.chunk):
            sb = torch.as_tensor(sets_np[s:s + args.chunk], dtype=torch.long, device=dev)
            F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, args.kind)
            F_ev = build_features(phi, Xev, Zev, sb, n_gen, args.kind)
            out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    stem = f"{ckpt.stem}_{args.kind}{args.tag}"
    subs_cache = ROOT / f"outputs/subs_o{m-1}_{stem}.npy"
    t0 = time.perf_counter()

    # ---------------- 模式 A：建 (m-1) 阶全量子集缓存 ----------------
    if args.mode == "build_subs":
        n_sub = comb(n_gen, m - 1)
        part = ROOT / f"outputs/subs_o{m-1}_{stem}_part{args.shard}of{args.nshard}.npy"
        lo = n_sub * args.shard // args.nshard
        hi = n_sub * (args.shard + 1) // args.nshard
        log("SCAN", "START", note=f"build_subs o{m-1}: 分片 {args.shard}/{args.nshard} "
                                  f"[{lo},{hi}) / {n_sub}")
        it = islice(combinations(range(n_gen), m - 1), lo, hi)
        buf_r, buf_v = [], []
        block = 200000
        done = 0
        while True:
            arr = np.fromiter((x for t in islice(it, block) for x in t), dtype=np.int8)
            if arr.size == 0:
                break
            arr = arr.reshape(-1, m - 1)
            buf_r.append(colex_ranks(arr.astype(np.int64)))
            buf_v.append(ev(arr.astype(np.int64)))
            done += len(arr)
            el = time.perf_counter() - t0
            print(f"  [{done}/{hi-lo}] {el:.0f}s ({el/done*1000:.2f} ms/集合)", flush=True)
        np.save(part, np.concatenate([np.concatenate(buf_r)[:, None].astype(np.float32),
                                      np.concatenate(buf_v)], axis=1))
        log("SCAN", "DONE", note=f"build_subs 分片 {args.shard} 完成 "
                                 f"{time.perf_counter()-t0:.0f}s → {part.name}")
        print(f"[build_subs] → {part.name}", flush=True)
        return

    # ---------------- 载入 / 合并子集缓存 ----------------
    # sample 模式若已有 (m-1) 阶全量缓存则直接查表：每候选省去 m 次子集求解（~m 倍加速）
    v_sub = None
    if args.mode == "sample" and subs_cache.exists():
        v_sub = np.load(subs_cache)
        print(f"[cache] 命中 {subs_cache.name} {v_sub.shape}，子集改为查表", flush=True)
    if args.mode == "full":
        if not subs_cache.exists():
            parts = sorted(ROOT.glob(f"outputs/subs_o{m-1}_{stem}_part*of*.npy"))
            assert parts, f"缺少子集缓存，请先跑 --mode build_subs：{subs_cache.name}"
            n_sub = comb(n_gen, m - 1)
            v_sub = np.zeros((n_sub, n_conf), dtype=np.float32)
            seen = np.zeros(n_sub, dtype=bool)
            for p in parts:
                d = np.load(p)
                r = d[:, 0].astype(np.int64)
                v_sub[r] = d[:, 1:]
                seen[r] = True
            assert seen.all(), f"子集缓存不完整：缺 {int((~seen).sum())} 个"
            np.save(subs_cache, v_sub)
            print(f"[merge] {len(parts)} 个分片 → {subs_cache.name}", flush=True)
        else:
            v_sub = np.load(subs_cache)
        log("SCAN", "SUBS", note=f"子集缓存就绪 {v_sub.shape}")

    # ---------------- 枚举 / 抽样 k 阶集合 ----------------
    rows_idx, rows_syn, rows_conf, rows_vset = [], [], [], []

    def process(arr: np.ndarray, best: np.ndarray):
        v_set = ev(arr)
        inc = v_set - best
        c = inc.argmax(1)
        rows_idx.append(arr.astype(np.int8))
        rows_syn.append(inc[np.arange(len(arr)), c].astype(np.float32))
        rows_conf.append(c.astype(np.int8))
        rows_vset.append(v_set.max(1).astype(np.float32))

    if args.mode == "full":
        n_all = comb(n_gen, m)
        lo = n_all * args.shard // args.nshard
        hi = n_all * (args.shard + 1) // args.nshard
        log("SCAN", "START", note=f"full o{m}: 分片 {args.shard}/{args.nshard} "
                                  f"[{lo},{hi}) / {n_all}")
        it = islice(combinations(range(n_gen), m), lo, hi)
        block = 100000
        done = 0
        while True:
            arr = np.fromiter((x for t in islice(it, block) for x in t), dtype=np.int8)
            if arr.size == 0:
                break
            arr = arr.reshape(-1, m).astype(np.int64)
            sub_r = colex_ranks(drop_one(arr).reshape(-1, m - 1)).reshape(len(arr), m)
            best = v_sub[sub_r].max(1)
            process(arr, best)
            done += len(arr)
            el = time.perf_counter() - t0
            print(f"  [{done}/{hi-lo}] {el:.0f}s ({el/done*1000:.2f} ms/集合, "
                  f"ETA {(hi-lo-done)*el/done/60:.0f} min)", flush=True)
    else:  # sample
        rng = np.random.RandomState(args.sample_seed + args.shard)
        n_need = args.n_sample
        log("SCAN", "START", note=f"sample o{m}: {n_need} 个候选（空间 {comb(n_gen,m)}）")
        cand = set()
        while len(cand) < n_need:
            batch = np.sort(np.array([rng.choice(n_gen, m, replace=False)
                                      for _ in range(min(50000, n_need - len(cand) + 1000))]),
                            axis=1)
            cand.update(map(tuple, batch))
        arr_all = np.array(sorted(cand)[:n_need], dtype=np.int64)
        block = 20000
        for s in range(0, len(arr_all), block):
            arr = arr_all[s:s + block]
            subs = drop_one(arr).reshape(-1, m - 1)
            r = colex_ranks(subs)
            if v_sub is not None:                      # 查表路径
                best = v_sub[r].reshape(len(arr), m, n_conf).max(1)
            else:                                      # 现场求解路径（去重后批量）
                uniq_r, inv = np.unique(r, return_inverse=True)
                first = np.zeros(len(uniq_r), dtype=np.int64)
                first[inv[::-1]] = np.arange(len(inv))[::-1]
                v_uni = ev(subs[first])
                best = v_uni[inv].reshape(len(arr), m, n_conf).max(1)
            process(arr, best)
            el = time.perf_counter() - t0
            done = s + len(arr)
            print(f"  [{done}/{len(arr_all)}] {el:.0f}s "
                  f"(ETA {(len(arr_all)-done)*el/done/60:.0f} min)", flush=True)

    idx = np.concatenate(rows_idx)
    syn = np.concatenate(rows_syn)
    conf = np.concatenate(rows_conf)
    vset = np.concatenate(rows_vset)

    # ---------------- 落盘：强候选 + 弱对照池 + 阈值计数 ----------------
    order_desc = np.argsort(-syn)
    keep_top = order_desc[:args.n_top]
    weak_pool = np.where(syn < 0.05)[0]
    rng2 = np.random.RandomState(1234 + args.shard)
    keep_weak = (rng2.choice(weak_pool, min(args.n_weak, len(weak_pool)), replace=False)
                 if len(weak_pool) else np.array([], dtype=int))
    keep = np.unique(np.concatenate([keep_top, keep_weak]))
    df = pd.DataFrame(dict(
        indices=[str(tuple(int(x) for x in idx[i])) for i in keep],
        syn=syn[keep].astype(float), conf=conf[keep].astype(int),
        v_set=vset[keep].astype(float)))
    out = ROOT / f"outputs/hi_o{m}_{stem}_s{args.shard}of{args.nshard}.parquet"
    df.to_parquet(out, index=False)

    stats = dict(order=m, mode=args.mode, n_scanned=int(len(syn)),
                 space=int(comb(n_gen, m)), shard=args.shard, nshard=args.nshard,
                 max_syn=float(syn.max()), mean_syn=float(syn.mean()),
                 n_gt_005=int((syn > 0.05).sum()), n_gt_010=int((syn > 0.10).sum()),
                 n_gt_015=int((syn > 0.15).sum()), n_gt_020=int((syn > 0.20).sum()),
                 sec=round(time.perf_counter() - t0, 1), ckpt=ckpt.stem)
    (ROOT / f"outputs/hi_o{m}_{stem}_s{args.shard}of{args.nshard}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[o{m}/{args.mode}/s{args.shard}] {len(syn)} 集合 max_syn={syn.max():.4f} "
          f">0.10={stats['n_gt_010']} → {out.name}", flush=True)
    log("SCAN", "DONE", note=f"o{m} {args.mode} 分片{args.shard}: {len(syn)} 集合 "
                             f"max={syn.max():.4f} >0.10={stats['n_gt_010']} "
                             f"{stats['sec']:.0f}s")


if __name__ == "__main__":
    main()
