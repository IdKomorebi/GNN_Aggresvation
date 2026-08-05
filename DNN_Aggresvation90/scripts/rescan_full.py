# -*- coding: utf-8 -*-
"""90 号 ★重扫：用 full-interaction 字典重算 3/4/5 阶 syn，检验 87 号"衰减律"是否为字典假象。

设计要点
--------
1. **嵌套字典对比**（干净分离字典效应）：
   · poly2 : raw + square + 两两乘积          （= 85/86/87 号一直用的）
   · full  : poly2 **再加上** 3..m 阶乘积项    （⊃ poly2，表达力严格更强）
   两者在同一 train 拟合、同一 alpha 网格、同一评估分片上比较。
2. **口径守规矩**（88 号修正）：train 拟合 → search 选/排序 → audit 只报最终数。
   本脚本同时输出 search 与 audit 两套 v，供上层选择。
3. **GPU 批量**：逐集合构造 (B,n,p) 项矩阵，batched Gram + batched solve，
   一次分解服务 12 个 conf。4 卡按集合分片并行。

syn_m(S) = max_c [ v_c(S) − max_{T⊂S,|T|=m−1} v_c(T) ]，v 用各自 size 的同族字典。
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
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from runlog import log  # noqa: E402

ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SPLIT_SEED = 880725      # 与 88 号一致
VAL_SEED = 20260724      # 与 85 号一致


def term_spec(m: int, kind: str):
    """返回项的 (阶数元组) 列表。full ⊃ poly2。"""
    spec = [(i,) for i in range(m)]                                  # raw
    spec += [(i, i) for i in range(m)]                               # square
    spec += [c for c in combinations(range(m), 2)]                   # 两两乘积
    if kind == "full":
        for r in range(3, m + 1):
            spec += [c for c in combinations(range(m), r)]           # 3..m 阶乘积
    return spec


def build_batch(Z: torch.Tensor, sets: torch.Tensor, spec) -> torch.Tensor:
    """Z:(n,nG)  sets:(B,m)  → (B,n,p)"""
    B = sets.shape[0]
    n = Z.shape[0]
    cols = []
    for combo in spec:
        prod = torch.ones(B, n, device=Z.device, dtype=Z.dtype)
        for j in combo:
            prod = prod * Z[:, sets[:, j]].T
        cols.append(prod)
    return torch.stack(cols, dim=2)


def r2_batch(Ztr, Ytr, Zev, Yev, sets, spec, fit_idx, val_idx, chunk=400):
    """返回 (B, nC) 的独立集 R²；alpha 用 train 内部 val 逐 (set,conf) 选。"""
    device = Ztr.device
    outs = []
    for s in range(0, sets.shape[0], chunk):
        sb = sets[s:s + chunk]
        T = build_batch(Ztr, sb, spec)                     # (b,n,p)
        E = build_batch(Zev, sb, spec)                     # (b,ne,p)
        mu = T.mean(1, keepdim=True)
        sd = T.std(1, keepdim=True).clamp_min(1e-8)
        T = (T - mu) / sd
        E = (E - mu) / sd
        b, n, p = T.shape
        Tf, Tv = T[:, fit_idx], T[:, val_idx]
        Yf, Yv = Ytr[fit_idx], Ytr[val_idx]
        ymf = Yf.mean(0)
        Gf = Tf.transpose(1, 2) @ Tf
        Hf = torch.einsum("bnp,nc->bpc", Tf, Yf - ymf)
        Gv = Tv.transpose(1, 2) @ Tv
        Hv = torch.einsum("bnp,nc->bpc", Tv, Yv - ymf)
        rtrv = ((Yv - ymf) ** 2).sum(0)                    # (nC,)
        eye = torch.eye(p, device=device, dtype=T.dtype).expand(b, p, p)
        best = torch.full((b, Ytr.shape[1]), float("inf"), device=device, dtype=T.dtype)
        best_a = torch.zeros(b, Ytr.shape[1], dtype=torch.long, device=device)
        for ai, a in enumerate(ALPHAS):
            beta = torch.linalg.solve(Gf + a * eye, Hf)     # (b,p,nC)
            sse = (rtrv - 2 * (beta * Hv).sum(1)
                   + (beta * (Gv @ beta)).sum(1))
            upd = sse < best
            best = torch.where(upd, sse, best)
            best_a = torch.where(upd, ai, best_a)
        # 全 train 拟合 + 独立集评估
        ym = Ytr.mean(0)
        Ga = T.transpose(1, 2) @ T
        Ha = torch.einsum("bnp,nc->bpc", T, Ytr - ym)
        Ge = E.transpose(1, 2) @ E
        He = torch.einsum("bnp,nc->bpc", E, Yev - ym)
        rtre = ((Yev - ym) ** 2).sum(0)
        sse_ev = torch.zeros(b, Ytr.shape[1], device=device, dtype=T.dtype)
        for ai, a in enumerate(ALPHAS):
            beta = torch.linalg.solve(Ga + a * eye, Ha)
            se = (rtre - 2 * (beta * He).sum(1) + (beta * (Ge @ beta)).sum(1))
            sse_ev = torch.where(best_a == ai, se, sse_ev)
        sst = ((Yev - Yev.mean(0)) ** 2).sum(0) + 1e-12
        outs.append((1 - sse_ev / sst).clamp_min(0.0))
    return torch.cat(outs, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True)
    ap.add_argument("--kind", choices=("poly2", "full"), required=True)
    ap.add_argument("--eval", choices=("search", "audit"), default="audit")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--permute", type=int, default=0,
                    help=">0 则打乱 Y 的行做置换零分布(种子=该值),破坏X-Y关系、保留边际分布")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    Xtr = data["train_data"][:, gi].astype(np.float64)
    Ytr = data["train_data"][:, ci].astype(np.float64)
    Xhe = data["test_data"][:, gi].astype(np.float64)
    Yhe = data["test_data"][:, ci].astype(np.float64)

    if args.permute:
        # 置换零分布：train 与 held-out 各自独立打乱 Y 的行
        rp = np.random.RandomState(args.permute)
        Ytr = Ytr[rp.permutation(len(Ytr))]
        Yhe = Yhe[rp.permutation(len(Yhe))]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(Xhe))
    half = len(perm) // 2
    idx = perm[:half] if args.eval == "search" else perm[half:]
    Xev, Yev = Xhe[idx], Yhe[idx]

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Ztr = torch.tensor((Xtr - mu) / sd, dtype=torch.float64, device=dev)
    Zev = torch.tensor((Xev - mu) / sd, dtype=torch.float64, device=dev)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float64, device=dev)
    Yev_t = torch.tensor(Yev, dtype=torch.float64, device=dev)
    n_gen = Ztr.shape[1]

    p2 = np.random.RandomState(VAL_SEED).permutation(len(Ztr))
    nv = round(len(p2) * 0.15)
    val_idx = torch.tensor(p2[:nv], device=dev)
    fit_idx = torch.tensor(p2[nv:], device=dev)

    m = args.order
    all_sets = list(combinations(range(n_gen), m))
    mine = [s for i, s in enumerate(all_sets) if i % args.nshard == args.shard]
    log("RESCAN", "START", note=f"order{m} {args.kind} eval={args.eval} "
                                f"shard{args.shard}/{args.nshard} n={len(mine)}")
    t0 = time.perf_counter()

    # 需要的子集（m-1 阶）
    subs = sorted({tuple(sorted(t)) for S in mine for t in combinations(S, m - 1)})
    sub_pos = {s: i for i, s in enumerate(subs)}

    spec_m = term_spec(m, args.kind)
    spec_s = term_spec(m - 1, args.kind)
    v_sub = r2_batch(Ztr, Ytr_t, Zev, Yev_t,
                     torch.tensor(subs, device=dev), spec_s, fit_idx, val_idx, args.chunk)
    log("RESCAN", "SUBS", note=f"{len(subs)} 个 {m-1} 阶子集完成 {time.perf_counter()-t0:.0f}s")
    v_set = r2_batch(Ztr, Ytr_t, Zev, Yev_t,
                     torch.tensor(mine, device=dev), spec_m, fit_idx, val_idx, args.chunk)
    log("RESCAN", "SETS", note=f"{len(mine)} 个 {m} 阶完成 {time.perf_counter()-t0:.0f}s")

    v_sub_np = v_sub.cpu().numpy()
    v_set_np = v_set.cpu().numpy()
    syn, conf = np.empty(len(mine)), np.empty(len(mine), dtype=int)
    for i, S in enumerate(mine):
        rows = [sub_pos[tuple(sorted(t))] for t in combinations(S, m - 1)]
        best = v_sub_np[rows].max(axis=0)
        inc = v_set_np[i] - best
        c = int(np.argmax(inc))
        syn[i], conf[i] = inc[c], c
    df = pd.DataFrame(dict(indices=[str(s) for s in mine], syn=syn, conf=conf,
                           v_set=v_set_np.max(axis=1)))
    tag = f"_perm{args.permute}" if args.permute else ""
    out = ROOT / f"outputs/rescan_o{m}_{args.kind}_{args.eval}{tag}_s{args.shard}of{args.nshard}.parquet"
    df.to_parquet(out, index=False)
    el = time.perf_counter() - t0
    log("RESCAN", "DONE", note=f"order{m} {args.kind} {args.eval} shard{args.shard} "
                               f"{el:.0f}s max_syn={syn.max():.4f} n>0.10={(syn>0.10).sum()}")
    print(f"[o{m}/{args.kind}/{args.eval}/s{args.shard}] {len(mine)} sets {el:.0f}s "
          f"max_syn={syn.max():.4f} n>0.10={(syn>0.10).sum()} → {out.name}", flush=True)


if __name__ == "__main__":
    main()
