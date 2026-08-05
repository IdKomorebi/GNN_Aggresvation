#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""97 号：降低重训认证协议的噪声底——判定 k=6 到底可不可认证。

为什么做这个而不是继续推 k=7
----------------------------
灵敏度分析（见 WORKLOG「两条曲线的交叉」）：可检测下限 = 2×噪声底 ÷ 回收率。

  PJM   k=5 下限 0.182 vs 外推真实最强 0.400（余量 2.2×，可认证）
        k=6 下限 0.228 vs 0.314（余量 1.4×，**边缘**）
        k=7 下限 0.974 vs 0.247（余量 0.25×，无望）
  CAISO k=6 已 0.9×（掉到线下），k=7 0.4×

k=7 差 3-4 倍，且瓶颈在真值端（受样本量限制），加算力无用。
唯独 k=6 只差一点点——只要把噪声底降下来就能给出明确判决。这是最后一个没动过的杠杆。

三处改动（可独立开关，便于归因）
--------------------------------
1. **配对选择**（--paired）：现协议在 audit 片上同时做两次 max（选机密字段 c、
   选最强子集 T）再报增量，是**系统性正偏**的来源——对照组认证均值 0.0315
   而标准差仅 0.0154，说明那个"噪声底"大半是偏差不是方差。
   改为：c* 与 T* 都在 **search 片**选定，只在 **audit 片**报增量。
   （两片来自 SPLIT_SEED=880725 对 held-out 测试集的对半划分，口径与全项目一致。）
2. **多种子平均**（--seeds）：syn 取 R 个种子的均值，方差 → σ/√R；
   种子间标准差直接给出**该集合的噪声底**，用于判断认证值是否显著。
3. **公共随机数**：S 与其子集用同一 seed 初始化与批序（现协议已有，此处保留并显式记录），
   使 v(S) 与 v(T) 的噪声正相关，差分方差进一步下降。

★同一批重训同时输出**新旧两个统计量**，故偏差消除了多少可以直接读出，
无需额外算力。

硬闸门（预注册，见 96 号计划阶段 B）
------------------------------------
先在 k=5 上跑：那里已有 93 号认证为真的 10 个协同。新协议必须复现它们，
否则说明改坏了，停下查错，**不得**用于 k=6 判决。

★红线：重训真值只用于评估与认证，绝不进入扫描/路由/校准路径。
search 片只用于**认证内部的选择**，不回流到估计器。
"""
from __future__ import annotations

import argparse
import ast
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
R96 = REPO / "DNN_Aggresvation96"
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import SPLIT_SEED  # noqa: E402
from runlog import log  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 120          # 与 88/93/96 号认证协议逐字一致
PICK_SEED = 930726


class DNN(nn.Module):
    """逐字沿用 88 号 worker 结构（hidden=128, 2 隐层, dropout 0.15）。"""

    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_generic(model, Xtr, Ytr):
    """88 号修正版：早停用训练集内部 15% val，测试集完全不参与。"""
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


def load_sets(args) -> pd.DataFrame:
    """两种取集合方式：--from_csv 复用已认证过的集合（k=5 闸门用）；否则读扫描产出。"""
    if args.from_csv:
        fs = sorted(glob.glob(args.from_csv))
        assert fs, f"找不到 {args.from_csv}"
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d["S"] = d["S"].map(ast.literal_eval)
        if "group" not in d:
            d["group"] = "strong"
        keep = ["S", "group"] + [c for c in ("syn_struct", "syn_true_audit") if c in d]
        return d[keep].rename(columns={"syn_true_audit": "syn_old_protocol"})
    fs = sorted(glob.glob(str(R96 / f"outputs/{args.source}_s*of*.parquet")))
    assert fs, f"找不到扫描结果 {args.source}"
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["S"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    strong = d.nlargest(args.n_top, "syn").assign(group="strong")
    ctrl = d[d.syn < 0.05].sample(args.n_ctrl, random_state=PICK_SEED).assign(group="control")
    out = pd.concat([strong, ctrl], ignore_index=True)
    return out.rename(columns={"syn": "syn_struct"})[["S", "group", "syn_struct"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True)
    ap.add_argument("--source", default=None, help="scan_highorder 产出前缀")
    ap.add_argument("--from_csv", default=None, help="改从已有认证 CSV 取集合（闸门用）")
    ap.add_argument("--seeds", default="0,1,2", help="重训种子；均值降方差，标准差给噪声底")
    ap.add_argument("--n_top", type=int, default=10)
    ap.add_argument("--n_ctrl", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    cfg = yaml.safe_load((R96 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    tr, te = di["train_data"], di["test_data"]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(te))
    s_idx, a_idx = perm[:len(perm) // 2], perm[len(perm) // 2:]   # search / audit

    picks = load_sets(args)
    mine = picks.iloc[args.shard::args.nshard].reset_index(drop=True)
    out_csv = ROOT / f"outputs/paired_o{args.order}{args.tag}_s{args.shard}of{args.nshard}.csv"
    done = set()
    for f in glob.glob(str(ROOT / f"outputs/paired_o{args.order}{args.tag}_s*of*.csv")):
        try:
            done |= set(pd.read_csv(f)["S"].tolist())
        except Exception:
            pass
    log("PAIR", "START", note=f"order-{args.order} shard{args.shard}/{args.nshard} "
                              f"{len(mine)} 个集合 seeds={seeds} 已完成 {len(done)}")

    Ytr_t = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Yte_s, Yte_a = te[s_idx][:, ci], te[a_idx][:, ci]

    def v_of(sel, seed):
        """重训一个专用 DNN，返回 (search 片 R², audit 片 R²)，各 12 维。

        公共随机数：同一 seed ⟹ S 与其子集共享初始化与批序，噪声正相关。
        """
        cols = gi[list(sel)]
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xtr = torch.as_tensor(tr[:, cols], dtype=torch.float32, device=DEV)
        Xte = torch.as_tensor(te[:, cols], dtype=torch.float32, device=DEV)
        model = train_generic(DNN(len(sel), len(ci)).to(DEV), Xtr, Ytr_t)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()
        return per_conf_r2(pred[s_idx], Yte_s), per_conf_r2(pred[a_idx], Yte_a)

    t0 = time.perf_counter()
    for n_done, r in enumerate(mine.itertuples(), 1):
        if str(tuple(r.S)) in done:
            continue
        S = tuple(r.S)
        subs = list(combinations(S, args.order - 1))
        paired, naive = [], []
        for sd in seeds:
            vS_s, vS_a = v_of(S, sd)
            sub = [v_of(T, sd) for T in subs]
            sub_s = np.stack([x[0] for x in sub])
            sub_a = np.stack([x[1] for x in sub])
            # ---- 配对：c* 与 T* 都在 search 片选定，只在 audit 片报增量 ----
            inc_s = vS_s - sub_s.max(0)
            c_star = int(np.argmax(inc_s))
            t_star = int(np.argmax(sub_s[:, c_star]))
            paired.append(float(vS_a[c_star] - sub_a[t_star, c_star]))
            # ---- 旧口径：audit 片上两次 max（同一批重训，零额外算力）----
            naive.append(float((vS_a - sub_a.max(0)).max()))
        row = dict(S=str(S), group=r.group,
                   syn_struct=float(getattr(r, "syn_struct", np.nan)),
                   syn_paired=float(np.mean(paired)),
                   syn_paired_sd=float(np.std(paired, ddof=1)) if len(seeds) > 1 else 0.0,
                   syn_naive=float(np.mean(naive)),
                   syn_naive_sd=float(np.std(naive, ddof=1)) if len(seeds) > 1 else 0.0,
                   n_seed=len(seeds), c_star_last=c_star)
        if hasattr(r, "syn_old_protocol"):
            row["syn_old_protocol"] = float(r.syn_old_protocol)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(),
                                   index=False)
        el = time.perf_counter() - t0
        print(f"  [{n_done}/{len(mine)}] {S} {r.group} "
              f"配对={row['syn_paired']:.4f}±{row['syn_paired_sd']:.4f} "
              f"旧口径={row['syn_naive']:.4f} ({el:.0f}s)", flush=True)

    log("PAIR", "DONE", note=f"shard{args.shard} {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
