#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""93 号 任务A：★用 DNN 重训认证 90 号 full 字典发现的强高阶——补上 91 号 §5.1 的唯一缺口。

背景
----
90 号换 full 字典后 order-5 的 `syn>0.10` 从 272 涨到 6714、`syn>0.2` 有 441 个，
并据此推翻了 87 号的"衰减律"。但那全部是**结构化查询自身算出的 syn**，没有外部真值。
91 号 E3 首次拿重训真值校验字典，发现 full 相对 poly2 没换来更好的排序、尾部反而更差——
可惜无偏池 syn3 最高只到 0.2，覆盖不到 90 号说的那一档。本脚本正面补这个缺口。

设计
----
· 目标组：order-5 full 的 `syn>0.2` 里按 syn 降序取 N_TOP 个；
· 对照组：从 `syn<0.05` 里随机抽 N_CTRL 个（同一份数据、同一流程，看认证后是否分得开）；
· 每个五元组重训 1(自身) + 5(四阶子集) = 6 个专用 DNN，得
  `syn5_true = max_c [ v(S)_c − max_{T⊂S,|T|=4} v(T)_c ]`；
· 重训口径**逐字沿用 88 号修正版 worker**（训练集内部 val 早停，测试集不参与），
  数据切分 seed 42；同时报 audit 分片（与 90 号可比）与全 test（与 67/68/77 既有真值可比）。

★标记 `syn==v_set` 的伪影：这类集合的全部四阶子集 v 被 clamp 到 0，
syn 退化成 v(S) 本身。全体只占 0.03%，但**恰好聚在 top 榜单**（441 个里 10 个），
故单列一栏，认证结果分开看。

断点续跑：逐集合追加写 CSV，重跑自动跳过已完成。
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
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import SPLIT_SEED  # noqa: E402
from runlog import log  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 120          # 与 67/68/77/88 号一致
N_TOP, N_CTRL = 40, 20
PICK_SEED = 930726


class DNN(nn.Module):
    """逐字沿用 88 号 worker 的结构（hidden=128, 2 隐层, dropout 0.15）。"""

    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_generic(model, Xtr, Ytr):
    """88 号修正版：早停用**训练集内部** validation，测试集完全不参与。"""
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


def load_rescan(order: int, kind: str, source: str | None = None) -> pd.DataFrame:
    """95_caiso：一律读本号 scan_order5 的产出前缀（--source 必填，无 90 号 PJM 回退）。"""
    assert source, "CAISO 复刻必须显式传 --source（scan_order5 产出前缀）"
    fs = sorted(glob.glob(str(ROOT / f"outputs/{source}_s*of*.parquet")))
    _ = order, kind
    assert fs, f"找不到扫描结果：source={source}"
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["S"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="重训随机性")
    ap.add_argument("--source", default=None,
                    help="读本号 scan_order5 的产出前缀（不给则读 90 号 full 重扫）")
    ap.add_argument("--n_top", type=int, default=N_TOP)
    ap.add_argument("--n_ctrl", type=int, default=N_CTRL)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/caiso_2025_hourly_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)                       # 数据切分口径与既有真值一致
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    tr, te = di["train_data"], di["test_data"]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(te))
    a_idx = perm[len(perm) // 2:]            # audit 分片，与 90 号同口径

    # ---- 选集合 ----
    d = load_rescan(args.order, "full", args.source)
    d["artifact"] = np.isclose(d.syn, d.v_set, atol=1e-9)
    # 90 号 full 的强组按 syn>0.2 取；本号 L1 扫出来一个 syn>0.2 都没有，
    # 故直接取 top-N（"该估计器自己认为最强的那些"），这正是要认证的对象。
    pool_strong = d[d.syn > 0.2] if (d.syn > 0.2).sum() >= args.n_top else d
    strong = pool_strong.nlargest(args.n_top, "syn").assign(group="strong")
    picks = [strong]
    if args.n_ctrl:
        ctrl = d[d.syn < 0.05].sample(args.n_ctrl, random_state=PICK_SEED)
        picks.append(ctrl.assign(group="control"))
    picks = pd.concat(picks, ignore_index=True)
    mine = picks.iloc[args.shard::args.nshard].reset_index(drop=True)

    out_csv = ROOT / (f"outputs/certify_o{args.order}{args.tag}"
                      f"_s{args.shard}of{args.nshard}.csv")
    # 断点续跑：只在**同 tag** 的分片文件内识别已完成（95_caiso 修正：原 glob 不含
    # tag，同 order 换 source 重跑时会把别组的集合当成已完成而静默跳过）。
    done = set()
    for f in glob.glob(str(ROOT / f"outputs/certify_o{args.order}{args.tag}_s*of*.csv")):
        try:
            done |= set(pd.read_csv(f)["S"].tolist())
        except Exception:
            pass
    log("CERT", "START", note=f"order-{args.order} shard{args.shard}/{args.nshard}: "
                              f"{len(mine)} 个集合（strong {int((mine.group=='strong').sum())} / "
                              f"control {int((mine.group=='control').sum())}），已完成 {len(done)}")

    Ytr_t = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Yte_full = te[:, ci]
    Yte_audit = te[a_idx][:, ci]

    def v_of(sel_local):
        """重训一个专用 DNN，返回 (audit R², full-test R²)，各 12 维。"""
        cols = gi[list(sel_local)]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        Xtr = torch.as_tensor(tr[:, cols], dtype=torch.float32, device=DEV)
        Xte = torch.as_tensor(te[:, cols], dtype=torch.float32, device=DEV)
        model = train_generic(DNN(len(sel_local), len(ci)).to(DEV), Xtr, Ytr_t)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()
        return per_conf_r2(pred[a_idx], Yte_audit), per_conf_r2(pred, Yte_full)

    t0 = time.perf_counter()
    for n_done, r in enumerate(mine.itertuples(), 1):
        if str(r.S) in done:
            continue
        S = tuple(r.S)
        vS_a, vS_f = v_of(S)
        subs = list(combinations(S, args.order - 1))
        sub_a = np.stack([v_of(T)[0] for T in subs])
        sub_f = np.stack([v_of(T)[1] for T in subs])
        inc_a, inc_f = vS_a - sub_a.max(0), vS_f - sub_f.max(0)
        ca, cf = int(np.argmax(inc_a)), int(np.argmax(inc_f))
        row = dict(S=str(S), group=r.group, artifact=bool(r.artifact),
                   syn_struct=float(r.syn), conf_struct=int(r.conf),
                   v_struct=float(r.v_set),
                   syn_true_audit=float(inc_a[ca]), conf_true_audit=ca,
                   v_true_audit=float(vS_a.max()), maxsub_true_audit=float(sub_a.max()),
                   syn_true_full=float(inc_f[cf]), conf_true_full=cf,
                   v_true_full=float(vS_f.max()),
                   syn_at_struct_conf=float(inc_a[int(r.conf)]))
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(),
                                   index=False)
        el = time.perf_counter() - t0
        print(f"  [{n_done}/{len(mine)}] {S} {r.group} "
              f"syn_struct={r.syn:.3f} → syn_true={inc_a[ca]:.3f} "
              f"({el:.0f}s)", flush=True)
        if n_done % 10 == 0:
            log("CERT", "PROGRESS", note=f"shard{args.shard} {n_done}/{len(mine)} {el:.0f}s")

    log("CERT", "DONE", note=f"shard{args.shard} 用时 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
