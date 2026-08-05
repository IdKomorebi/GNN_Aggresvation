# -*- coding: utf-8 -*-
"""91 号 ★★E4 核心诊断：amortization gap 到底在**读出层**还是**特征层**？

本号的整个假设是"共享的线性读出是瓶颈，特征层已经够用"。
这个假设可以被一个实验直接判定——把同一批集合放在四种处理下比较：

  oracle      共享读出，零成本               （当前方案）
  L0          冻结特征 + **闭式重解读出**     （本号方案：一次前向 + 一个 solve）
  ft-K        微调 K 步（特征与读出都动）     （现行的"逐集合细化"）
  ft-K + L0   微调 K 步后**再**闭式重解读出   （看两者是否叠加）

判据（预注册，三种结果都算数）：
  · L0 ≈ ft-25  ⟹ **闭式读出可替代微调**，用户要的"比微调更高效"成立；
  · ft-25 ≫ L0  ⟹ **特征层才是瓶颈**，"读出层假说"被证伪（则 L0 必须换成 L1 重训）；
  · ft-25+L0 > ft-25 ⟹ 闭式读出是微调的**免费增强**（微调没把读出层调到该层最优）。

微调协议逐字沿用 75/76 号（BATCH 256 / Adam lr 1e-3 wd 5e-4 / 同种子），
评估一律在 audit 分片（88 号口径）。真值取 68 号逐 (i,j,conf) 重训 synergy。
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
from scipy.stats import spearmanr

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
TRUTH = REPO / "DNN_Aggresvation68/outputs/synergy2_perconf.csv"
BATCH, LR, WD = 256, 1e-3, 5e-4          # 与 75/76 号逐项一致
KGRID = (5, 25, 50)
SEED = 0


@torch.no_grad()
def r2_of(model, X, Y, sel, n_gen, sst):
    """oracle 共享读出的逐 conf R²（一个集合）。"""
    mask = torch.zeros(X.shape[0], n_gen, device=X.device)
    mask[:, sel] = 1.0
    pred = model(X, mask).double()
    return (1 - ((pred - Y) ** 2).sum(0) / sst).clamp_min(0.0).cpu().numpy()


def closed_form(model, kind, Xtr, Xev, Ztr, Zev, Ytr, Yev, sel, n_gen, fit_i, val_i):
    """闭式重解读出：冻结给定 model 的主干，取激活作 φ，逐 (S,c) 解岭回归。"""
    k0 = "cat3" if kind.startswith("cat3") else "last"
    phi = FrozenPhi(model, k0)
    sb = torch.as_tensor([sel], device=Ztr.device)
    F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, kind)
    F_ev = build_features(phi, Xev, Zev, sb, n_gen, kind)
    return ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="cat3+p2")
    ap.add_argument("--n_band", type=int, default=40, help="强/中/弱各取多少 pair")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi, ci = np.asarray(data["general_indices"]), np.asarray(data["confidential_indices"])
    Xtr_np = data["train_data"][:, gi].astype(np.float64)
    Ytr_np = data["train_data"][:, ci].astype(np.float64)
    Xhe_np = data["test_data"][:, gi].astype(np.float64)
    Yhe_np = data["test_data"][:, ci].astype(np.float64)
    n_gen, n_conf = Xtr_np.shape[1], Ytr_np.shape[1]
    conf_names = list(data["confidential"])

    perm = np.random.RandomState(SPLIT_SEED).permutation(len(Xhe_np))
    a_idx = perm[len(perm) // 2:]
    Xev_np, Yev_np = Xhe_np[a_idx], Yhe_np[a_idx]

    mu, sd = Xtr_np.mean(0), Xtr_np.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=dev)
    Xev = torch.tensor(Xev_np, dtype=torch.float32, device=dev)
    Ytr_f = torch.tensor(Ytr_np, dtype=torch.float32, device=dev)
    Ztr = torch.tensor((Xtr_np - mu) / sd, dtype=torch.float64, device=dev)
    Zev = torch.tensor((Xev_np - mu) / sd, dtype=torch.float64, device=dev)
    Ytr = torch.tensor(Ytr_np, dtype=torch.float64, device=dev)
    Yev = torch.tensor(Yev_np, dtype=torch.float64, device=dev)
    sst = ((Yev - Yev.mean(0)) ** 2).sum(0) + 1e-12
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)

    # ---- 选集合：按真值 synergy 分强/中/弱三档，各取 n_band 个 pair ----
    tr = pd.read_csv(TRUTH)
    cmap = {c: k for k, c in enumerate(conf_names)}
    assert not set(tr["conf"].unique()) - set(cmap)
    tr["ck"] = tr["conf"].map(cmap)
    best = tr.loc[tr.groupby(["i", "j"])["synergy"].idxmax()].sort_values(
        "synergy", ascending=False).reset_index(drop=True)
    nb, N = args.n_band, len(best)
    picks = pd.concat([best.iloc[:nb],
                       best.iloc[N // 2 - nb // 2: N // 2 - nb // 2 + nb],
                       best.iloc[-nb:]]).drop_duplicates(subset=["i", "j"])
    bands = ["strong"] * nb + ["mid"] * nb + ["weak"] * nb
    picks = picks.assign(band=bands[:len(picks)])
    log("E4", "START", note=f"{len(picks)} 个 pair（强/中/弱各 ~{nb}），kind={args.kind}，"
                            f"K∈{KGRID}，audit={len(a_idx)} 行")

    # 单字段 v̂ 需与 pair 同处理，才能算 syn ⟹ 收集所有涉及的单字段
    singles = sorted({int(v) for r in picks.itertuples() for v in (r.i, r.j)})

    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    state = ck["state"] if "state" in ck else ck
    base = load_oracle(CKPT, n_gen, n_conf, dev)

    def all_treatments(sel):
        """返回 dict: 处理名 → (nC,) 的 R²。"""
        out = {"oracle": r2_of(base, Xev, Yev, sel, n_gen, sst),
               "L0": closed_form(base, args.kind, Xtr, Xev, Ztr, Zev,
                                 Ytr, Yev, sel, n_gen, fit_i, val_i)}
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = MLPOracle(n_gen, n_conf)
        model.load_state_dict(state)
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        mask_tr = torch.zeros(len(Xtr), n_gen, device=dev)
        mask_tr[:, sel] = 1.0
        rng = np.random.RandomState(SEED)
        step = 0
        while step < max(KGRID):
            order = rng.permutation(len(Xtr))
            for b in range(0, len(order), BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=dev)
                model.train()
                opt.zero_grad()
                loss = ((model(Xtr[ix], mask_tr[ix]) - Ytr_f[ix]) ** 2).mean()
                loss.backward()
                opt.step()
                step += 1
                if step in KGRID:
                    model.eval()
                    out[f"ft{step}"] = r2_of(model, Xev, Yev, sel, n_gen, sst)
                    out[f"ft{step}+L0"] = closed_form(
                        model, args.kind, Xtr, Xev, Ztr, Zev, Ytr, Yev,
                        sel, n_gen, fit_i, val_i)
                if step >= max(KGRID):
                    break
        return out

    t0 = time.perf_counter()
    v_single = {i: all_treatments([i]) for i in singles}
    log("E4", "SINGLES", note=f"{len(singles)} 个单字段完成 {time.perf_counter()-t0:.0f}s")

    rows = []
    for n_done, r in enumerate(picks.itertuples(), 1):
        vp = all_treatments([int(r.i), int(r.j)])
        for name, arr in vp.items():
            c = int(r.ck)
            syn = arr[c] - max(v_single[int(r.i)][name][c], v_single[int(r.j)][name][c])
            rows.append(dict(i=int(r.i), j=int(r.j), conf=r.conf, band=r.band,
                             method=name, v_pair=float(arr[c]), syn=float(syn),
                             truth=float(r.synergy)))
        if n_done % 20 == 0:
            print(f"  {n_done}/{len(picks)} [{time.perf_counter()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/readout_vs_feature.csv", index=False)

    # ---- 汇总 ----
    summ = []
    for name, g in df.groupby("method"):
        row = dict(method=name,
                   rho=round(float(spearmanr(g.syn, g.truth).statistic), 4),
                   mae=round(float(np.abs(g.syn - g.truth).mean()), 4))
        for band in ("strong", "mid", "weak"):
            gb = g[g.band == band]
            row[f"syn_{band}"] = round(float(gb.syn.mean()), 4)
            row[f"truth_{band}"] = round(float(gb.truth.mean()), 4)
            row[f"gap_{band}"] = round(float((gb.truth - gb.syn).mean()), 4)
        summ.append(row)
    order = ["oracle", "L0"] + [f"ft{k}{s}" for k in KGRID for s in ("", "+L0")]
    sm = pd.DataFrame(summ).set_index("method").reindex(order).reset_index()
    sm.to_csv(ROOT / "outputs/readout_vs_feature_summary.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n=== ★E4：读出层 vs 特征层（syn 对 68 号重训真值）===")
    print(sm.to_string(index=False))
    print("\n[判据] L0≈ft25 ⟹ 闭式读出可替代微调；ft25≫L0 ⟹ 特征层才是瓶颈；"
          "ft25+L0>ft25 ⟹ 闭式读出是微调的免费增强")
    log("E4", "DONE", note=f"总用时 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
