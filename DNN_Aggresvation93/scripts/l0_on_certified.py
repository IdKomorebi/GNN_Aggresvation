# -*- coding: utf-8 -*-
"""93 号 任务A 延伸：★在**已认证的五阶集合**上比各估计器，看谁少犯假阳性。

任务A 拿到重训真值后，就第一次有了在**五阶**上比较估计器的裁判
（91 号 E2/E3 只到二阶/三阶，因为更高阶此前没有真值）。
这批集合还特别难——它们是 full 字典判定的"最强高阶协同"，是假阳性最集中的地方。

对每个已认证的五元组 S，各估计器算 `syn5 = max_c[v(S)_c − max_{T⊂S,|T|=4} v(T)_c]`：
    poly2 / full   手工字典（90 号用的）
    last / cat3    冻结 oracle 特征 + 闭式读出（91 号方案）
再与 `syn_true_audit`（专用 DNN 重训）对比。

关键指标不是相关系数（强组被选择过，相关性会被截断），而是：
  · **虚高倍数** = 估计 syn / 真值 syn；
  · **误报数** = 真值 syn<0.1 却被估计成 >0.2 的个数。
"""
from __future__ import annotations

import argparse
import ast
import glob
import sys
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
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, poly2_block, ridge_r2)
from runlog import log  # noqa: E402

CKPT = REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt"


def full_block(Z, sets):
    B, m = sets.shape
    raw = Z[:, sets.reshape(-1)].T.reshape(B, m, -1).transpose(1, 2)
    cols = [poly2_block(Z, sets)]
    for r in range(3, m + 1):
        for c in combinations(range(m), r):
            p = raw[:, :, c[0]]
            for j in c[1:]:
                p = p * raw[:, :, j]
            cols.append(p.unsqueeze(2))
    return torch.cat(cols, dim=2)


@torch.no_grad()
def oracle_r2(model, X, Y, sl, n_gen, sst):
    out = []
    for S in sl:
        m = torch.zeros(X.shape[0], n_gen, device=X.device)
        m[:, list(S)] = 1.0
        out.append((1 - ((model(X, m).double() - Y) ** 2).sum(0) / sst)
                   .clamp_min(0.0).cpu().numpy())
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="oracle,poly2,full,last,cat3")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

    fs = sorted(glob.glob(str(ROOT / "outputs/certify_o5_s*of*.csv")))
    cert = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    cert["Stup"] = cert["S"].map(lambda s: tuple(ast.literal_eval(s)))
    sets = list(cert["Stup"])
    subs = sorted({tuple(sorted(t)) for S in sets for t in combinations(S, 4)})
    sub_pos = {s: i for i, s in enumerate(subs)}

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
    sst = ((Yev - Yev.mean(0)) ** 2).sum(0) + 1e-12
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)
    log("L0CERT", "START", note=f"{len(sets)} 个已认证五元组 + {len(subs)} 个四阶子集")

    truth = cert["syn_true_audit"].to_numpy()
    rows, per_set = [], {"S": cert["S"].values, "group": cert["group"].values,
                         "artifact": cert["artifact"].values,
                         "syn_true": truth, "syn_struct_90": cert["syn_struct"].values}
    for kind in args.kinds.split(","):
        if kind == "oracle":
            model = load_oracle(ckpt, n_gen, n_conf, dev)
            v_sub = oracle_r2(model, Xev, Yev, subs, n_gen, sst)
            v_set = oracle_r2(model, Xev, Yev, sets, n_gen, sst)
        else:
            phi = None
            if kind.startswith(("last", "cat3")):
                phi = FrozenPhi(load_oracle(ckpt, n_gen, n_conf, dev),
                                "cat3" if kind.startswith("cat3") else "last")

            def ev(sl):
                out = []
                for s in range(0, len(sl), args.chunk):
                    sb = torch.as_tensor(sl[s:s + args.chunk], device=dev)
                    if kind == "full":
                        F_tr, F_ev = full_block(Ztr, sb), full_block(Zev, sb)
                    else:
                        F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, kind)
                        F_ev = build_features(phi, Xev, Zev, sb, n_gen, kind)
                    out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
                return np.concatenate(out, 0)
            v_sub, v_set = ev(subs), ev(sets)

        est = np.array([(v_set[i] - v_sub[[sub_pos[tuple(sorted(t))]
                                           for t in combinations(S, 4)]].max(0)).max()
                        for i, S in enumerate(sets)])
        per_set[f"syn_{kind}"] = est
        weak = truth < 0.1
        rows.append(dict(kind=kind,
                         mean_est=round(float(est.mean()), 4),
                         mean_true=round(float(truth.mean()), 4),
                         虚高倍数=round(float(est.mean() / max(truth.mean(), 1e-9)), 2),
                         rho=round(float(spearmanr(est, truth).statistic), 4),
                         误报_真弱估强=int((weak & (est > 0.2)).sum()),
                         真弱数=int(weak.sum()),
                         est过0_2=int((est > 0.2).sum()),
                         真值过0_2=int((truth > 0.2).sum())))
        log("L0CERT", "DONE", note=f"{kind} mean_est={est.mean():.4f}")

    df = pd.DataFrame(rows)
    pd.DataFrame(per_set).to_csv(ROOT / "outputs/l0_on_certified_detail.csv", index=False)
    df.to_csv(ROOT / "outputs/l0_on_certified.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== 在 {len(sets)} 个已认证五元组上比各估计器（真值 = 专用 DNN 重训）===")
    print(df.to_string(index=False))
    print("\n[读法] 虚高倍数越接近 1 越好；误报 = 真值 syn<0.1 却被估成 >0.2 的个数")


if __name__ == "__main__":
    main()
