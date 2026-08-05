# -*- coding: utf-8 -*-
"""91 号 ★E3：三阶 syn 对**无偏**重训真值——full≠poly2 的地方，也是 90 号发现字典效应之处。

E2（二阶）的 full 与 poly2 恒等（m=2 时最高阶就是两两乘积），
所以"学出来的特征 vs 手工全交互字典"必须到三阶才分得开。

真值用 77 号 `h2_unbiased_pool.csv`（2000 个**随机抽取**的三元组重训 syn3）。
选它而非 68 号 `triples_certified.csv`，是因为后者由 top-K 筛选而来、带选择偏差
（73 号总结里已记过这条），而无偏池才能支持"密度/召回"一类的总体主张。
⚠ 同 E2：该真值出自 67/68 号 worker，88 号查出其早停用了测试集，绝对值偏乐观；
本号只用它做排序参照，且所有方法面对同一份参照。
"""
from __future__ import annotations

import argparse
import ast
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
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, poly2_block, ridge_r2)
from runlog import log  # noqa: E402

CKPT = REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt"
POOL = REPO / "DNN_Aggresvation77/outputs/h2_unbiased_pool.csv"
BANDS = [-9, 0.05, 0.10, 0.20, 0.35, 9]
BAND_LAB = ["<.05", ".05-.1", ".1-.2", ".2-.35", ">.35"]


def full_block(Z, sets):
    """poly2 + 3..m 阶乘积（= 90 号 full 字典，⊃ poly2）。"""
    B, m = sets.shape
    raw = Z[:, sets.reshape(-1)].T.reshape(B, m, -1).transpose(1, 2)
    cols = [poly2_block(Z, sets)]
    for r in range(3, m + 1):
        for c in combinations(range(m), r):
            prod = raw[:, :, c[0]]
            for j in c[1:]:
                prod = prod * raw[:, :, j]
            cols.append(prod.unsqueeze(2))
    return torch.cat(cols, dim=2)


@torch.no_grad()
def oracle_r2(model, X, Y, sets_list, n_gen, sst, chunk=512):
    out = []
    for S in sets_list:
        mask = torch.zeros(X.shape[0], n_gen, device=X.device)
        mask[:, list(S)] = 1.0
        pred = model(X, mask).double()
        out.append((1 - ((pred - Y) ** 2).sum(0) / sst).clamp_min(0.0).cpu().numpy())
    _ = chunk
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="oracle,poly2,full,last,last+p2,cat3")
    ap.add_argument("--chunk", type=int, default=64)
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
    Ztr = torch.tensor((Xtr_np - mu) / sd, dtype=torch.float64, device=dev)
    Zev = torch.tensor((Xev_np - mu) / sd, dtype=torch.float64, device=dev)
    Ytr = torch.tensor(Ytr_np, dtype=torch.float64, device=dev)
    Yev = torch.tensor(Yev_np, dtype=torch.float64, device=dev)
    sst = ((Yev - Yev.mean(0)) ** 2).sum(0) + 1e-12
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)

    # ---- 真值：无偏随机三元组池 ----
    pool = pd.read_csv(POOL)
    pool = pool[pool["group"] == "triple_rand"].copy()
    cmap = {c: k for k, c in enumerate(conf_names)}
    assert not set(pool["conf"].unique()) - set(cmap)
    pool["ck"] = pool["conf"].map(cmap)
    pool["S"] = pool["ix"].map(lambda s: tuple(sorted(ast.literal_eval(s))))
    triples = sorted(set(pool["S"]))
    pairs = sorted({tuple(sorted(t)) for S in triples for t in combinations(S, 2)})
    tri_pos = {s: i for i, s in enumerate(triples)}
    pair_pos = {s: i for i, s in enumerate(pairs)}
    log("E3", "START", note=f"{len(triples)} 无偏三元组 × {len(pool)} 条真值，"
                            f"{len(pairs)} 个二阶子集，audit={len(a_idx)} 行")

    rows, dumps = [], []
    for kind in args.kinds.split(","):
        t0 = time.perf_counter()
        if kind == "oracle":
            model = load_oracle(CKPT, n_gen, n_conf, dev)
            v_p = oracle_r2(model, Xev, Yev, pairs, n_gen, sst)
            v_t = oracle_r2(model, Xev, Yev, triples, n_gen, sst)
        else:
            phi = None
            if kind.startswith(("last", "cat3")):
                phi = FrozenPhi(load_oracle(CKPT, n_gen, n_conf, dev),
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
            v_p, v_t = ev(pairs), ev(triples)

        est = np.array([v_t[tri_pos[r.S], r.ck]
                        - max(v_p[pair_pos[tuple(sorted(t))], r.ck]
                              for t in combinations(r.S, 2))
                        for r in pool.itertuples()])
        truth = pool["syn3_true"].to_numpy()
        rho = float(spearmanr(est, truth).statistic)
        top = np.argsort(-truth)[:100]
        rho_tail = float(spearmanr(est[top], truth[top]).statistic)
        strong = truth > 0.2
        order = np.argsort(-est)
        rank_of = np.empty(len(est), dtype=int)
        rank_of[order] = np.arange(len(est))
        row = dict(kind=kind, n=len(est), rho=round(rho, 4),
                   rho_tail=round(rho_tail, 4),
                   hit50=round(float(strong[order[:50]].sum()
                                     / max(min(50, strong.sum()), 1)), 3),
                   med_rank_true_top10=float(np.median(rank_of[np.argsort(-truth)[:10]])),
                   n_strong=int(strong.sum()), sec=round(time.perf_counter() - t0, 1))
        band = pd.cut(truth, BANDS, labels=BAND_LAB)
        for b in BAND_LAB:
            sel = band == b
            row[f"gap_{b}"] = round(float((truth[sel] - est[sel]).mean()), 4) if sel.sum() else np.nan
        rows.append(row)
        dumps.append(pd.DataFrame(dict(kind=kind, S=pool["ix"].values,
                                       conf=pool["conf"].values, est=est, truth=truth)))
        log("E3", "DONE", note=f"{kind} rho={rho:.4f} tail={rho_tail:.4f} "
                               f"{time.perf_counter()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/order3_truth.csv", index=False)
    pd.concat(dumps).to_csv(ROOT / "outputs/order3_triples.csv", index=False)
    pd.set_option("display.width", 260)
    print("\n=== E3：三阶 syn 对无偏重训真值（audit；oracle 行 = 当前方案基线）===")
    print(df[["kind", "n", "rho", "rho_tail", "hit50",
              "med_rank_true_top10", "n_strong", "sec"]].to_string(index=False))
    print("\n=== 各真值档的低估量 gap = truth − est（越小越好）===")
    print(df[["kind"] + [f"gap_{b}" for b in BAND_LAB]].to_string(index=False))


if __name__ == "__main__":
    main()
