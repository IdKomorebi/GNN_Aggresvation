# -*- coding: utf-8 -*-
"""91 号 ★E2：真实 conf 上的二阶 syn 对重训真值——这才是决定方案实用价值的实验。

E1（注入）测的是 φ 的**通用表达力**，对 L0 偏严：φ 是为预测真实 12 个 conf 学的，
注入的合成 y 与它们无关。E2 测的是**方案实际要用的场景**：真实 conf 的协同发现。

对照组（全部同一 train 拟合 / 同一 audit 评估）：
  oracle   — 现有 oracle 直接前向算 R²（**共享读出**，即当前方案，我要打败的基线）
  poly2    — 90 号手工字典 + 闭式读出
  full     — poly2 + 3..m 阶乘积项
  last/cat3(+p2) — **冻结 oracle 特征 + 闭式读出**（本号方案 L0）

真值：68 号 `synergy2_perconf.csv` 的逐 (i,j,conf) 重训 synergy。
⚠ 诚实标注：该真值由 67/68 号 worker 产生，而 88 号查出那个 worker 用**测试集早停**，
故其绝对值偏乐观。本号只用它做**排序参照**，且所有方法面对同一份参照，比较是公平的。
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
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, ridge_r2)
from runlog import log  # noqa: E402

CKPT = REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt"
TRUTH = REPO / "DNN_Aggresvation68/outputs/synergy2_perconf.csv"


def full_block2(Z, sets):
    """二阶时 full 与 poly2 只差…没有差（m=2 的最高阶就是两两乘积），故 E2 不含 full。"""
    raise NotImplementedError


@torch.no_grad()
def oracle_r2(model, X, Y, sets_list, n_gen, affine=False, Xtr=None, Ytr=None):
    """现有方案：oracle 直接前向 → 逐 conf R²（共享读出，无闭式重解）。

    affine=True 时先做**逐 (S,c) 的仿射校准** a·pred+b（a,b 用 train 最小二乘拟合）。
    这是必要的对照：闭式读出比 oracle 好，可能只是因为 oracle 的输出有系统性尺度/偏置漂移。
    若 2 参数的仿射校准就能补上大部分差距，"共享读出是瓶颈"的论点就该削弱；
    若补不上、必须 256 维闭式重解才行，论点才成立。
    """
    sst = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    out = []
    for S in sets_list:
        mask = torch.zeros(X.shape[0], n_gen, device=X.device)
        mask[:, list(S)] = 1.0
        pred = model(X, mask).double()
        if affine:
            mtr = torch.zeros(Xtr.shape[0], n_gen, device=X.device)
            mtr[:, list(S)] = 1.0
            p_tr = model(Xtr, mtr).double()
            pm, ym = p_tr.mean(0), Ytr.mean(0)
            cov = ((p_tr - pm) * (Ytr - ym)).sum(0)
            var = ((p_tr - pm) ** 2).sum(0).clamp_min(1e-12)
            a = cov / var
            pred = a * (pred - pm) + ym
        out.append((1 - ((pred - Y) ** 2).sum(0) / sst).clamp_min(0.0).cpu().numpy())
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="oracle,oracle_affine,poly2,last,cat3,last+p2,cat3+p2")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--ckpt", default=str(CKPT), help="换 seed 做稳健性检查")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

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

    singles = [(i,) for i in range(n_gen)]
    pairs = list(combinations(range(n_gen), 2))
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)
    log("E2", "START", note=f"{len(singles)} 单 + {len(pairs)} 对，audit={len(a_idx)} 行")

    # ---- 真值 ----
    tr = pd.read_csv(TRUTH)
    # confidential_indices = range(n_general, n_general+n_conf)，与 data["confidential"] 同序
    conf_names = list(data["confidential"])
    cmap = {c: k for k, c in enumerate(conf_names)}
    miss = set(tr["conf"].unique()) - set(cmap)
    assert not miss, f"真值里有无法映射的 conf: {sorted(miss)}"
    tr["ck"] = tr["conf"].map(cmap)
    log("E2", "TRUTH", note=f"{len(tr)} 条 (i,j,conf) 重训真值，conf 映射 {len(cmap)} 个")

    pair_pos = {p: k for k, p in enumerate(pairs)}
    rows, curves = [], []
    for kind in args.kinds.split(","):
        t0 = time.perf_counter()
        if kind.startswith("oracle"):
            model = load_oracle(ckpt, n_gen, n_conf, dev)
            aff = kind.endswith("affine")
            kw = dict(affine=aff, Xtr=Xtr if aff else None, Ytr=Ytr if aff else None)
            v_s = oracle_r2(model, Xev, Yev, singles, n_gen, **kw)
            v_p = oracle_r2(model, Xev, Yev, pairs, n_gen, **kw)
        else:
            phi = None
            if kind.startswith(("last", "cat3")):
                k0 = "cat3" if kind.startswith("cat3") else "last"
                phi = FrozenPhi(load_oracle(ckpt, n_gen, n_conf, dev), k0)

            def ev(sl):
                out = []
                for s in range(0, len(sl), args.chunk):
                    sb = torch.as_tensor(sl[s:s + args.chunk], device=dev)
                    F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, kind)
                    F_ev = build_features(phi, Xev, Zev, sb, n_gen, kind)
                    out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
                return np.concatenate(out, 0)
            v_s, v_p = ev(singles), ev(pairs)

        est = np.array([v_p[pair_pos[(r.i, r.j)], r.ck]
                        - max(v_s[r.i, r.ck], v_s[r.j, r.ck])
                        for r in tr.itertuples()])
        truth = tr["synergy"].to_numpy()
        rho = spearmanr(est, truth).statistic
        # 尾部：真值 top-100 内部的排序一致性（最需要准的那一档）
        top = np.argsort(-truth)[:100]
        rho_tail = spearmanr(est[top], truth[top]).statistic
        # 检出：真强协同 (syn>0.2) 的 top-K 命中
        strong = truth > 0.2
        order = np.argsort(-est)
        hit20 = strong[order[:20]].sum() / max(min(20, strong.sum()), 1)
        hit100 = strong[order[:100]].sum() / max(min(100, strong.sum()), 1)
        # 真 top-10 的估计排名中位数（相变指纹，与 76 号同一指标）
        rank_of = np.empty(len(est), dtype=int)
        rank_of[order] = np.arange(len(est))
        med_rank = float(np.median(rank_of[np.argsort(-truth)[:10]]))
        rows.append(dict(kind=kind, n=len(est), rho=round(float(rho), 4),
                         rho_tail=round(float(rho_tail), 4),
                         hit20=round(float(hit20), 3), hit100=round(float(hit100), 3),
                         med_rank_true_top10=med_rank,
                         n_strong=int(strong.sum()),
                         sec=round(time.perf_counter() - t0, 1)))
        curves.append(pd.DataFrame(dict(kind=kind, i=tr["i"].values, j=tr["j"].values,
                                        conf=tr["conf"].values, est=est, truth=truth)))
        log("E2", "DONE", note=f"{kind} rho={rho:.4f} tail={rho_tail:.4f} "
                               f"hit20={hit20:.3f} {time.perf_counter()-t0:.0f}s")

    df = pd.DataFrame(rows).assign(ckpt=ckpt.stem)
    df.to_csv(ROOT / f"outputs/order2_truth{args.tag}.csv", index=False)
    pd.concat(curves).to_csv(ROOT / f"outputs/order2_pairs{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== E2：二阶 syn 对 68 号重训真值（audit 分片；oracle 行 = 当前方案基线）===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
