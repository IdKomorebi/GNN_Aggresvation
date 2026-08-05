# -*- coding: utf-8 -*-
"""95_caiso：闭式共线 R² 加验（"协同=共线放大"机制的 CAISO 复刻，零 GPU）。

closed R²_c(S) = r' R⁻¹ r（train 相关矩阵，高斯线性假设下的可达 R²），
syn_closed(S) = max_c [R²_c(S) − max_{T⊂S,|T|=m-1} R²_c(T)]。

检验三件事：
1. order-2 全空间：closed syn2 vs 990 集合重训真值（Spearman/尾部/hit）；
2. 认证集合（o3/o4/o5 各批 strong+control）：closed syn 对认证真值的排序与判别（AUC）；
3. closed 与 L1 估计的相关（机制解释：图的价值在求逆非聚合）。
"""
from __future__ import annotations

import ast
import glob
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation69"))
from src.data_processing import prepare_data  # noqa: E402

cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/caiso_2025_hourly_cleaned.csv")
np.random.seed(42)
di = prepare_data(cfg)
gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
tr = di["train_data"]
X, Y = tr[:, gi].astype(np.float64), tr[:, ci].astype(np.float64)
X = (X - X.mean(0)) / X.std(0).clip(1e-9)
Y = (Y - Y.mean(0)) / Y.std(0).clip(1e-9)
n = len(X)
Rgg = X.T @ X / n                       # (44,44)
Rgc = X.T @ Y / n                       # (44,12)
RIDGE = 1e-6                            # 数值稳定（44 字段内有完全重复列对）


def closed_r2(S):
    """返回 (nC,) 的闭式 R²。"""
    idx = list(S)
    R = Rgg[np.ix_(idx, idx)] + RIDGE * np.eye(len(idx))
    r = Rgc[idx]                        # (m,nC)
    return np.einsum("mc,mk,kc->c", r, np.linalg.inv(R), r).clip(0, 1)


def closed_syn(S):
    m = len(S)
    v = closed_r2(S)
    best = np.max([closed_r2(t) for t in combinations(S, m - 1)], axis=0)
    inc = v - best
    c = int(np.argmax(inc))
    return float(inc[c]), c


def load_cert(pattern):
    fs = sorted(glob.glob(str(ROOT / f"outputs/{pattern}")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    d["Stup"] = d["S"].map(lambda s: tuple(ast.literal_eval(s)))
    return d


# ---- 1. order-2 全空间 ----
truth = pd.read_csv(ROOT / "outputs/synergy2_perconf.csv")
conf_names = list(di["confidential"])
cmap = {c: k for k, c in enumerate(conf_names)}
truth["ck"] = truth["conf"].map(cmap)
r2_single = np.stack([closed_r2((i,)) for i in range(len(gi))])       # (44,nC)
pairs = sorted({(r.i, r.j) for r in truth.itertuples()})
r2_pair = {p: closed_r2(p) for p in pairs}
est = np.array([r2_pair[(r.i, r.j)][r.ck] - max(r2_single[r.i, r.ck],
                                                r2_single[r.j, r.ck])
                for r in truth.itertuples()])
tv = truth["synergy"].to_numpy()
rho = spearmanr(est, tv).statistic
top = np.argsort(-tv)[:100]
rho_tail = spearmanr(est[top], tv[top]).statistic
order = np.argsort(-est)
strong = tv > 0.2
hit20 = strong[order[:20]].sum() / min(20, strong.sum())
print(f"1) order-2 全空间 closed-form: rho={rho:.4f} rho_tail={rho_tail:.4f} "
      f"hit20={hit20:.3f}  （对照 L1 rho≈0.947/tail≈0.97，oracle 0.747/0.596）")

# ---- 2. 认证集合上的判别 ----
print("\n2) 认证集合上的 closed syn vs 认证真值：")
for order_m, pat in [(3, "certify_o3_fdrtop_s*.csv"), (4, "certify_o4_l1top_s*.csv"),
                     (5, "certify_o5_l1top_s*.csv")]:
    cert = load_cert(pat)
    if cert is None or not len(cert):
        print(f"  o{order_m}: 认证未就绪")
        continue
    cs = np.array([closed_syn(S)[0] for S in cert.Stup])
    cert[f"closed_syn"] = cs
    sp = spearmanr(cs, cert.syn_true_audit).statistic
    lab = (cert.syn_true_audit > 0.10).to_numpy()
    auc = np.nan
    if 0 < lab.sum() < len(lab):
        pos, neg = cs[lab], cs[~lab]
        auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    print(f"  o{order_m} (n={len(cert)}): Spearman(closed, 认证)={sp:.3f}  "
          f"AUC(>0.10 判别)={auc:.3f}  closed 范围 [{cs.min():.3f},{cs.max():.3f}]")
    cert.to_csv(ROOT / f"outputs/closed_form_o{order_m}.csv", index=False)

# ---- 3. 与 L1 的相关（o4 top 群体） ----
fs = sorted(glob.glob(str(ROOT / "outputs/scan_o4_oracle_l1_aug8_seed0_last_s*of*.parquet")))
scan4 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
scan4["Stup"] = scan4["indices"].map(lambda s: tuple(ast.literal_eval(s)))
top_l1 = scan4.nlargest(200, "syn")
cs = np.array([closed_syn(S)[0] for S in top_l1.Stup])
print(f"\n3) L1 o4 top-200 上 closed vs L1: Spearman="
      f"{spearmanr(cs, top_l1.syn).statistic:.3f}")
