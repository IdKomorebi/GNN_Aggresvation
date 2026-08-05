# -*- coding: utf-8 -*-
"""90 号 消融：syn 失效是【字典表达力】还是【统计量】的问题？

注入实验显示 syn（R²差）完全找不到注入的纯四阶结构。有两个可能原因：
  (a) **字典表达力**：v(S) 用的 poly2 字典只有 raw+square+两两乘积，
      结构上无法表示四阶乘积 ⟹ v(S) 与 v(子集) 都拟合不了，作差≈0；
  (b) **统计量**：R² 差本身不校正自由度/噪声，尾部不可靠（89 号实证）。

本脚本对同一注入信号，只换字典重算 syn：
  · poly2      —— 当前用的（raw + square + 两两积）
  · full-inter —— 全部 2^m−1 个子集乘积项（含 m 阶）
若 full-inter 下 syn 变得可检出 ⟹ 主因是 (a) 字典；
若仍不可检出 ⟹ 主因是 (b) 统计量。两者可同时成立。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from fstat import build_terms, highorder_t  # noqa: E402
from runlog import log  # noqa: E402

ALPHAS = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
SIGNALS = (0.20, 0.30, 0.45)
N_RAND = 300
N_TRIAL = 5


def dict_poly2(Z):
    cols = [Z, Z ** 2]
    if Z.shape[1] > 1:
        cols.append(np.column_stack([Z[:, a] * Z[:, b]
                                     for a, b in combinations(range(Z.shape[1]), 2)]))
    return np.column_stack(cols)


def dict_full(Z):
    return build_terms(Z - Z.mean(0), Z.shape[1])


def ridge_r2(Ztr, ytr, Zte, yte, dictf) -> float:
    Xtr, Xte = dictf(Ztr), dictf(Zte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    n, p = Xtr.shape
    r = np.random.RandomState(0).permutation(n)
    nv = max(int(n * 0.15), 1)
    vi, fi = r[:nv], r[nv:]
    best, ba = np.inf, 1.0
    for a in ALPHAS:
        b = np.linalg.solve(Xtr[fi].T @ Xtr[fi] + a * np.eye(p), Xtr[fi].T @ (ytr[fi] - ym))
        s = np.sum((ytr[vi] - ym - Xtr[vi] @ b) ** 2)
        if s < best:
            best, ba = s, a
    b = np.linalg.solve(Xtr.T @ Xtr + ba * np.eye(p), Xtr.T @ (ytr - ym))
    pred = Xte @ b + ym
    return float(np.clip(1 - np.sum((yte - pred) ** 2) /
                         (np.sum((yte - yte.mean()) ** 2) + 1e-12), 0.0, None))


def syn_of(S, Ztr, ytr, Zte, yte, dictf):
    vS = ridge_r2(Ztr[:, list(S)], ytr, Zte[:, list(S)], yte, dictf)
    sub = max(ridge_r2(Ztr[:, list(T)], ytr, Zte[:, list(T)], yte, dictf)
              for T in combinations(S, len(S) - 1))
    return vS - sub, vS, sub


def main() -> None:
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    Xtr = data["train_data"][:, gi].astype(np.float64)
    Xte = data["test_data"][:, gi].astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    n_gen = Ztr.shape[1]

    rng = np.random.RandomState(91)
    rows = []
    for alpha in SIGNALS:
        for tr in range(N_TRIAL):
            S = tuple(sorted(rng.choice(n_gen, 4, replace=False)))
            ptr, pte = np.prod(Ztr[:, S], axis=1), np.prod(Zte[:, S], axis=1)
            m_, s_ = ptr.mean(), ptr.std() + 1e-12
            ptr, pte = (ptr - m_) / s_, (pte - m_) / s_
            ytr = alpha * ptr + np.sqrt(1 - alpha ** 2) * rng.randn(len(ptr))
            yte = alpha * pte + np.sqrt(1 - alpha ** 2) * rng.randn(len(pte))

            rec = dict(alpha=alpha, trial=tr, true_r2=round(alpha ** 2, 4))
            for name, dictf in (("poly2", dict_poly2), ("full", dict_full)):
                s_inj, v_inj, sub_inj = syn_of(S, Ztr, ytr, Zte, yte, dictf)
                srand = []
                for _ in range(N_RAND):
                    R = tuple(sorted(rng.choice(n_gen, 4, replace=False)))
                    srand.append(syn_of(R, Ztr, ytr, Zte, yte, dictf)[0])
                srand = np.array(srand)
                rec[f"syn_{name}"] = round(s_inj, 4)
                rec[f"v_{name}"] = round(v_inj, 4)
                rec[f"sub_{name}"] = round(sub_inj, 4)
                rec[f"pct_{name}"] = round(float((srand < s_inj).mean()) * 100, 2)
                rec[f"rank_{name}"] = int((srand >= s_inj).sum()) + 1
            t, _, _ = highorder_t(Ztr[:, S], ytr[:, None])
            rec["t_stat"] = round(float(t[0]), 2)
            rows.append(rec)
            log("ABL", "ROW", note=f"alpha={alpha} tr={tr} poly2 syn={rec['syn_poly2']}"
                                   f"(pct{rec['pct_poly2']}) full syn={rec['syn_full']}"
                                   f"(pct{rec['pct_full']}) t={rec['t_stat']}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/dict_ablation.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print("\n=== 按注入强度汇总（N_RAND=%d）===" % N_RAND)
    g = df.groupby("alpha").agg(
        true_r2=("true_r2", "first"),
        v_poly2=("v_poly2", "mean"), syn_poly2=("syn_poly2", "mean"),
        pct_poly2=("pct_poly2", "mean"), rank_poly2=("rank_poly2", "mean"),
        v_full=("v_full", "mean"), syn_full=("syn_full", "mean"),
        pct_full=("pct_full", "mean"), rank_full=("rank_full", "mean"),
        t_stat=("t_stat", "mean"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
