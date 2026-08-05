# -*- coding: utf-8 -*-
"""90 号 ★决定性实验：注入已知的【纯高阶】结构，看各方法能否找到。

之前一直缺这个对照——88 号说"反层级集合搜不到"，89 号说"那些集合本身是假阳性"。
两说都无法区分：**数据里没有反层级** vs **方法找不到反层级**。
注入一个真实存在、强度已知、所有低阶子集都无信号的结构，就能把两者分开。

注入：y = α·normalize(x_a·x_b·x_c·x_d) + sqrt(1−α²)·ε
      （字段先中心化标准化；乘积项与各低阶项近似正交 ⟹ 纯四阶）

对比三种统计量在【注入集合】上的表现与其在随机集合中的排名：
  1. **t 检验**（本号新法）：控制全部低阶项后，最高阶项的偏回归 t；
  2. **syn = R² 差**（当前方法）：v(S) − max_T v(T)，poly2+ridge，out-of-sample；
  3. **v(S) 绝对值**（S2 上界筛）。

同时报告四个三元子集的 v，用以确认注入structure确实是"纯高阶"（子集应≈0）。
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
from fstat import highorder_t, t_to_p  # noqa: E402
from runlog import log  # noqa: E402

ALPHAS_RIDGE = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
SIGNALS = (0.10, 0.20, 0.30, 0.45)
N_RAND = 400          # 随机对照集合数（算排名用）
N_TRIAL = 6           # 每个强度重复几次（不同注入字段）


def poly2_cols(Z: np.ndarray) -> np.ndarray:
    """raw + square + pairwise product（与 85/88 号同族字典）。"""
    cols = [Z, Z ** 2]
    if Z.shape[1] > 1:
        cols.append(np.column_stack([Z[:, a] * Z[:, b]
                                     for a, b in combinations(range(Z.shape[1]), 2)]))
    return np.column_stack(cols)


def ridge_r2(Ztr, ytr, Zte, yte) -> float:
    """训练集选 alpha（内部 val），返回独立集 R²。"""
    Xtr, Xte = poly2_cols(Ztr), poly2_cols(Zte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    n = len(Xtr)
    rs = np.random.RandomState(0).permutation(n)
    nv = max(int(n * 0.15), 1)
    vi, fi = rs[:nv], rs[nv:]
    best, best_a = np.inf, 1.0
    for a in ALPHAS_RIDGE:
        beta = np.linalg.solve(Xtr[fi].T @ Xtr[fi] + a * np.eye(Xtr.shape[1]),
                               Xtr[fi].T @ (ytr[fi] - ym))
        sse = np.sum((ytr[vi] - ym - Xtr[vi] @ beta) ** 2)
        if sse < best:
            best, best_a = sse, a
    beta = np.linalg.solve(Xtr.T @ Xtr + best_a * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
    pred = Xte @ beta + ym
    sse = np.sum((yte - pred) ** 2)
    sst = np.sum((yte - yte.mean()) ** 2) + 1e-12
    return float(np.clip(1 - sse / sst, 0.0, None))


def main() -> None:
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    Xtr_all = data["train_data"][:, gi].astype(np.float64)
    Xte_all = data["test_data"][:, gi].astype(np.float64)
    n_gen = Xtr_all.shape[1]

    # 标准化（注入前）
    mu, sd = Xtr_all.mean(0), Xtr_all.std(0)
    sd[sd < 1e-9] = 1.0
    Ztr = (Xtr_all - mu) / sd
    Zte = (Xte_all - mu) / sd
    log("INJ", "DATA", note=f"train={Ztr.shape} test={Zte.shape}")

    rng = np.random.RandomState(90)
    rows = []
    for alpha in SIGNALS:
        for trial in range(N_TRIAL):
            S = tuple(sorted(rng.choice(n_gen, 4, replace=False)))
            # ---- 注入纯四阶 ----
            ptr = np.prod(Ztr[:, S], axis=1)
            pte = np.prod(Zte[:, S], axis=1)
            m_, s_ = ptr.mean(), ptr.std() + 1e-12
            ptr, pte = (ptr - m_) / s_, (pte - m_) / s_
            etr = rng.randn(len(ptr))
            ete = rng.randn(len(pte))
            ytr = alpha * ptr + np.sqrt(max(1 - alpha ** 2, 0)) * etr
            yte = alpha * pte + np.sqrt(max(1 - alpha ** 2, 0)) * ete

            # ---- 1. t 检验（train 上做，标准显著性检验）----
            t_inj, _, pr2 = highorder_t(Ztr[:, S], ytr[:, None])
            t_inj, pr2 = float(t_inj[0]), float(pr2[0])

            # ---- 2/3. syn 与 v(S)（train 拟合，test 评估）----
            v_S = ridge_r2(Ztr[:, S], ytr, Zte[:, S], yte)
            v_subs = [ridge_r2(Ztr[:, list(T)], ytr, Zte[:, list(T)], yte)
                      for T in combinations(S, 3)]
            syn_inj = v_S - max(v_subs)

            # ---- 随机对照：算 t 与 syn 的分布，得到注入集合的排名 ----
            t_rand, syn_rand = [], []
            for _ in range(N_RAND):
                R = tuple(sorted(rng.choice(n_gen, 4, replace=False)))
                tr_, _, _ = highorder_t(Ztr[:, R], ytr[:, None])
                t_rand.append(abs(float(tr_[0])))
                vR = ridge_r2(Ztr[:, R], ytr, Zte[:, R], yte)
                vRs = max(ridge_r2(Ztr[:, list(T)], ytr, Zte[:, list(T)], yte)
                          for T in combinations(R, 3))
                syn_rand.append(vR - vRs)
            t_rand, syn_rand = np.array(t_rand), np.array(syn_rand)

            rows.append(dict(
                alpha=alpha, trial=trial, S=str(S),
                true_r2=round(alpha ** 2, 4),
                # t 检验
                t_stat=round(t_inj, 2), partial_r2=round(pr2, 4),
                t_pct=round(float((t_rand < abs(t_inj)).mean()) * 100, 2),
                t_rank=int((t_rand >= abs(t_inj)).sum()) + 1,
                # syn
                syn=round(syn_inj, 4), v_S=round(v_S, 4),
                max_sub_v=round(max(v_subs), 4),
                syn_pct=round(float((syn_rand < syn_inj).mean()) * 100, 2),
                syn_rank=int((syn_rand >= syn_inj).sum()) + 1,
            ))
            log("INJ", "ROW", note=f"alpha={alpha} trial={trial} t={t_inj:.1f}(pct{rows[-1]['t_pct']}) "
                                   f"syn={syn_inj:.4f}(pct{rows[-1]['syn_pct']}) maxsub_v={max(v_subs):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/injection.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print("\n=== 逐次结果 ===")
    print(df.to_string(index=False))
    print("\n=== 按注入强度汇总（N_RAND=%d 个随机对照）===" % N_RAND)
    g = df.groupby("alpha").agg(
        true_r2=("true_r2", "first"),
        max_sub_v=("max_sub_v", "mean"),          # 低阶子集能拿到多少（应≈0 ⟹ 纯高阶）
        t_stat=("t_stat", "mean"), t_pct=("t_pct", "mean"), t_rank=("t_rank", "mean"),
        syn=("syn", "mean"), syn_pct=("syn_pct", "mean"), syn_rank=("syn_rank", "mean"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
