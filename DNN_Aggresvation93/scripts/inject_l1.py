# -*- coding: utf-8 -*-
"""91 号 ★E1 判定实验：注入已知的纯四阶结构，看 Frozen-φ Ridge 能否找到。

为什么用这个当判据
------------------
90 号建好了唯一干净的裁判：注入 y = α·normalize(x_a x_b x_c x_d) + √(1−α²)·ε，
所有低阶子集无信号，强度已知。已有成绩（真实 R²=0.202 时在 400 个随机四元组中的百分位）：
    poly2 + syn（现方案）  43%  ← 与随机无异
    full 字典 + syn        83%
    in-sample t 检验       85%
本号问：**把手工字典换成冻结 oracle 学到的条件特征**，能到多少？

判据（预注册，两个结果都算数）
    · L0 若超过 83% ⟹ 学出来的特征强于手工全交互字典，且**不需要人工指定阶数**
      （full 字典是"我知道要找四阶所以手工放了四阶项"，φ 是自动的，含金量不同）；
    · L0 若平庸 ⟹ 冻结主干的特征已过度专门化到 12 个真实 conf，必须 meta-训练(L1)。

这对 L0 是**偏严**的测试：φ 是为预测真实 12 个 conf 学的，而注入的 y 与它们无关。
故本号同时跑 E2（真实 conf 的二阶 syn 对重训真值），两者一起看才完整。

关键实现优化
------------
φ(x⊙m, m) 只依赖 mask、**与 y 无关** ⟹ 把 4 强度 × 6 trial = 24 个注入信号
拼成 (n,24) 的 Y 一次算完；Gram ΦᵀΦ 与 y 无关，只有 H=ΦᵀY 依赖 y。
400 个随机对照集合跨所有 trial 共用，φ 只前向一次。
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
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, poly2_block, ridge_r2)
from fstat import highorder_t  # noqa: E402
from runlog import log  # noqa: E402

SIGNALS = (0.10, 0.20, 0.30, 0.45)     # 与 90 号一致
N_TRIAL = 6
N_RAND = 400
INJ_SEED = 90                          # 与 90 号同种子 ⟹ 注入集合可比
CKPT = REPO / "DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt"


def full_block(Z: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
    """poly2 再加 3..m 阶乘积项（= 90 号 full 字典，⊃ poly2）。"""
    B, m = sets.shape
    raw = Z[:, sets.reshape(-1)].T.reshape(B, m, -1).transpose(1, 2)     # (B,n,m)
    cols = [poly2_block(Z, sets)]
    for r in range(3, m + 1):
        for c in combinations(range(m), r):
            prod = raw[:, :, c[0]]
            for j in c[1:]:
                prod = prod * raw[:, :, j]
            cols.append(prod.unsqueeze(2))
    return torch.cat(cols, dim=2)


def features(kind, phi, Xtr, Xev, Ztr, Zev, sets, n_gen):
    if kind == "full":
        return full_block(Ztr, sets), full_block(Zev, sets)
    return (build_features(phi, Xtr, Ztr, sets, n_gen, kind),
            build_features(phi, Xev, Zev, sets, n_gen, kind))


def eval_sets(kind, phi, Xtr, Xev, Ztr, Zev, Ytr, Yev, sets_list,
              fit_i, val_i, n_gen, chunk):
    """对一批同阶集合算独立集 R²，返回 (n_sets, nC)。"""
    out = []
    for s in range(0, len(sets_list), chunk):
        sb = torch.as_tensor(sets_list[s:s + chunk], device=Ztr.device)
        F_tr, F_ev = features(kind, phi, Xtr, Xev, Ztr, Zev, sb, n_gen)
        out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
    return np.concatenate(out, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="poly2,full,last,cat3,last+p2,cat3+p2")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    Xtr_np = data["train_data"][:, gi].astype(np.float64)
    Xhe_np = data["test_data"][:, gi].astype(np.float64)
    n_gen = Xtr_np.shape[1]

    # audit 分片（88 号口径：held-out 一分为二，只用后一半报数）
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(Xhe_np))
    a_idx = perm[len(perm) // 2:]
    Xev_np = Xhe_np[a_idx]
    log("E1", "DATA", note=f"train={Xtr_np.shape} audit={Xev_np.shape} "
                           f"raw_mean={Xtr_np.mean():.4f} raw_std={Xtr_np.std():.4f}")

    # 喂 oracle 用 prepare_data 的原始口径（与 75 号一致）；构造乘积项用标准化版
    mu, sd = Xtr_np.mean(0), Xtr_np.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=dev)
    Xev = torch.tensor(Xev_np, dtype=torch.float32, device=dev)
    Ztr = torch.tensor((Xtr_np - mu) / sd, dtype=torch.float64, device=dev)
    Zev = torch.tensor((Xev_np - mu) / sd, dtype=torch.float64, device=dev)
    Ztr_np, Zev_np = Ztr.cpu().numpy(), Zev.cpu().numpy()

    # ---- 构造 24 个注入信号（4 强度 × 6 trial），拼成 (n,24) 一次算完 ----
    rng = np.random.RandomState(INJ_SEED)
    inj_sets, ytr_cols, yev_cols, meta = [], [], [], []
    for alpha in SIGNALS:
        for trial in range(N_TRIAL):
            S = tuple(sorted(rng.choice(n_gen, 4, replace=False)))
            ptr, pev = np.prod(Ztr_np[:, S], axis=1), np.prod(Zev_np[:, S], axis=1)
            m_, s_ = ptr.mean(), ptr.std() + 1e-12
            ptr, pev = (ptr - m_) / s_, (pev - m_) / s_
            k = np.sqrt(max(1 - alpha ** 2, 0))
            ytr_cols.append(alpha * ptr + k * rng.randn(len(ptr)))
            yev_cols.append(alpha * pev + k * rng.randn(len(pev)))
            inj_sets.append(S)
            meta.append(dict(alpha=alpha, trial=trial, S=str(S)))
    Ytr = torch.tensor(np.column_stack(ytr_cols), dtype=torch.float64, device=dev)
    Yev = torch.tensor(np.column_stack(yev_cols), dtype=torch.float64, device=dev)
    nC = Ytr.shape[1]

    # ---- 400 个随机对照四元组（跨全部 trial 共用）----
    r2 = np.random.RandomState(INJ_SEED + 1)
    rand_sets = []
    seen = set(inj_sets)
    while len(rand_sets) < N_RAND:
        S = tuple(sorted(r2.choice(n_gen, 4, replace=False)))
        if S not in seen:
            seen.add(S)
            rand_sets.append(S)

    all4 = inj_sets + rand_sets
    subs = sorted({tuple(sorted(t)) for S in all4 for t in combinations(S, 3)})
    sub_pos = {s: i for i, s in enumerate(subs)}
    log("E1", "SETS", note=f"注入 {len(inj_sets)} + 随机 {len(rand_sets)} 四元组，"
                           f"三阶子集 {len(subs)} 个；Y 拼成 {nC} 列一次算")

    phi_cache = {}
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)
    rows = []
    for kind in args.kinds.split(","):
        t0 = time.perf_counter()
        if kind.startswith(("last", "cat3")):
            k0 = "cat3" if kind.startswith("cat3") else "last"
            if k0 not in phi_cache:
                phi_cache[k0] = FrozenPhi(load_oracle(ckpt, n_gen, 12, dev), k0)
            phi = phi_cache[k0]
        else:
            phi = None
        v_sub = eval_sets(kind, phi, Xtr, Xev, Ztr, Zev, Ytr, Yev, subs,
                          fit_i, val_i, n_gen, args.chunk)
        v_set = eval_sets(kind, phi, Xtr, Xev, Ztr, Zev, Ytr, Yev, all4,
                          fit_i, val_i, n_gen, args.chunk)
        # syn(S) 逐列（逐注入信号）
        syn = np.empty_like(v_set)
        for i, S in enumerate(all4):
            best = v_sub[[sub_pos[tuple(sorted(t))] for t in combinations(S, 3)]].max(0)
            syn[i] = v_set[i] - best

        n_inj = len(inj_sets)
        for j in range(nC):
            s_inj, s_rand = syn[j, j], syn[n_inj:, j]
            v_inj = v_set[j, j]
            rows.append(dict(kind=kind, **meta[j],
                             true_r2=round(float(v_inj), 4),
                             syn=round(float(s_inj), 4),
                             pct=round(float((s_rand < s_inj).mean() * 100), 2),
                             rank=int((s_rand >= s_inj).sum()) + 1,
                             rand_p99=round(float(np.percentile(s_rand, 99)), 4)))
        log("E1", "DONE", note=f"{kind} 用时 {time.perf_counter()-t0:.0f}s")

    # ---- t 检验对照（90 号方法，in-sample，train 上做）----
    t0 = time.perf_counter()
    t_all = np.empty((len(all4), nC))
    for i, S in enumerate(all4):
        t_all[i] = highorder_t(Ztr_np[:, list(S)], Ytr.cpu().numpy())[0]
    for j in range(nC):
        ti, tr = abs(t_all[j, j]), np.abs(t_all[len(inj_sets):, j])
        rows.append(dict(kind="t_test", **meta[j], true_r2=np.nan,
                         syn=round(float(ti), 3),
                         pct=round(float((tr < ti).mean() * 100), 2),
                         rank=int((tr >= ti).sum()) + 1, rand_p99=np.nan))
    log("E1", "DONE", note=f"t_test 用时 {time.perf_counter()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"outputs/inject_l0{args.tag}.csv", index=False)
    piv = df.groupby(["kind", "alpha"]).agg(
        pct=("pct", "mean"), rank=("rank", "mean"), r2=("true_r2", "mean")).reset_index()
    piv.to_csv(ROOT / f"outputs/inject_l0_summary{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== 注入集合在 400 个随机四元组中的百分位（越高越好，均值 over 6 trial）===")
    print(piv.pivot(index="kind", columns="alpha", values="pct").round(1).to_string())
    print("\n=== 排名（1 = 最好）===")
    print(piv.pivot(index="kind", columns="alpha", values="rank").round(1).to_string())
    print("\n=== 注入集合上实测 v(S)（判断注入强度是否被捕捉到）===")
    print(piv.pivot(index="kind", columns="alpha", values="r2").round(3).to_string())


if __name__ == "__main__":
    main()
