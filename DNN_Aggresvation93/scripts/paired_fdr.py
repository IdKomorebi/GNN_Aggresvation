# -*- coding: utf-8 -*-
"""93 号 任务C：★配对检验 + BH-FDR——把"硬阈值 top-K"换成有统计保证的协同清单。

为什么现在才能做
----------------
90 号 §6 的 t 检验有两个毛病：**in-sample**（真实字段高度相关⟹乘积项病态共线）、
且与 syn 是两套口径。91 号的闭式读出恰好同时修掉这两点：

  · S 与其子集 T 在**同一批 audit 样本**上各有一组逐样本残差平方 ⟹ 可做**配对**比较；
  · 闭式解**没有优化噪声**（微调 S 和 T 各自带独立的初始化/批序噪声，差分放大 √2），
    S 与 T 的误差来自**同一个 φ**、高度相关，差分大量抵消 ⟹ 配对方差远小于独立估计。

这正是 91 号 §2 所说"闭式读出对估计小的不可约增量是**结构性**优势"的兑现。

口径（严格遵守 88 号三分割，避免选择性推断）
--------------------------------------------
  train  : 拟合 β、选 λ
  search : **选择**——每个 S 用哪个 conf、哪个子集 T 是最强父集
  audit  : **只算 p 值**，一次
若在 audit 上既选又检验，就是典型的赢家诅咒；分开后配对 t 的零分布才是有效的。

统计量：d_i = err²_T(i) − err²_S(i)（同一样本 i），
  syn = mean(d)/Var(Y_c)，t = mean(d)/(sd(d)/√n)，单边 p，再做 Benjamini–Hochberg。
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
from scipy import stats

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


def bh_fdr(p: np.ndarray, q: float = 0.05):
    """Benjamini–Hochberg：返回 (是否拒绝, q 值)。"""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = np.arange(1, n + 1) / n * q
    passed = ranked <= crit
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, dtype=bool)
    rej[order[:k]] = True
    qv = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out_q = np.empty(n)
    out_q[order] = np.clip(qv, 0, 1)
    return rej, out_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kind", default="cat3", help="三阶上 cat3 最优（91 号 E3）")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--taus", default="0,0.05,0.10,0.15",
                    help="★检验 H0: syn ≤ τ。τ=0 是点零假设（几乎总被拒绝，见 §结论）")
    ap.add_argument("--max_sets", type=int, default=0, help=">0 则只跑前 N 个（调试）")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")]

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
    half = len(perm) // 2
    s_idx, a_idx = perm[:half], perm[half:]

    mu, sd = Xtr_np.mean(0), Xtr_np.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=dev)
    Ztr = torch.tensor((Xtr_np - mu) / sd, dtype=torch.float64, device=dev)
    Ytr = torch.tensor(Ytr_np, dtype=torch.float64, device=dev)

    def prep(idx):
        return (torch.tensor(Xhe_np[idx], dtype=torch.float32, device=dev),
                torch.tensor((Xhe_np[idx] - mu) / sd, dtype=torch.float64, device=dev),
                torch.tensor(Yhe_np[idx], dtype=torch.float64, device=dev))
    Xs, Zs, Ys = prep(s_idx)
    Xa, Za, Ya = prep(a_idx)
    fit_i, val_i = fit_val_idx(len(Xtr_np), dev)

    phi = FrozenPhi(load_oracle(CKPT, n_gen, n_conf, dev),
                    "cat3" if args.kind.startswith("cat3") else "last")
    m = args.order
    sets = list(combinations(range(n_gen), m))
    if args.max_sets:
        sets = sets[:args.max_sets]
    subs = sorted({tuple(sorted(t)) for S in sets for t in combinations(S, m - 1)})
    sub_pos = {s: i for i, s in enumerate(subs)}
    log("FDR", "START", note=f"order-{m}: {len(sets)} 个集合 + {len(subs)} 个子集，"
                             f"kind={args.kind}，search={len(s_idx)} audit={len(a_idx)}")
    t0 = time.perf_counter()

    def run(sl, X_ev, Z_ev, Y_ev):
        """返回 (R²:(B,nC), 逐样本残差平方:(B,n_ev,nC))。"""
        r2s, res = [], []
        for s in range(0, len(sl), args.chunk):
            sb = torch.as_tensor(sl[s:s + args.chunk], device=dev)
            F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, args.kind)
            F_ev = build_features(phi, X_ev, Z_ev, sb, n_gen, args.kind)
            r, e, _ = ridge_r2(F_tr, Ytr, F_ev, Y_ev, fit_i, val_i, ret_resid=True)
            r2s.append(r.cpu().numpy())
            res.append(e.cpu().numpy().astype(np.float32))
        return np.concatenate(r2s, 0), np.concatenate(res, 0)

    # ---- search：只用来做选择 ----
    r2_sub_s, _ = run(subs, Xs, Zs, Ys)
    r2_set_s, _ = run(sets, Xs, Zs, Ys)
    log("FDR", "SEARCH", note=f"search 侧完成 {time.perf_counter()-t0:.0f}s")

    # ---- audit：只用来算 p 值 ----
    _, res_sub_a = run(subs, Xa, Za, Ya)
    _, res_set_a = run(sets, Xa, Za, Ya)
    log("FDR", "AUDIT", note=f"audit 侧完成 {time.perf_counter()-t0:.0f}s")

    var_y = ((Ya - Ya.mean(0)) ** 2).mean(0).cpu().numpy()
    n_a = len(a_idx)
    rows = []
    for i, S in enumerate(sets):
        srows = [sub_pos[tuple(sorted(t))] for t in combinations(S, m - 1)]
        inc_s = r2_set_s[i] - r2_sub_s[srows].max(0)      # (nC,) 仅用于选择
        c = int(np.argmax(inc_s))                          # ← 选 conf（search）
        best_t = int(srows[int(np.argmax(r2_sub_s[srows, c]))])   # ← 选最强父集（search）
        d = res_sub_a[best_t, :, c] - res_set_a[i, :, c]   # 配对损失差（audit）
        vy = var_y[c]
        syn = float(d.mean()) / vy                          # 点估计（= R² 差）
        se = float(d.std(ddof=1)) / np.sqrt(n_a) / vy       # 配对标准误
        rows.append(dict(S=str(S), conf=c, syn_search=float(inc_s[c]),
                         syn_audit=syn, se=se))
    df = pd.DataFrame(rows)

    # ---- 对每个 τ 做 H0: syn ≤ τ 的单边检验 + BH-FDR ----
    summary = []
    for tau in taus:
        t = (df.syn_audit - tau) / df.se.clip(lower=1e-12)
        p = np.asarray(stats.t.sf(t.to_numpy(), n_a - 1), dtype=float)
        rej, qv = bh_fdr(p, args.q)
        df[f"p@{tau}"], df[f"q@{tau}"], df[f"rej@{tau}"] = p, qv, rej
        n_hard = int((df.syn_audit > tau).sum())
        both = int(((df.syn_audit > tau) & rej).sum())
        summary.append(dict(tau=tau, 过FDR=int(rej.sum()),
                            占比=round(rej.mean() * 100, 2),
                            纯硬阈值=n_hard, 交集=both,
                            过阈值但不显著=n_hard - both,
                            显著但不过阈值=int(rej.sum()) - both))
    df.sort_values("syn_audit", ascending=False).to_csv(
        ROOT / f"outputs/paired_fdr_o{m}_{args.kind}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== 任务C：order-{m}（{len(df)} 个集合，audit n={n_a}，BH q={args.q}）===")
    print("检验 H0: syn ≤ τ —— τ=0 是点零假设，τ>0 才是审计真正要问的问题\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n★读法：τ=0 时通过率极高，说明'增量是否 >0'在真实数据上几乎恒真——"
          "\n  多加一个字段几乎总能带来统计上可检测的信息。审计要控的错误是"
          "\n  '声称 syn>τ 但实际 ≤τ'，所以 τ 必须取审计关心的强度，不能取 0。")
    tau_m = taus[min(2, len(taus) - 1)]
    sel = df[df[f"rej@{tau_m}"]]
    print(f"\n--- τ={tau_m} 下通过 FDR 且 syn 最大的 10 个 ---")
    print(sel.nlargest(10, "syn_audit")[
        ["S", "conf", "syn_search", "syn_audit", "se", f"q@{tau_m}"]
    ].round(4).to_string(index=False))
    log("FDR", "DONE", note=f"order-{m} {args.kind}: " +
        " | ".join(f"τ={r['tau']}:{r['过FDR']}" for r in summary) +
        f"，用时 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
