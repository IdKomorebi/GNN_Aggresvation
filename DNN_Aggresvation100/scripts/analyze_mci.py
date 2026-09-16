# -*- coding: utf-8 -*-
"""RQ3：M^(K) 相对近似 full-MCI 的饱和。

MCI_lower(i,c) = max( M^(3)_true(i,c),  max_{候选大背景 T} Δ_true(T) )，其中候选由 mci_search.py 用通用模型生成、
由专用重训认证（DNN，val 选轮次；大背景不做单调闭包，报告 raw 值）。它是 MCI 的**已认证下界**，所以
饱和比 M^(K)/MCI_lower 是真实饱和比的**上估**（真实 MCI 可能更大）。
输出：各 K 饱和比分布、达到 90%/95% 的占比、大背景超出 M^(3) 的条目比例与幅度、估计器在大背景上的误差。
"""
import sys, glob, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"; rep = ["# 100 号 MCI 近似与饱和（自动生成）\n"]
for ds in ["pjm", "caiso"]:
    fs = sorted(glob.glob(str(ROOT / f"outputs/truth/{ds}_mci_seed0_s*of3.npz")))
    mz = A / f"{ds}_M_dnn.npz"
    if len(fs) != 3 or not mz.exists(): rep.append(f"## {ds}: 未完成"); continue
    meta = pickle.load(open(ROOT / f"outputs/sets/{ds}_mci_meta.pkl", "rb")); n = 2 * len(meta)
    V = np.zeros((n, 12), np.float32)
    for f in fs:
        z = np.load(f); V[z["idx"]] = np.clip(z["clean"], 0, 1)
    Z = np.load(mz); M = Z["M"]; act = list(Z["active"])
    low = M[:, 3].copy(); extra = []
    for r, m in enumerate(meta):
        d = V[2 * r + 1, m["conf"]] - V[2 * r, m["conf"]]; p = act.index(m["i"])
        extra.append(dict(p=p, conf=m["conf"], size=len(m["T"]), dtrue=d, dhat=m["dhat"]))
        low[p, m["conf"]] = max(low[p, m["conf"]], d)
    E = pd.DataFrame(extra)
    ok = low > 0.05
    rows = []
    for K in range(4):
        ratio = M[:, K][ok] / low[ok]
        rows.append(dict(K=K, 饱和比均值=ratio.mean(), 饱和比中位数=np.median(ratio), 达到90占比=(ratio >= 0.9).mean(), 达到95占比=(ratio >= 0.95).mean(),
                         M均值=M[:, K].mean()))
    gap = low - M[:, 3]
    rep.append(f"## {ds} 饱和比 M^(K)/MCI_lower（MCI_lower>0.05 的 {int(ok.sum())} 条）\n\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
    rep.append(f"- 大背景（≥4 字段）认证边际超过 M^(3) 0.02 以上的条目占比：{(gap > 0.02).mean():.3f}；超过 0.05：{(gap > 0.05).mean():.3f}；最大超出 {gap.max():.3f}\n"
               f"- 候选大背景：{len(E)} 个，平均规模 {E['size'].mean():.1f}；估计边际 vs 认证边际 MAE {np.abs(E.dhat - E.dtrue).mean():.4f}，偏差 {(E.dhat - E.dtrue).mean():+.4f}")
    np.savez(A / f"{ds}_mci_lower.npz", low=low, M=M)
(A / "report_mci.md").write_text("\n\n".join(rep), encoding="utf-8"); print("\n\n".join(rep))
