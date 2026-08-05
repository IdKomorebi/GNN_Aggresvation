# -*- coding: utf-8 -*-
"""94 号 分析②：L1 的发现是否随训练种子稳定（主结果的稳健性检查）。

93 号的 p=0.013/0.003 建立在 seed0 单个 checkpoint 上。本脚本检验：
  A. 估计器质量随 seed 稳定吗（o2 的 ρ/尾部 ρ，已由 eval 脚本产出，这里汇总）；
  B. **o4 全扫的排序在 seed 间一致吗**（全体 Spearman + top 重叠率）；
  C. seed0 已认证的真协同（o4 9 个 + o5 10 个），其他 seed 是否也把它们排在前列
     ——这是最要紧的：真发现不应依赖某个特定 seed。
"""
from __future__ import annotations

import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R93 = ROOT.parent / "DNN_Aggresvation93"


def load_scan(prefix):
    fs = sorted(glob.glob(str(R93 / f"outputs/{prefix}_s*of*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["S"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    return d.set_index("S")


def main() -> None:
    # ---- A. 各 seed 的 o2 质量 ----
    print("=== A. 估计器质量随 seed（o2 对重训真值，last 配置）===")
    rows = []
    for tag, name in [("_aug8", "seed0"), ("_aug8s1", "seed1"), ("_aug8s2", "seed2")]:
        p = R93 / f"outputs/eval_l1_order2{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            r = d[d.kind == "last"]
            if len(r):
                rows.append(dict(seed=name, rho=r.rho.iloc[0], rho_tail=r.rho_tail.iloc[0],
                                 hit100=r.hit100.iloc[0]))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- B/C. o4 扫描的跨 seed 一致性 ----
    s0 = load_scan("scan_o4_oracle_l1_aug8_seed0_last")
    for other in ("seed1", "seed2"):
        so = load_scan(f"scan_o4_oracle_l1_aug8_{other}_last")
        if so is None:
            print(f"\n({other} 的 o4 扫描尚未完成，跳过)")
            continue
        a = s0["syn"]
        b = so["syn"].reindex(a.index)
        rho = spearmanr(a, b).statistic
        topN = 208   # seed0 的 n>0.1
        t0 = set(a.nlargest(topN).index)
        t1 = set(b.nlargest(topN).index)
        print(f"\n=== B. seed0 vs {other}（o4 全扫 {len(a)} 个）===")
        print(f"  全体 Spearman = {rho:.4f}   top-{topN} 重叠 = "
              f"{len(t0 & t1)}/{topN} ({len(t0 & t1)/topN*100:.0f}%)")

        # C. seed0 已认证真协同在 other 里的排名
        rank_o = {S: i + 1 for i, S in enumerate(b.sort_values(ascending=False).index)}
        cert4 = pd.concat([pd.read_csv(f) for f in
                           glob.glob(str(R93 / "outputs/certify_o4_l1top_s*.csv"))])
        cert4 = cert4[cert4.group == "strong"]
        cert4["Stup"] = cert4["S"].map(lambda s: tuple(ast.literal_eval(s)))
        true4 = cert4[cert4.syn_true_audit > 0.10]
        ranks = sorted(rank_o.get(S, -1) for S in true4.Stup)
        print(f"  C. seed0 认证的 {len(true4)} 个真 o4 协同在 {other} 扫描中的排名: {ranks}")
        print(f"     中位 {int(np.median(ranks))} / {len(a):,}"
              f"（进 top-208 的 {sum(r <= 208 for r in ranks)}/{len(ranks)}）")


if __name__ == "__main__":
    main()
