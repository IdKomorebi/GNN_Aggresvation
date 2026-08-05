# -*- coding: utf-8 -*-
"""H-1 协议评估（修正口径）：把"保留率"定义在【13244 个候选空间】上，而不是评测池上。

为什么要改
----------
analyze_h1.py 里的 recall_curve 是"在评测池里保留前 r 比例"，这对**单个排序器**的
比较是对的，但对**两级协议**是错的：第二级只对第一级筛出的候选有估计值，
若仍按评测池比例截断，会把"第一级已经淘汰掉的候选"混进分母，得出偏低的天花板。

正确口径
--------
协议 (R1, R2)：
  第一级：K=0 全扫 13244 个三元组，按 max_c ŝyn3 排序，保留前 R1（成本≈0）
  第二级：对这 R1 部分做 K=K* 微调，重新按 max_c ŝyn3 排序，保留前 R2（相对 R1）
  某个已认证三元组"存活" = 它同时通过两级
  召回 = P(存活 | 该三元组含真强三阶协同)     ← 在**无偏池**上估计
  成本 = 13244·R1 · K* 步微调（第一级免费）

红线：两级的排序键全部来自 oracle 估计，零重训真值。
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

R77 = Path(__file__).resolve().parents[1]
R76 = R77.parent / "DNN_Aggresvation76"
R68 = R77.parent / "DNN_Aggresvation68"
R69 = R77.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R77 / "src"))
from src.data_processing import prepare_data   # noqa: E402
from runlog import log                          # noqa: E402

OUT = R77 / "outputs"
DELTA = 0.10


def main():
    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    n2i = {n: i for i, n in enumerate(di["general"])}

    # ---- 第一级：全 13244 的 max_c ŝyn3@K=0 ----
    s3 = pd.read_csv(OUT / "triples_syn3_k0.csv")
    lvl1 = s3.groupby(["i", "j", "k"]).syn3_est_k0.max().sort_values(ascending=False)
    N_ALL = len(lvl1)
    rank1 = {t: r for r, t in enumerate(lvl1.index)}          # 0 = 最强

    # ---- 第二级：top-30% 的 max_c ŝyn3@K* ----
    ks_files = sorted(OUT.glob("triples_kstar_shard*.csv"))
    d2 = pd.concat([pd.read_csv(p) for p in ks_files])
    low = pd.read_csv(OUT / "lowfour_k0.csv")
    v2 = low[low["size"] == 2].set_index(["key", "conf"]).est.to_dict()

    def key(*ix):
        return "_".join(f"{x:02d}" for x in sorted(ix))

    syn = []
    for r in d2.itertuples():
        p = [v2.get((key(r.i, r.j), r.conf)), v2.get((key(r.i, r.k), r.conf)),
             v2.get((key(r.j, r.k), r.conf))]
        syn.append(r.est - max(p) if all(x is not None for x in p) else np.nan)
    d2["syn3_est_kstar"] = syn
    lvl2 = d2.dropna(subset=["syn3_est_kstar"]).groupby(
        ["i", "j", "k"]).syn3_est_kstar.max().sort_values(ascending=False)
    N_L1 = len(lvl2)
    rank2 = {t: r for r, t in enumerate(lvl2.index)}
    R1_ACTUAL = N_L1 / N_ALL
    KSTAR = int(d2.K.iloc[0])

    # ---- 真值：68 号已认证三元组（按三元组聚合：任一 conf 强即为强）----
    tri = pd.read_csv(R68 / "outputs/triples_certified.csv")
    tri["ix"] = [tuple(sorted((n2i[a], n2i[b], n2i[c])))
                 for a, b, c in zip(tri.fi, tri.fj, tri.fk)]
    g = tri.groupby(["ix", "group"]).syn3_true.max().reset_index()

    print("=" * 92)
    print(f"两级协议（修正口径：保留率定义在 {N_ALL} 个候选空间上）")
    print(f"第一级 K=0 全扫 {N_ALL} 个（2 s）→ 保留 top {R1_ACTUAL:.0%}（{N_L1} 个）"
          f" → 第二级 K*={KSTAR} 微调")
    print("=" * 92)

    rows = []
    for pool in ["triple_rand", "triple_top"]:
        sub = g[g.group == pool]
        strong = sub[sub.syn3_true > DELTA]
        if not len(strong):
            continue
        print(f"\n--- {pool}（n={len(sub)} 个三元组，含真强三阶 {len(strong)} 个，"
              f"基率 {len(strong)/len(sub):.1%}）---")
        print(f"{'协议':<34}{'保留候选数':>10}{'K* 微调次数':>12}{'召回':>8}")
        # 只用第一级
        for R1 in (0.02, 0.05, 0.10, 0.20, 0.30):
            n_keep = int(round(N_ALL * R1))
            rec = np.mean([rank1.get(t, 10**9) < n_keep for t in strong.ix])
            print(f"{'仅第一级 K=0，保留 ' + f'{R1:.0%}':<34}{n_keep:>10}{0:>12}{rec:>8.0%}")
            rows.append(dict(pool=pool, stage="L1", R1=R1, R2=np.nan,
                             n_keep=n_keep, n_finetune=0, recall=float(rec)))
        # 两级
        for R2 in (0.05, 0.10, 0.20, 0.35, 0.50, 1.00):
            n_keep = int(round(N_L1 * R2))
            rec = np.mean([(rank1.get(t, 10**9) < N_L1) and (rank2.get(t, 10**9) < n_keep)
                           for t in strong.ix])
            lab = f"两级：L1 {R1_ACTUAL:.0%} → L2 保留 {R2:.0%}"
            print(f"{lab:<34}{n_keep:>10}{N_L1:>12}{rec:>8.0%}")
            rows.append(dict(pool=pool, stage="L1+L2", R1=R1_ACTUAL, R2=R2,
                             n_keep=n_keep, n_finetune=N_L1, recall=float(rec)))

    pd.DataFrame(rows).to_csv(OUT / "h1_protocol.csv", index=False)

    # ---- 成本 ----
    print("\n" + "=" * 92)
    print("成本对比")
    print("=" * 92)
    print(f"  68 号原版：K=200 全扫 {N_ALL} 个           ≈ 5400 s（90 min）")
    print(f"  本协议第一级：K=0 批量前向 {N_ALL} 个        = 2 s")
    print(f"  本协议第二级：K={KSTAR} 微调 {N_L1} 个        ≈ 225 s（实测两分片各 ~112 s）")
    print(f"  → 合计 ≈ 227 s，相对原版省 {5400/227:.0f}×")
    log("H-1", "DECISION",
        note=f"两级协议（L1 K=0 全扫 {N_ALL} 个 2s → 保留 {R1_ACTUAL:.0%} → "
             f"L2 K*={KSTAR} 微调）合计 ≈227 s，相对 68 号 K=200 全扫省 ~24×")


if __name__ == "__main__":
    main()
