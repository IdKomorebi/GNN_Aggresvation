# -*- coding: utf-8 -*-
"""H-2 分析：扩容后的无偏池——基率、纯高阶占比、剪枝器召回与新的 Wilson CI。

扩容前（68 号 rand200）：无偏池只有 19 条真强三阶【条目】/ 14 个真强三阶【三元组】，
Wilson 95% CI ≈ [0.75, 0.99]，撑不起论文主张。
本子项目补 1800 个新随机三元组，把无偏池做到 2000。

同时用扩容后的池子重新评估 H-1 的两级协议——这是 H-1 中期发现
"第二级未显示优势"能否定论的关键（当时 n=14，差异只有 1 个样本）。
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

R77 = Path(__file__).resolve().parents[1]
R68 = R77.parent / "DNN_Aggresvation68"
R69 = R77.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R77 / "src"))
from src.data_processing import prepare_data   # noqa: E402
from runlog import log                          # noqa: E402

OUT = R77 / "outputs"
DELTA = 0.10


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    n2i = {n: i for i, n in enumerate(di["general"])}
    CONF = di["confidential"]

    # ---- 低阶重训真值（仅用于【评估】：算 syn3_true 与 Apriori 对照）----
    reg67 = json.load(open(R77.parent / "DNN_Aggresvation67/outputs/subsets.json"))
    t67 = {}
    for sid, m in reg67.items():
        p = R77.parent / "DNN_Aggresvation67/outputs/retrain" / f"{sid}_dnn_seed0.json"
        if p.exists():
            d = json.load(open(p))
            t67[tuple(sorted(n2i[f] for f in d["fields"]))] = d["per_conf_r2"]

    # ---- 新增的 1800 个随机三元组真值 ----
    rows = []
    for p in sorted((OUT / "retrain").glob("r*_dnn_seed0.json")):
        d = json.load(open(p))
        ix = tuple(sorted(n2i[f] for f in d["fields"]))
        pairs = [tuple(sorted(q)) for q in itertools.combinations(ix, 2)]
        for c in CONF:
            pv = [t67.get(q, {}).get(c) for q in pairs]
            if any(v is None for v in pv):
                continue
            rows.append(dict(ix=ix, conf=c, v_ijk=d["per_conf_r2"][c],
                             best_pair=max(pv),
                             syn3_true=d["per_conf_r2"][c] - max(pv),
                             syn2_true_max=max(
                                 max(pv[a] - max(t67.get((q[0],), {}).get(c, 0.0),
                                                 t67.get((q[1],), {}).get(c, 0.0))
                                     for a, q in enumerate(pairs)), 0.0),
                             group="triple_rand_ext"))
    new = pd.DataFrame(rows)

    # ---- 68 号原有的 rand200 ----
    old = pd.read_csv(R68 / "outputs/triples_certified.csv")
    old = old[old.group == "triple_rand"].copy()
    old["ix"] = [tuple(sorted((n2i[a], n2i[b], n2i[c])))
                 for a, b, c in zip(old.fi, old.fj, old.fk)]
    syn2t = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    t2 = {(min(r.fi, r.fj), max(r.fi, r.fj), r.conf): r.synergy for r in syn2t.itertuples()}
    old["syn2_true_max"] = [
        max([t2.get((min(a, b), max(a, b), c), 0.0)
             for a, b in itertools.combinations([r.fi, r.fj, r.fk], 2)])
        for r, c in zip(old.itertuples(), old.conf)]
    old = old[["ix", "conf", "syn3_true", "syn2_true_max"]].assign(group="triple_rand")

    pool = pd.concat([old, new[["ix", "conf", "syn3_true", "syn2_true_max", "group"]]],
                     ignore_index=True)
    pool.to_csv(OUT / "h2_unbiased_pool.csv", index=False)

    n_tri = pool.ix.nunique()
    strong = pool[pool.syn3_true > DELTA]
    n_strong_ent, n_ent = len(strong), len(pool)
    tri_strong = strong.ix.nunique()

    print("=" * 92)
    print("H-2：扩容后的无偏认证池")
    print("=" * 92)
    print(f"  三元组数        {n_tri}（68 号 200 + 本次 {new.ix.nunique()}）")
    print(f"  (三元组,conf) 条目 {n_ent}")
    print(f"  真强三阶条目     {n_strong_ent}   基率 {n_strong_ent/n_ent:.3%}"
          f"   [扩容前 19/2400 = 0.79%]")
    print(f"  含真强三阶的三元组 {tri_strong}   占 {tri_strong/n_tri:.2%}   [扩容前 14/200]")

    # ---- 纯高阶占比（Apriori 系的天花板）----
    pure = (strong.syn2_true_max <= DELTA).mean()
    lo, hi = wilson(int((strong.syn2_true_max <= DELTA).sum()), n_strong_ent)
    print(f"\n  纯高阶（无强二阶子对）占比 {pure:.1%}   Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  → Apriori 系剪枝器的天花板 recall = {1-pure:.1%}")
    print(f"  [扩容前：5.3%（1/19），CI 极宽]")

    # ---- 用扩容池重评两级协议 ----
    s3 = pd.read_csv(OUT / "triples_syn3_k0.csv")
    lvl1 = s3.groupby(["i", "j", "k"]).syn3_est_k0.max().sort_values(ascending=False)
    N_ALL = len(lvl1)
    rank1 = {t: r for r, t in enumerate(lvl1.index)}
    ks_files = sorted(OUT.glob("triples_kstar_shard*.csv"))
    rank2, N_L1, KSTAR = {}, 0, None
    if ks_files:
        d2 = pd.concat([pd.read_csv(p) for p in ks_files])
        KSTAR = int(d2.K.iloc[0])
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
        l2 = d2.dropna(subset=["syn3_est_kstar"]).groupby(
            ["i", "j", "k"]).syn3_est_kstar.max().sort_values(ascending=False)
        N_L1 = len(l2)
        rank2 = {t: r for r, t in enumerate(l2.index)}

    tri_ok = sorted(strong.ix.unique())
    print("\n" + "=" * 92)
    print(f"两级协议重评（扩容无偏池：{len(tri_ok)} 个真强三元组，扩容前 14）")
    print("=" * 92)
    print(f"{'协议':<36}{'保留候选数':>10}{'召回':>8}{'Wilson 95% CI':>20}")
    res = []
    for R1 in (0.02, 0.05, 0.10, 0.20, 0.30):
        n_keep = int(round(N_ALL * R1))
        hit = sum(rank1.get(t, 10**9) < n_keep for t in tri_ok)
        lo, hi = wilson(hit, len(tri_ok))
        print(f"{'仅第一级 K=0，保留 ' + f'{R1:.0%}':<36}{n_keep:>10}"
              f"{hit/len(tri_ok):>8.0%}   [{lo:.0%}, {hi:.0%}]")
        res.append(dict(stage="L1", R1=R1, n_keep=n_keep, recall=hit/len(tri_ok),
                        ci_lo=lo, ci_hi=hi))
    if rank2:
        for R2 in (0.10, 0.20, 0.35, 0.50, 1.00):
            n_keep = int(round(N_L1 * R2))
            hit = sum((rank1.get(t, 10**9) < N_L1) and (rank2.get(t, 10**9) < n_keep)
                      for t in tri_ok)
            lo, hi = wilson(hit, len(tri_ok))
            lab = f"两级：L1 {N_L1/N_ALL:.0%} → L2 {R2:.0%}"
            print(f"{lab:<36}{n_keep:>10}{hit/len(tri_ok):>8.0%}   [{lo:.0%}, {hi:.0%}]")
            res.append(dict(stage="L1+L2", R1=N_L1/N_ALL, R2=R2, n_keep=n_keep,
                            recall=hit/len(tri_ok), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(res).to_csv(OUT / "h2_protocol_recall.csv", index=False)

    log("H-2", "DECISION",
        note=f"无偏池扩到 {n_tri} 个三元组（真强 {tri_strong} 个 / {n_strong_ent} 条目，"
             f"基率 {n_strong_ent/n_ent:.2%}）；纯高阶 {pure:.1%} CI [{lo:.0%},{hi:.0%}]")


if __name__ == "__main__":
    main()
