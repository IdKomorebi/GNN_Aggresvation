# -*- coding: utf-8 -*-
"""高阶协同搜索：把"真值在环"的做法换成"纯 oracle"，看代价多大。

问题
----
68 号的三阶搜索链条里，Apriori 剪枝器用的是【低阶重训真值】的二阶协同
（syn2_true > δ 才保留该三元组）。70 号据此报了 recall 85% / 2.4× 加速。
但按红线，重训真值只能用于评估、不能进查询路径 —— 部署时没人有那 946 个字段对的真值
（约 107 GPU·h）。所以这条链的诚实版本必须把剪枝器的输入换成 **oracle 估计的二阶协同**。

本脚本在 68 号已认证的 397 个三元组上，对比四种候选排序/剪枝器的
"保留率 → 召回率"曲线（召回 = 真强三阶 syn3_true>0.1 被保留的比例）：

  A. Apriori(真值)   : 用 syn2_true 排序        —— 68/70 号原版，真值在环，不合法
  B. Apriori(oracle) : 用 oracle 估计的 syn2 排序 —— 诚实版
  C. 直接 oracle 三阶 : 用 oracle 估计的 syn3 排序 —— 不用低阶，直接扫
  D. 随机            : 下界

同时分开报【有偏池 triple_top】与【无偏池 triple_rand】——
70 号的 85% 是在有偏池上算的（候选本就由估计器挑出），无偏池才是可外推的口径。

用法：python scripts/highorder_search_honest.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[2]
R75, R68 = R / "DNN_Aggresvation75", R / "DNN_Aggresvation68"
OUT = Path(__file__).resolve().parents[1] / "outputs"
DELTA = 0.10          # "显著三阶协同"阈值，与 68/70 号一致
KS = [0, 10, 50, 200]


def load_est(seed=0):
    ev = json.load(open(R75 / "outputs/evalset.json"))
    key2sid = {frozenset(m["fields"]): s for s, m in ev.items()}
    d = pd.read_csv(R75 / f"outputs/est_uniform_seed{seed}.csv")
    est = {K: d[d.K == K].set_index(["sid", "conf"]).est.to_dict() for K in KS}
    return key2sid, est


def build(seed=0):
    key2sid, est = load_est(seed)
    tri = pd.read_csv(R68 / "outputs/triples_certified.csv")
    syn2t = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    t2 = {(min(r.fi, r.fj), max(r.fi, r.fj), r.conf): r.synergy for r in syn2t.itertuples()}

    rows = []
    for r in tri.itertuples():
        F = [r.fi, r.fj, r.fk]
        sid3 = key2sid.get(frozenset(F))
        pairs = [tuple(sorted(p)) for p in itertools.combinations(F, 2)]
        sid2 = [key2sid.get(frozenset(p)) for p in pairs]
        sid1 = [key2sid.get(frozenset([f])) for f in F]
        if sid3 is None or any(s is None for s in sid2 + sid1):
            continue
        rec = dict(group=r.group, conf=r.conf, syn3_true=r.syn3_true)
        # A：真值二阶协同（不合法，仅作对照）
        v = [t2.get((p[0], p[1], r.conf), np.nan) for p in pairs]
        rec["syn2_true_max"] = np.nanmax(v) if not all(np.isnan(v)) else np.nan
        # B/C：oracle 估计
        for K in KS:
            e = est[K]
            v3 = e.get((sid3, r.conf))
            v2 = [e.get((s, r.conf)) for s in sid2]
            v1 = [e.get((s, r.conf)) for s in sid1]
            if v3 is None or any(x is None for x in v2 + v1):
                continue
            # oracle 估计的二阶协同（每个子对）：v̂(ij) − max(v̂(i), v̂(j))
            s2e = [v2[a] - max(v1[i], v1[j])
                   for a, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)])]
            rec[f"syn2_est_max_K{K}"] = max(s2e)
            rec[f"syn3_est_K{K}"] = v3 - max(v2)
        rows.append(rec)
    return pd.DataFrame(rows).dropna()


def recall_curve(df, score, retentions=(0.05, 0.10, 0.20, 0.30, 0.42, 0.50)):
    """按 score 降序保留前 r 比例，返回各保留率下真强三阶的召回率。"""
    strong = df.syn3_true > DELTA
    if strong.sum() == 0:
        return {r: np.nan for r in retentions}
    order = np.argsort(-df[score].values)
    out = {}
    for r in retentions:
        n = max(1, int(round(len(df) * r)))
        keep = np.zeros(len(df), bool)
        keep[order[:n]] = True
        out[r] = keep[strong.values].mean()
    return out


def main():
    df = build()
    rng = np.random.RandomState(0)
    df["random"] = rng.random_sample(len(df))
    df.to_csv(OUT / "highorder_candidates.csv", index=False)

    RET = (0.05, 0.10, 0.20, 0.30, 0.42, 0.50)
    for pool, lab in [("triple_top", "有偏池 triple_top（68/70 号原口径）"),
                      ("triple_rand", "无偏池 triple_rand（可外推口径）"),
                      (None, "合并")]:
        d = df if pool is None else df[df.group == pool]
        strong = (d.syn3_true > DELTA).sum()
        print("=" * 96)
        print(f"{lab}   n={len(d)}  真强三阶(syn3>{DELTA})={strong}  基率={strong/max(len(d),1):.2%}")
        print("=" * 96)
        if strong == 0:
            print("  （无真强三阶，跳过）\n")
            continue
        print(f"{'剪枝器 / 排序依据':<34}" + "".join(f"{int(r*100):>7}%" for r in RET) + f"{'  ρ(vs真值)':>12}")
        cands = [("A. Apriori(真值 syn2)  ← 不合法", "syn2_true_max")]
        cands += [(f"B. Apriori(oracle syn2, K={K})", f"syn2_est_max_K{K}") for K in KS]
        cands += [(f"C. 直接 oracle syn3   (K={K})", f"syn3_est_K{K}") for K in KS]
        cands += [("D. 随机", "random")]
        for name, col in cands:
            if col not in d.columns:
                continue
            c = recall_curve(d, col, RET)
            rho = spearmanr(d[col], d.syn3_true).correlation
            print(f"{name:<34}" + "".join(f"{c[r]:>7.0%}" for r in RET) + f"{rho:>12.3f}")
        print()

    # 纯高阶（无强二阶子对）的占比 —— 决定 Apriori 类方法的天花板
    print("=" * 96)
    print("纯高阶三阶协同的占比（无强二阶子对 → Apriori 系必漏）")
    print("=" * 96)
    for pool in ("triple_top", "triple_rand"):
        d = df[(df.group == pool) & (df.syn3_true > DELTA)]
        if not len(d):
            continue
        pure = (d.syn2_true_max <= DELTA).mean()
        print(f"  {pool:<14} 真强三阶 {len(d):>3} 条，其中纯高阶 {pure:.1%} "
              f"（Apriori 天花板 recall = {1-pure:.1%}）")
    print(f"\n已写出 {OUT / 'highorder_candidates.csv'}")


if __name__ == "__main__":
    main()
