# -*- coding: utf-8 -*-
"""80 号 第 1 步：从 79 号 K 网格明细，为每个三元组构造【多种候选排序键】。

背景（79 号热图暴露的反常）
---------------------------
真值最强档（syn3_true>0.20）在 K=0 的 Top30% 召回只有 4.7%，而中等档（0.10-0.15）有 85-95%。
逐三元组分解发现根因：**强协同的父集合估计 v̂(ijk) 被严重低估**（甚至低于 best_pair），
导致增量 syn3_est = v̂(ijk) − max v̂(pair) 被压成负数、排到全体后 3/4。
机制：强协同 = 三字段必须精确配合的除法/减法电路，容量需求最大、训练掩码出现率最低，
共享权重的摊销回归没给它分配容量（75_Correction §5 "干涉" 的极端形态）。

本脚本据此构造若干【只用 oracle 估计、不碰真值】的候选排序键，供第 2 步评估。
所有键都按 max over 12 conf 聚合到三元组级（发现口径：任一 conf 强即记录）。

排序键（各 K ∈ {0,5,10,25,50}）
------------------------------
S1  syn3_est          现状：v̂(ijk) − max v̂(pair)
S2  v_ijk             方向A：父集合联合泄露本身（强协同的 v(ijk) 真值一定高）
S3  v_ijk_minus_single  v̂(ijk) − max v̂(single)  （减单字段，减得少，保留更多父信号）
S4  syn3_over_headroom  syn3_est / (1 − max v̂(pair) + eps)  （相对天花板的归一化增量）
S5  v_ijk_minus_meanpair  v̂(ijk) − mean v̂(pair)
S6  rankfuse_vijk_syn3    v_ijk 百分位 与 syn3_est 百分位 的均值（融合"高泄露"与"有增量"）

输出：outputs/triple_scores.parquet —— indices, K, S1..S6, 以及真值 join 列
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

R80 = Path(__file__).resolve().parents[1]
R79 = R80.parent / "DNN_Aggresvation79" / "outputs"
sys.path.insert(0, str(R80 / "src"))
from runlog import log  # noqa: E402

KS = [0, 5, 10, 25, 50]


def main():
    log("BUILD", "START", note="构造候选排序键")

    # ---- 三阶明细：v_ijk / best_pair / syn3 per (triple, conf, K) ----
    e = pd.read_csv(R79 / "syn3_entries_kgrid.csv.gz")
    e = e[e.K.isin(KS)]

    # ---- 单字段 K 网格：per (field, conf, K) 的 v̂(single) ----
    lo = pd.read_csv(R79 / "kgrid_low_uniform_seed0_shard0of1.csv.gz")
    single = lo[lo["size"] == 1].copy()
    sing = {(int(r.i), r.conf, int(r.K)): r.est for r in single.itertuples()}

    # 每个三元组的三个单字段 max（per conf, per K）
    def max_single(i, j, k, conf, K):
        return max(sing.get((i, conf, K), 0.0), sing.get((j, conf, K), 0.0),
                   sing.get((k, conf, K), 0.0))

    # best_pair_est 已在 e 里（max over 三个二元子对）。mean_pair 需另算——
    # e 里只有 best_pair（max），没有三个 pair 的原始值；mean 用 kgrid_low 的 pair 补。
    pair = lo[lo["size"] == 2].copy()
    pr = {(int(r.i), int(r.j), r.conf, int(r.K)): r.est for r in pair.itertuples()}

    def mean_pair(i, j, k, conf, K):
        ps = [pr.get((min(a, b), max(a, b), conf, K)) for a, b in combinations((i, j, k), 2)]
        ps = [x for x in ps if x is not None]
        return float(np.mean(ps)) if ps else np.nan

    # ---- 逐 (triple, conf, K) 造 entry 级派生量，再 max over conf ----
    e["v_ijk"] = e["est"]
    e["max_single"] = [max_single(r.i, r.j, r.k, r.conf, r.K) for r in e.itertuples()]
    e["mean_pair"] = [mean_pair(r.i, r.j, r.k, r.conf, r.K) for r in e.itertuples()]
    e["s3_minus_single"] = e["v_ijk"] - e["max_single"]
    e["s3_minus_meanpair"] = e["v_ijk"] - e["mean_pair"]
    e["s3_over_head"] = e["syn3_est"] / (1.0 - e["best_pair_est"] + 1e-6)

    # entry -> triple（max over conf），各 K
    g = e.groupby(["i", "j", "k", "K"])
    tri = g.agg(S1_syn3=("syn3_est", "max"),
                S2_vijk=("v_ijk", "max"),
                S3_minus_single=("s3_minus_single", "max"),
                S4_over_head=("s3_over_head", "max"),
                S5_minus_meanpair=("s3_minus_meanpair", "max")).reset_index()
    tri["indices"] = list(zip(tri.i, tri.j, tri.k))

    # S6 rank-fusion（在同一 K 内，对全体 13244 三元组取百分位）
    for K in KS:
        m = tri.K == K
        rv = tri.loc[m, "S2_vijk"].rank(pct=True)
        rs = tri.loc[m, "S1_syn3"].rank(pct=True)
        tri.loc[m, "S6_rankfuse"] = (rv + rs) / 2

    # ---- join 真值（79 号 split）----
    sp = pd.read_csv(R79 / "truth_design_split.csv")
    sp["indices"] = sp["indices"].apply(eval)
    tri = tri.merge(sp[["indices", "syn3_true", "weight", "strong", "split"]],
                    on="indices", how="left")

    out = R80 / "outputs/triple_scores.parquet"
    tri.to_parquet(out)
    n_truth = tri[tri.K == 0].syn3_true.notna().sum()
    log("BUILD", "DONE", note=f"13244 三元组 × {len(KS)} K；带真值 {n_truth} 个", n_total=len(tri))
    print(f"写出 {out}，{len(tri)} 行；带真值三元组 {n_truth}")


if __name__ == "__main__":
    main()
