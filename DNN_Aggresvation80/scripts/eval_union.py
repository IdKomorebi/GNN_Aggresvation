# -*- coding: utf-8 -*-
"""80 号 第 3 步：既然 syn3 和 v(ijk) 盲区互补，测【并集/融合】排序方案。

第 2 步的核心发现
-----------------
- syn3（现状）：全体强协同 Top30% 召回 82.6%，但【最强档 >0.20】只有 4.7%（结构性漏最强）；
- v(ijk)（方向A）：最强档 81.1%，但全体只有 40.5%（把"三字段各自都强但无交互"的误当协同）。
两者盲区互补 → 单键无解，测组合。

组合方案（都只用 oracle 估计）
------------------------------
U1  并集：syn3 的 Top-a% ∪ v(ijk) 的 Top-b%（总预算 = a+b 去重后）
U2  min-rank：每个三元组取 (syn3 分位, v(ijk) 分位) 的较好者，按它排序
U3  两阈值：v(ijk) 必须 ≥ 某分位（保证是高泄露）AND syn3 ≥ 某分位（保证有增量）→ 太严，作对照
U4  条件并集：v(ijk) 极高（top 5%）的直接入选（不管 syn3）+ 其余按 syn3 排

关注：在【同等总候选预算】下，能否让最强档和全体都不塌。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R80 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R80 / "src"))
from runlog import log  # noqa: E402

BANDS = [(0.10, 0.12, "0.10-0.12"), (0.12, 0.15, "0.12-0.15"),
         (0.15, 0.20, "0.15-0.20"), (0.20, 1.01, ">0.20")]


def load(K):
    df = pd.read_parquet(R80 / "outputs/triple_scores.parquet")
    df = df[df.K == K].dropna(subset=["syn3_true"]).copy()
    df["indices"] = df["indices"].apply(lambda x: tuple(int(v) for v in x))
    # 全体 13244 的分位（排序键在全空间取分位）
    full = pd.read_parquet(R80 / "outputs/triple_scores.parquet")
    full = full[full.K == K]
    df["pct_syn3"] = df["S1_syn3"].rank(pct=True) if False else None  # 占位
    # 用全空间分位映射
    df["pct_syn3"] = df["S1_syn3"].map(lambda v: (full["S1_syn3"] < v).mean())
    df["pct_vijk"] = df["S2_vijk"].map(lambda v: (full["S2_vijk"] < v).mean())
    return df, len(full)


def recall_of(df, kept_idx, lo, hi):
    tgt = df[(df.syn3_true >= lo) & (df.syn3_true < hi) & df.strong]
    if tgt.weight.sum() == 0:
        return np.nan, 0
    hit = tgt[tgt.indices.isin(kept_idx)].weight.sum()
    return hit / tgt.weight.sum(), len(tgt)


def report(df, n_full, kept_idx, tag, budget):
    line = f"{tag:<40}预算{budget/n_full:>5.0%}"
    for lo, hi, bn in BANDS:
        r, n = recall_of(df, kept_idx, lo, hi)
        line += f"  {bn}={r:.2f}"
    ra, _ = recall_of(df, kept_idx, 0.10, 1.01)
    line += f"  |全体={ra:.3f}"
    print(line)
    return ra


def main():
    K = 0     # 核心：能不能在【免费 K=0】就解决
    df, n_full = load(K)
    full = pd.read_parquet(R80 / "outputs/triple_scores.parquet")
    full = full[full.K == K].copy()
    full["indices"] = full["indices"].apply(lambda x: tuple(int(v) for v in x))

    print("=" * 108)
    print(f"K={K}（免费）组合排序方案，各在【全体 13244】上取候选，评估加权召回分档")
    print("=" * 108)
    print(f"{'方案':<40}{'预算':>7}  {'各强度档召回（0.10-0.12 / 0.12-0.15 / 0.15-0.20 / >0.20）':<52}{'全体':>8}")

    def topset(col, frac):
        n = int(round(n_full * frac))
        return set(full.nlargest(n, col).indices)

    # 单键基线
    report(df, n_full, topset("S1_syn3", 0.30), "S1 syn3 单键 Top30%", int(n_full * 0.30))
    report(df, n_full, topset("S2_vijk", 0.30), "S2 v(ijk) 单键 Top30%", int(n_full * 0.30))
    report(df, n_full, topset("S4_over_head", 0.30), "S4 syn3/天花板 单键 Top30%", int(n_full * 0.30))

    print("-" * 108)
    # U1 并集：syn3 Top25% ∪ v(ijk) Top10%
    for a, b in [(0.25, 0.05), (0.25, 0.10), (0.30, 0.05), (0.20, 0.10)]:
        u = topset("S1_syn3", a) | topset("S2_vijk", b)
        report(df, n_full, u, f"U1 并集 syn3·Top{a:.0%} ∪ v(ijk)·Top{b:.0%}", len(u))

    print("-" * 108)
    # U2 min-rank（取两个分位的较小名次=较好），按融合分位排 Top30%
    df["u2"] = np.minimum(1 - df["pct_syn3"], 1 - df["pct_vijk"])  # 越小越好
    full_pct_s = full["S1_syn3"].rank(pct=True)
    full_pct_v = full["S2_vijk"].rank(pct=True)
    full["u2"] = np.minimum(1 - full_pct_s, 1 - full_pct_v)
    u2 = set(full.nsmallest(int(n_full * 0.30), "u2").indices)
    report(df, n_full, u2, "U2 min-rank(syn3,vijk) Top30%", len(u2))

    # U4 条件并集：v(ijk) top5% 直接入 + 其余 syn3 补到 30%
    must = topset("S2_vijk", 0.05)
    rest_budget = int(n_full * 0.30) - len(must)
    rest = [i for i in full.nlargest(n_full, "S1_syn3").indices if i not in must][:rest_budget]
    u4 = must | set(rest)
    report(df, n_full, u4, "U4 v(ijk)top5%强制入 + syn3补至30%", len(u4))

    print("=" * 108)
    print("结论：看哪个方案能让【>0.20 档】和【全体】同时不塌")
    log("UNION", "DONE", note="组合排序方案评估完成，见 outputs/union_eval（stdout）")


if __name__ == "__main__":
    main()
