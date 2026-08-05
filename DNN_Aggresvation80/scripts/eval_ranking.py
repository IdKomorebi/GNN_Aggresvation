# -*- coding: utf-8 -*-
"""80 号 第 2 步：评估各候选排序键——重点看【最强档】能否被救回。

评估口径（继承 79 号，加权 + test split）
------------------------------------------
- 真值分层加权：397 精确层 weight=1，1800 随机层 weight=7.137；
- 报告用 test split（避免选择键时偷看），也附全体加权；
- 强度档：0.10-0.12 / 0.12-0.15 / 0.15-0.20 / >0.20；
- 主指标：Recall@Top-X%（X ∈ {10,20,30,60}），排序键在【全部 13244】上取分位。

红线：所有排序键只用 oracle 估计（build_scores 已保证），真值只在评估时用。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R80 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R80 / "src"))
from runlog import log  # noqa: E402

KEYS = {"S1_syn3": "现状 syn3=v(ijk)−max·pair",
        "S2_vijk": "★方向A v(ijk) 联合泄露",
        "S3_minus_single": "v(ijk)−max·single",
        "S4_over_head": "syn3/剩余天花板",
        "S5_minus_meanpair": "v(ijk)−mean·pair",
        "S6_rankfuse": "融合 v(ijk)+syn3 排名"}
BANDS = [(0.10, 0.12, "0.10-0.12"), (0.12, 0.15, "0.12-0.15"),
         (0.15, 0.20, "0.15-0.20"), (0.20, 1.01, ">0.20")]
RET = [0.10, 0.20, 0.30, 0.60]


def wrecall(df, key, K, ret, lo, hi, split=None):
    """加权召回：真值∈[lo,hi) 且 strong 的三元组，被排序键 Top-ret 收进的加权比例。"""
    all_k = df[df.K == K]
    n_keep = int(round(len(all_k) * ret))
    kept = set(all_k.nlargest(n_keep, key).indices)
    tgt = all_k[(all_k.syn3_true >= lo) & (all_k.syn3_true < hi) & all_k.strong]
    if split:
        tgt = tgt[tgt.split == split]
    if tgt.weight.sum() == 0:
        return np.nan, 0
    hit = tgt[tgt.indices.isin(kept)].weight.sum()
    return hit / tgt.weight.sum(), len(tgt)


def main():
    df = pd.read_parquet(R80 / "outputs/triple_scores.parquet")
    df = df.dropna(subset=["syn3_true"]).copy()
    df["indices"] = df["indices"].apply(lambda x: tuple(int(v) for v in x))

    rows = []
    for key in KEYS:
        for K in sorted(df.K.unique()):
            for ret in RET:
                for lo, hi, bn in BANDS:
                    r_all, n = wrecall(df, key, K, ret, lo, hi)
                    r_te, nte = wrecall(df, key, K, ret, lo, hi, "test")
                    rows.append(dict(key=key, K=int(K), ret=ret, band=bn,
                                     recall_all=r_all, recall_test=r_te, n=n, n_test=nte))
    res = pd.DataFrame(rows)
    res.to_csv(R80 / "outputs/ranking_recall.csv", index=False)

    # ============ 打印：核心问题——最强档 >0.20 的召回，各键 × 各 K ============
    print("=" * 96)
    print("最强档（syn3_true>0.20）Top30% 召回：排序键 × K（加权，全体）")
    print("现状 S1 在此档 K=0 只有 ~5%——这是 79 号热图的反常")
    print("=" * 96)
    piv = res[(res.band == ">0.20") & (res.ret == 0.30)].pivot(
        index="key", columns="K", values="recall_all").reindex(KEYS)
    piv.index = [KEYS[k] for k in piv.index]
    print(piv.round(3).to_string())

    print("\n" + "=" * 96)
    print("全体强协同（>0.10）Top30% 召回：排序键 × K（加权，全体）")
    print("=" * 96)
    # 用整体（>0.10）：对每个 key,K 算加权总召回
    rows2 = []
    for key in KEYS:
        for K in sorted(df.K.unique()):
            r, n = wrecall(df, key, K, 0.30, 0.10, 1.01)
            rows2.append(dict(key=key, K=int(K), recall=r))
    p2 = pd.DataFrame(rows2).pivot(index="key", columns="K", values="recall").reindex(KEYS)
    p2.index = [KEYS[k] for k in p2.index]
    print(p2.round(3).to_string())

    # ============ 分档全景：S1 vs S2 在 K=0 ============
    print("\n" + "=" * 96)
    print("K=0（免费）分档 Top30% 召回：现状 S1 vs 方向A S2")
    print("=" * 96)
    print(f"{'强度档':<12}{'样本n':>7}{'S1 现状':>10}{'S2 v(ijk)':>11}{'提升':>8}")
    for lo, hi, bn in BANDS:
        r1, n = wrecall(df, "S1_syn3", 0, 0.30, lo, hi)
        r2, _ = wrecall(df, "S2_vijk", 0, 0.30, lo, hi)
        print(f"{bn:<12}{n:>7}{r1:>10.3f}{r2:>11.3f}{r2-r1:>+8.3f}")

    # 最优键 + 结论
    best_strong = res[(res.band == ">0.20") & (res.ret == 0.30) & (res.K == 0)].sort_values(
        "recall_all", ascending=False)
    bk = best_strong.iloc[0]
    log("EVAL", "DECISION",
        note=f"最强档 K=0 Top30%：现状 S1={piv.loc[KEYS['S1_syn3'],0]:.3f}，"
             f"最优键 {bk.key}={bk.recall_all:.3f}")
    print(f"\n→ K=0 最强档最优排序键：{bk.key}（{KEYS[bk.key]}）召回 {bk.recall_all:.3f}"
          f"（现状 S1 仅 {piv.loc[KEYS['S1_syn3'],0]:.3f}）")


if __name__ == "__main__":
    main()
