# -*- coding: utf-8 -*-
"""88 号 诊断二：反层级集合还剩什么可利用信号？（决定通道 B 该怎么设计）

背景：通道 B v1（极小性剥离）实测失败——贪心剥离被强低阶结构吸引，
48 次重启只产出 6 个候选，盲区召回 0.000。所以必须换设计。
换之前必须先回答：**反层级强集合身上还有没有任何非低阶的信号？**

查三件事（全用 order-4/5 全枚举真值，零 GPU）：
  A. 反层级强集合的 v(S)（联合泄露绝对值）是否显著高于随机集合？
     → 若是，"贪心最大化 v(S)"（单调近似子模，(1−1/e) 保证）可作通道 B。
  B. 它们的 v(S) 与嵌套型强集合相比如何？
  C. 若按 v(S) 全局排名，反层级集合排在多靠前？（决定候选预算）
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R86 = REPO / "DNN_Aggresvation86"
sys.path.insert(0, str(ROOT / "src"))
from query import Poly2Query  # noqa: E402
from runlog import log  # noqa: E402

TAU = 0.10
BEAM = 1000


def load_enum(m: int) -> dict:
    df = pd.read_parquet(R86 / f"outputs/enum{m}_syn.parquet")
    col = "syn4" if m == 4 else "syn"
    return {tuple(int(v) for v in t): s for t, s in zip(df["indices"], df[col])}


def main() -> None:
    q = Poly2Query()
    n = q.n_general
    enum4 = load_enum(4)
    syn3 = {S: q.syn(S)[0] for S in combinations(range(n), 3)}
    rank3 = {S: i + 1 for i, (S, _) in enumerate(sorted(syn3.items(), key=lambda kv: -kv[1]))}

    strong = [S for S, v in enum4.items() if v > TAU]
    blind, nested = [], []
    for S in strong:
        best = min(rank3[tuple(sorted(T))] for T in combinations(S, 3))
        (blind if best > BEAM else nested).append(S)
    log("DIAG2", "SPLIT", note=f"强四阶 {len(strong)}: 盲区 {len(blind)} / 嵌套 {len(nested)}")

    # v(S) = max_c v̂_c(S)
    t = time.perf_counter()
    v_blind = np.array([q.vmax(S) for S in blind])
    v_nested = np.array([q.vmax(S) for S in nested])
    rng = np.random.RandomState(0)
    rand_sets = [tuple(sorted(rng.choice(n, 4, replace=False))) for _ in range(3000)]
    v_rand = np.array([q.vmax(S) for S in rand_sets])
    log("DIAG2", "VMAX", note=f"{time.perf_counter()-t:.0f}s")

    print(f"\n=== A/B. v(S)=max_c v̂_c(S) 分布对比（强四阶 τ>{TAU}）===")
    for name, arr in (("盲区强四阶", v_blind), ("嵌套强四阶", v_nested), ("随机四阶", v_rand)):
        print(f"  {name:10s} n={len(arr):5d}  中位={np.median(arr):.4f}  "
              f"均值={arr.mean():.4f}  25%={np.percentile(arr,25):.4f}  75%={np.percentile(arr,75):.4f}")

    # C. 反层级集合按 v(S) 排序的全局百分位（用随机样本估分布）
    print(f"\n=== C. 盲区强四阶在 v(S) 全局分布中的百分位 ===")
    pct_blind = np.array([(v_rand < v).mean() for v in v_blind]) * 100
    pct_nested = np.array([(v_rand < v).mean() for v in v_nested]) * 100
    print(f"  盲区强四阶：中位百分位 {np.median(pct_blind):.1f}%  "
          f"（>90 百分位的占 {(pct_blind>90).mean():.0%}）")
    print(f"  嵌套强四阶：中位百分位 {np.median(pct_nested):.1f}%  "
          f"（>90 百分位的占 {(pct_nested>90).mean():.0%}）")

    out = pd.DataFrame(dict(
        group=["blind"] * len(blind) + ["nested"] * len(nested),
        indices=[str(s) for s in blind + nested],
        syn4=[enum4[s] for s in blind + nested],
        vmax=list(v_blind) + list(v_nested),
        vmax_pct=list(pct_blind) + list(pct_nested)))
    out.to_csv(ROOT / "outputs/blind_signal.csv", index=False)

    print(f"\n=== 判定 ===")
    if np.median(pct_blind) > 85:
        print("  盲区集合的 v(S) 显著偏高 ⟹ 贪心最大化 v(S)（单调子模，(1−1/e)）可作通道B")
    elif np.median(pct_blind) > 60:
        print("  盲区集合 v(S) 中等偏高 ⟹ v(S) 可作弱信号，需与其他机制结合")
    else:
        print("  盲区集合 v(S) 无区分度 ⟹ 无局部信号可用，只能靠随机/稀疏恢复类方法")


if __name__ == "__main__":
    main()
