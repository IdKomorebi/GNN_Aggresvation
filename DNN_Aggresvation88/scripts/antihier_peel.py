# -*- coding: utf-8 -*-
"""88 号 ★反层级安全通道：极小性剥离 + 随机重启（不使用任何低阶信号）。

动机（对偶关系，本号的方法学核心）：
    syn_m(S) = v(S) − max_{T⊂S,|T|=m−1} v(T) > τ
  ⟺ 删掉 S 中任何一个元素，v 都下降 > τ
  ⟺ S 是"缺一不可"的极小充分集（irreducible / inclusion-minimal）

于是同一个 syn 有两种搜索方式，且性质对偶：
  · beam（自下而上）用【嵌套性】：需要强父集信号 → 只能找嵌套型；
  · peel（自上而下）用【极小性】：**完全不需要低阶信号** → 专找反层级型。

这修正了 80 号"剥离法不做发现工具"的判定——那个判定针对**嵌套型**成立
（若 {a,b} 已超阈，剥离会删掉 c，发现不了 abc）；对**纯高阶/反层级**则恰恰相反。

算法（per-conf，因 82 号已证协同是 conf-specific 的）：
    S ← 随机大集合 U
    repeat: 找 i* = argmax_i v_c(S\{i})（删掉损失最小的）
            若 v_c(S) − v_c(S\{i*}) > τ  → S 已极小，停
            否则 S ← S\{i*}
成本以【闭式求解次数】计量，与 beam 等预算比较。
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))
from query import Poly2Query  # noqa: E402
from runlog import log  # noqa: E402


def peel_one(q: Poly2Query, U: tuple[int, ...], c: int, tau: float, min_size: int = 3):
    """对目标 conf c，从 U 贪心剥离到极小充分集。返回 (S, syn_c(S))。"""
    S = tuple(sorted(U))
    while len(S) > min_size:
        vS = q.vhat(S)[c]
        best_i, best_v = None, -1e9
        for i in S:
            sub = tuple(x for x in S if x != i)
            v = q.vhat(sub)[c]
            if v > best_v:
                best_v, best_i = v, i
        if vS - best_v > tau:          # 删任何一个都大跌 ⟹ 已极小
            return S, vS - best_v
        S = tuple(x for x in S if x != best_i)
    vS = q.vhat(S)[c]
    best_v = max(q.vhat(tuple(x for x in S if x != i))[c] for i in S)
    return S, vS - best_v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=400)
    ap.add_argument("--usize", type=int, default=14, help="随机起始大集合尺寸")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="peel_results.csv")
    args = ap.parse_args()

    q = Poly2Query()
    n = q.n_general
    rng = np.random.RandomState(args.seed)
    start_solves = q.n_solve
    t0 = time.perf_counter()

    found = {}                       # S -> (syn, conf)
    for r in range(args.restarts):
        U = tuple(sorted(rng.choice(n, args.usize, replace=False)))
        for c in range(len(q.conf_names)):
            S, s = peel_one(q, U, c, args.tau)
            if s > args.tau and (S not in found or s > found[S][0]):
                found[S] = (float(s), c)
        if (r + 1) % 50 == 0:
            log("PEEL", "PROG", note=f"restart {r+1}/{args.restarts} 已找到 {len(found)} 个极小集; "
                                     f"solves={q.n_solve-start_solves}")

    solves = q.n_solve - start_solves
    rows = [dict(indices=S, size=len(S), syn_peel=v, conf=c) for S, (v, c) in found.items()]
    df = pd.DataFrame(rows).sort_values("syn_peel", ascending=False)
    df.to_csv(ROOT / "outputs" / args.out, index=False)

    log("PEEL", "DONE", note=f"restarts={args.restarts} usize={args.usize} tau={args.tau} "
                             f"solves={solves} 用时{time.perf_counter()-t0:.0f}s 找到{len(df)}个")
    print(f"\n剥离通道：{args.restarts} 次重启 × {len(q.conf_names)} conf，"
          f"闭式求解 {solves} 次，用时 {time.perf_counter()-t0:.0f}s")
    print(f"找到极小充分集 {len(df)} 个，尺寸分布：")
    print(df["size"].value_counts().sort_index().to_string())
    print(f"\nsyn 最强 10 个：")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
