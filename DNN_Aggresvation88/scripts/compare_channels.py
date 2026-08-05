# -*- coding: utf-8 -*-
"""88 号 决定性验证：双通道 vs 单 beam —— 剥离通道能否补上 beam 的结构性盲区？

等【闭式求解次数】预算比较（不是等集合数，因为两个通道每个集合的查询数不同）。
用 order-4/5 全枚举真值评估，重点看三个数：
  1. beam 覆盖率（已知：order-4 0.896 / order-5 0.649）
  2. peel 覆盖率（peel 完全不用低阶信号）
  3. **在 beam 盲区子集上（best_parent_rank > B，beam 结构上不可达），peel 的覆盖率**
     ——这是"双通道是否真的互补"的唯一决定性证据。

随机基线按 GPT 指出的修正：**无放回、等集合预算、多种子**。
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
from search_highorder import beam_search  # noqa: E402
from antihier_peel import peel_one  # noqa: E402
from runlog import log  # noqa: E402


def load_enum(m: int) -> dict:
    df = pd.read_parquet(R86 / f"outputs/enum{m}_syn.parquet")
    col = "syn4" if m == 4 else "syn"
    return {tuple(int(v) for v in t): s for t, s in zip(df["indices"], df[col])}


def rand_baseline(n: int, m: int, budget: int, seeds: int, strong: set) -> tuple[float, float]:
    """无放回、等集合预算、多种子的随机基线：返回 (均值覆盖率, 标准差)。"""
    covs = []
    for s in range(seeds):
        rng = np.random.RandomState(1000 + s)
        picked = set()
        while len(picked) < budget:
            picked.add(tuple(sorted(rng.choice(n, m, replace=False))))
        covs.append(len(picked & strong) / max(len(strong), 1))
    return float(np.mean(covs)), float(np.std(covs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--usize", type=int, default=14)
    ap.add_argument("--orders", default="4,5")
    args = ap.parse_args()

    q = Poly2Query()
    n = q.n_general
    enum = {m: load_enum(m) for m in (4, 5)}
    # 三阶/四阶的全局 syn 排名（判定 beam 盲区用）
    syn3 = {S: q.syn(S)[0] for S in combinations(range(n), 3)}
    rank = {}
    for m, tbl in ((3, syn3), (4, enum[4])):
        rank[m] = {S: i + 1 for i, (S, _) in enumerate(sorted(tbl.items(), key=lambda kv: -kv[1]))}

    rows = []
    for m in [int(x) for x in args.orders.split(",")]:
        strong = {S for S, v in enum[m].items() if v > args.tau}
        # 盲区：所有 (m-1)-父集的排名都 > B
        blind = set()
        for S in strong:
            best = min(rank[m - 1][tuple(sorted(T))] for T in combinations(S, m - 1))
            if best > args.beam:
                blind.add(S)

        # ---- 通道 A：beam ----
        base = q.n_solve
        t = time.perf_counter()
        touched, _ = beam_search(q, n, args.beam, "syn", target_size=m)
        beam_solves = q.n_solve - base
        beam_t = time.perf_counter() - t
        Sb = touched[m]

        # ---- 通道 B：peel（等求解预算）----
        base = q.n_solve
        t = time.perf_counter()
        rng = np.random.RandomState(0)
        peel_found = {}
        restarts = 0
        while q.n_solve - base < beam_solves:
            U = tuple(sorted(rng.choice(n, args.usize, replace=False)))
            for c in range(len(q.conf_names)):
                S, s = peel_one(q, U, c, args.tau)
                if s > args.tau:
                    peel_found[S] = max(s, peel_found.get(S, -9))
            restarts += 1
        peel_solves = q.n_solve - base
        peel_t = time.perf_counter() - t
        Sp = {S for S in peel_found if len(S) == m}

        rmean, rstd = rand_baseline(n, m, len(Sb), 10, strong)
        row = dict(
            order=m, tau=args.tau, n_strong=len(strong), n_blind=len(blind),
            blind_frac=round(len(blind) / len(strong), 4),
            beam_solves=beam_solves, beam_sets=len(Sb), beam_s=round(beam_t, 1),
            peel_solves=peel_solves, peel_restarts=restarts, peel_sets=len(Sp),
            peel_s=round(peel_t, 1),
            cov_beam=round(len(Sb & strong) / len(strong), 4),
            cov_peel=round(len(Sp & strong) / len(strong), 4),
            cov_union=round(len((Sb | Sp) & strong) / len(strong), 4),
            cov_rand=round(rmean, 4), rand_std=round(rstd, 4),
        )
        if blind:
            row["blind_cov_beam"] = round(len(Sb & blind) / len(blind), 4)
            row["blind_cov_peel"] = round(len(Sp & blind) / len(blind), 4)
        rows.append(row)
        log("CMP", "ROW", note=f"order{m} blind={len(blind)}/{len(strong)} "
                               f"cov beam={row['cov_beam']} peel={row['cov_peel']} "
                               f"union={row['cov_union']} blind_peel={row.get('blind_cov_peel')}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/channel_comparison.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
