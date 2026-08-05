# -*- coding: utf-8 -*-
"""88 号 ★通道B v2：用【严格必要条件】剪枝，绕开层级假设。

数学（对任意阶 m 都成立，只需二元信息）：
  对任意真子集 U ⊊ S，存在 (m−1)-子集 T ⊇ U，由 v 单调得 v_c(T) ≥ v_c(U)
  ⟹ syn_c(S) = v_c(S) − max_{|T|=m−1} v_c(T) ≤ v_c(S) − max_{|U|=2, U⊊S} v_c(U)
  ⟹ **syn_c(S) > τ  ⟹  所有二元子集 U 满足 v_c(U) < v_c(S) − τ ≤ v_c(V) − τ**

与 beam 的根本区别：
  · beam：低阶 syn 强 → 高阶可能强（启发式，无保证，结构性漏反层级）
  · 本法：低阶 v 高 → 高阶 syn 必弱（**严格必要条件**，剪枝不漏任何强协同）

组合结构：令可行图 G_c = {(i,j) : v_c({i,j}) < v_c(V) − τ}，
则任何 syn_c(S) > τ 的集合 S 必是 G_c 上的**团**。于是高阶搜索
= 在 G_c 上枚举团（Bron–Kerbosch 等成熟算法），而非枚举 C(n,m)。

本脚本验证三件事：
  1. 可行图有多稀疏（剪枝力度）；
  2. 已知强四阶/五阶是否 100% 落在团里（验证不等式无漏，理论上必然，实测防实现错）；
  3. 团空间 vs C(44,m) 的压缩比（真实收益）。
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

TAUS = (0.05, 0.10, 0.15)


def load_enum(m: int) -> dict:
    df = pd.read_parquet(R86 / f"outputs/enum{m}_syn.parquet")
    col = "syn4" if m == 4 else "syn"
    return {tuple(int(v) for v in t): s for t, s in zip(df["indices"], df[col])}


def count_cliques_of_size(adj: np.ndarray, m: int, cap: int = 5_000_000) -> int:
    """计数 G 上大小为 m 的团（DFS 剪枝）。cap 为安全上限。"""
    n = adj.shape[0]
    cnt = 0

    def dfs(start: int, cur: list):
        nonlocal cnt
        if len(cur) == m:
            cnt += 1
            return
        if cnt > cap:
            return
        for v in range(start, n):
            if all(adj[v, u] for u in cur):
                cur.append(v)
                dfs(v + 1, cur)
                cur.pop()
    dfs(0, [])
    return cnt


def main() -> None:
    q = Poly2Query()
    n = q.n_general
    nc = len(q.conf_names)

    # ---- 二元 v（946×12）与全集 v ----
    t = time.perf_counter()
    v_pair = {}
    for pair in combinations(range(n), 2):
        v_pair[pair] = np.array(q.vhat(pair))
    v_full = np.array(q.vhat(tuple(range(n))))
    log("CLIQUE", "PAIRS", note=f"946 二元 + 全集 {time.perf_counter()-t:.1f}s")
    print(f"二元 v 计算完成 [{time.perf_counter()-t:.1f}s]；各 conf 全集 v_c(V):")
    print("  " + ", ".join(f"{q.conf_names[c][:18]}={v_full[c]:.3f}" for c in range(nc)))

    enum4, enum5 = load_enum(4), load_enum(5)
    rows = []
    for tau in TAUS:
        for c in range(nc):
            thr = v_full[c] - tau
            adj = np.zeros((n, n), dtype=bool)
            for (i, j), vv in v_pair.items():
                if vv[c] < thr:
                    adj[i, j] = adj[j, i] = True
            n_edges = int(adj.sum() // 2)
            rows.append(dict(tau=tau, conf=q.conf_names[c], v_full=round(float(v_full[c]), 4),
                             thr=round(float(thr), 4), edges=n_edges,
                             edge_frac=round(n_edges / 946, 4)))
    ed = pd.DataFrame(rows)
    ed.to_csv(ROOT / "outputs/feasible_graph_edges.csv", index=False)
    print(f"\n=== 可行图稀疏度（边数/946）===")
    print(ed.groupby("tau")["edge_frac"].describe()[["min", "50%", "max"]].to_string())

    # ---- 验证：已知强集合是否 100% 落在其 conf 的团里 ----
    print(f"\n=== 验证必要条件（应 100%，否则实现有误）+ 团空间压缩 ===")
    res = []
    for tau in (0.10,):
        for m, enum in ((4, enum4), (5, enum5)):
            strong = [S for S, v in enum.items() if v > tau]
            # 强集合的 conf：用 q.syn 取 argmax conf
            ok = 0
            for S in strong:
                _, c = q.syn(S)
                thr = v_full[c] - tau
                if all(v_pair[tuple(sorted(p))][c] < thr for p in combinations(S, 2)):
                    ok += 1
            res.append(dict(tau=tau, order=m, n_strong=len(strong), in_clique=ok,
                            frac=round(ok / max(len(strong), 1), 4)))
            log("CLIQUE", "VERIFY", note=f"order{m} tau={tau}: {ok}/{len(strong)} 落在团内")
    print(pd.DataFrame(res).to_string(index=False))

    # ---- 团空间大小（取边最少与中位的 conf 各一个）----
    print(f"\n=== 团空间 vs 全枚举（tau=0.10）===")
    tau = 0.10
    sizes = []
    for c in range(nc):
        thr = v_full[c] - tau
        adj = np.zeros((n, n), dtype=bool)
        for (i, j), vv in v_pair.items():
            if vv[c] < thr:
                adj[i, j] = adj[j, i] = True
        for m in (4, 5):
            k = count_cliques_of_size(adj, m)
            sizes.append(dict(conf=q.conf_names[c], order=m, edges=int(adj.sum() // 2),
                              cliques=k, full=len(list(combinations(range(1), 1))) if False else None))
    sd = pd.DataFrame(sizes)
    from math import comb
    sd["full_space"] = sd["order"].map(lambda m: comb(n, m))
    sd["compress"] = (sd["full_space"] / sd["cliques"].clip(lower=1)).round(1)
    sd.to_csv(ROOT / "outputs/clique_space.csv", index=False)
    print(sd.to_string(index=False))
    print(f"\n★ 合计（对所有 conf 求和，因每个 conf 独立搜索）：")
    for m in (4, 5):
        s = sd[sd.order == m]
        print(f"  order-{m}: 团总数 {int(s.cliques.sum()):,} vs 全枚举×12conf {comb(n,m)*nc:,} "
              f"⟹ 压缩 {comb(n,m)*nc/max(int(s.cliques.sum()),1):.1f}×")


if __name__ == "__main__":
    main()
