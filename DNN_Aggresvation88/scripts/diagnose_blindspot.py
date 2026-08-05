# -*- coding: utf-8 -*-
"""88 号 诊断：量化 beam 的【结构性盲区】——反层级高阶协同占多少？

用户的核心质疑（正确）：beam 默认"强 m 阶由强 (m-1) 阶扩展而来"，
会漏掉"所有低阶父集都不强、完整高阶集合突然很强"的组合。
而且这种层级假设已被 ProxySPEX(2025) 等使用，不能单独作为主创新。

本脚本用**全枚举真值**精确回答盲区有多大（零 GPU）：

对每个强 m 阶集合 S（syn_m > τ）：
  - 算它全部 m 个 (m-1)-子集的 syn_{m-1}；
  - 取 best_parent_rank = min over 子集的【全局排名】（在全部 (m-1) 阶集合的 syn 降序中）；
  - beam 宽 B 能到达 S 的**必要条件**是 best_parent_rank ≤ B
    （否则没有任何父集进得了第 (m-1) 层的 beam，S 结构上不可达）。

⟹ 反层级占比 = #{S : best_parent_rank > B} / #{强 S}   —— 这就是 beam 调参也修不好的盲区。

注意：本诊断只用于刻画**结构**（嵌套与否），不作性能声明；
估计器的 test-split 问题（GPT 指出，已接受）在正式实验中另行修正，
不影响"嵌套结构是否存在"这一定性结论。
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

BEAMS = (200, 500, 1000, 2000, 5000)
TAUS = (0.05, 0.10, 0.15)


def load_enum(m: int):
    p = R86 / f"outputs/enum{m}_syn.parquet"
    df = pd.read_parquet(p)
    col = "syn4" if m == 4 else "syn"
    sets = [tuple(int(v) for v in t) for t in df["indices"]]
    return dict(zip(sets, df[col].to_numpy()))


def main() -> None:
    q = Poly2Query()
    n = q.n_general

    # ---- 各阶 syn 表：3 阶现算，4/5 阶复用 86 号全枚举 ----
    t = time.perf_counter()
    syn = {3: {S: q.syn(S)[0] for S in combinations(range(n), 3)}}
    log("DIAG", "SYN3", note=f"13244 三阶 {time.perf_counter()-t:.0f}s")
    syn[4] = load_enum(4)
    syn[5] = load_enum(5)

    # 各阶的全局排名（syn 降序，rank 从 1 开始）
    rank = {}
    for m in (3, 4):
        items = sorted(syn[m].items(), key=lambda kv: -kv[1])
        rank[m] = {S: i + 1 for i, (S, _) in enumerate(items)}

    rows = []
    for m in (4, 5):
        parent_m = m - 1
        for tau in TAUS:
            strong = [S for S, v in syn[m].items() if v > tau]
            if not strong:
                continue
            best_rank = []
            best_parent_syn = []
            for S in strong:
                rs, vs = [], []
                for T in combinations(S, parent_m):
                    T = tuple(sorted(T))
                    rs.append(rank[parent_m][T])
                    vs.append(syn[parent_m][T])
                best_rank.append(min(rs))          # 最好的父集排名（越小越容易被 beam 保留）
                best_parent_syn.append(max(vs))
            best_rank = np.array(best_rank)
            row = dict(order=m, tau=tau, n_strong=len(strong),
                       parent_space=len(syn[parent_m]),
                       med_best_parent_rank=int(np.median(best_rank)),
                       med_best_parent_syn=round(float(np.median(best_parent_syn)), 4))
            for B in BEAMS:
                row[f"blind_B{B}"] = round(float((best_rank > B).mean()), 4)
            rows.append(row)
            log("DIAG", "ROW", note=f"order{m} tau={tau} n={len(strong)} "
                                    f"medRank={row['med_best_parent_rank']} "
                                    f"blindB1000={row['blind_B1000']}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/blindspot.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print(out.to_string(index=False))

    # ---- 反层级样例：父集排名最差的强四阶 ----
    print("\n=== 反层级样例（强四阶中，其最好三元父集排名最靠后的 8 个）===")
    strong4 = [(S, v) for S, v in syn[4].items() if v > 0.10]
    ex = []
    for S, v in strong4:
        rs = [(rank[3][tuple(sorted(T))], syn[3][tuple(sorted(T))]) for T in combinations(S, 3)]
        br, bs = min(rs, key=lambda x: x[0])
        ex.append(dict(S=S, syn4=round(v, 4), best_parent_rank=br,
                       best_parent_syn3=round(bs, 4)))
    ex = pd.DataFrame(ex).sort_values("best_parent_rank", ascending=False)
    print(ex.head(8).to_string(index=False))
    ex.to_csv(ROOT / "outputs/antihierarchy_examples.csv", index=False)


if __name__ == "__main__":
    main()
