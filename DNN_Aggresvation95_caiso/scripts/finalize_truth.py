#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""95_caiso：聚合重训真值分片 → 评估用真值文件。

输入：outputs/truth_o2_sets_s*of*.csv（990 集合 × 12 conf 的 v_audit/v_full）
     outputs/truth_o3_sets_s*of*.csv（随机三元组，可选）
输出：
  outputs/synergy2_perconf.csv   列 i,j,conf,synergy,synergy_full,v_pair,v_i,v_j
                                 （synergy = audit 口径，供 eval_l1_order2）
  outputs/h3_unbiased_pool.csv   列 ix,conf,syn3_true,syn3_true_full,group
                                 （group=triple_rand，供 eval_l1_order3）
"""
from __future__ import annotations

import ast
import glob
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_sets(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / f"outputs/{pattern}")))
    assert fs, pattern
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["St"] = d["S"].map(lambda s: tuple(ast.literal_eval(s)))
    d = d.drop_duplicates(subset=["St", "conf"], keep="first")
    return d


def main() -> None:
    d2 = load_sets("truth_o2_sets_s*of*.csv")
    # (set, conf) → (v_audit, v_full)
    va = {(r.St, r.conf): r.v_audit for r in d2.itertuples()}
    vf = {(r.St, r.conf): r.v_full for r in d2.itertuples()}
    confs = sorted(d2["conf"].unique())
    singles = sorted({s for s in d2["St"] if len(s) == 1})
    pairs = sorted({s for s in d2["St"] if len(s) == 2})
    print(f"o2 真值：{len(singles)} 单 + {len(pairs)} 对 × {len(confs)} conf")

    rows = []
    for (i, j) in pairs:
        for c in confs:
            key_p, ki, kj = ((i, j), c), ((i,), c), ((j,), c)
            if key_p not in va or ki not in va or kj not in va:
                continue
            rows.append(dict(
                i=i, j=j, conf=c,
                synergy=va[key_p] - max(va[ki], va[kj]),
                synergy_full=vf[key_p] - max(vf[ki], vf[kj]),
                v_pair=va[key_p], v_i=va[ki], v_j=va[kj]))
    out2 = pd.DataFrame(rows)
    out2.to_csv(ROOT / "outputs/synergy2_perconf.csv", index=False)
    print(f"synergy2_perconf.csv: {len(out2)} 行，"
          f"max_syn={out2.synergy.max():.4f}，n>0.2={(out2.synergy > 0.2).sum()}，"
          f"n>0.1={(out2.synergy > 0.1).sum()}")

    fs3 = glob.glob(str(ROOT / "outputs/truth_o3_sets_s*of*.csv"))
    if fs3:
        d3 = load_sets("truth_o3_sets_s*of*.csv")
        va3 = {(r.St, r.conf): r.v_audit for r in d3.itertuples()}
        vf3 = {(r.St, r.conf): r.v_full for r in d3.itertuples()}
        triples = sorted({s for s in d3["St"] if len(s) == 3})
        rows = []
        n_miss = 0
        for S in triples:
            for c in confs:
                if (S, c) not in va3:
                    continue
                sub_a = [va.get((tuple(sorted(t)), c)) for t in combinations(S, 2)]
                sub_f = [vf.get((tuple(sorted(t)), c)) for t in combinations(S, 2)]
                if any(v is None for v in sub_a):
                    n_miss += 1
                    continue
                rows.append(dict(
                    ix=str(S), conf=c,
                    syn3_true=va3[(S, c)] - max(sub_a),
                    syn3_true_full=vf3[(S, c)] - max(sub_f),
                    group="triple_rand"))
        out3 = pd.DataFrame(rows)
        out3.to_csv(ROOT / "outputs/h3_unbiased_pool.csv", index=False)
        print(f"h3_unbiased_pool.csv: {len(triples)} 三元组 → {len(rows)} 行"
              f"（缺子集 {n_miss}），max_syn3={out3.syn3_true.max():.4f}，"
              f"n>0.1={(out3.syn3_true > 0.1).sum()}")


if __name__ == "__main__":
    main()
