#!/usr/bin/env python3
"""仅用 search 结果构造 DNN 验证样本，audit 不参与选择。

三组四元组：
- aligned：any-conf 与 same-conf 父集均能进入 top-B；
- cross_conf：any-conf 能进入，但高阶 search 目标上的父集进不了；
- blind：所有父集在 any-conf 排名下也进不了。

每个四元组认证自身、4 个三元父集和 6 个二元子集。
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = 1000
N_PER_GROUP = 8


def parse_key(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def spread_pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.sort_values("child_search_syn", ascending=False).reset_index(drop=True)
    if len(df) <= n:
        return df
    positions = [round(i * (len(df) - 1) / (n - 1)) for i in range(n)]
    return df.iloc[positions]


def main() -> None:
    d = pd.read_csv(ROOT / "outputs/hierarchy_details_o4.csv")
    d = d[d.child_search_syn > 0.10].copy()
    aligned = d[
        (d.any_rank_search <= B)
        & (d.aligned_rank_search_csearch <= B)
        & d.best_parent_conf_matches_child_search
    ]
    cross = d[
        (d.any_rank_search <= B) & (d.aligned_rank_search_csearch > B)
    ]
    if len(cross) < N_PER_GROUP:
        cross = d[
            (d.any_rank_search <= B) & ~d.best_parent_conf_matches_child_search
        ].sort_values(
            ["aligned_rank_search_csearch", "child_search_syn"],
            ascending=[False, False],
        )
    blind = d[d.any_rank_search > B]

    picked = []
    for name, group in (
        ("aligned", aligned),
        ("cross_conf", cross),
        ("blind", blind),
    ):
        chosen = spread_pick(group, N_PER_GROUP).copy()
        chosen["selection_group"] = name
        picked.append(chosen)
    picked = pd.concat(picked, ignore_index=True)
    picked.to_csv(ROOT / "outputs/dnn_selected_quads.csv", index=False)

    subsets: dict[str, dict] = {}
    quad_meta = {}

    def sid(key: tuple[int, ...]) -> str:
        return "s" + "_".join(f"{x:02d}" for x in key)

    for row in picked.itertuples():
        quad = parse_key(row.indices)
        for size in (2, 3, 4):
            for key in combinations(quad, size):
                k = tuple(sorted(key))
                subsets.setdefault(
                    sid(k),
                    {"indices": list(k), "size": size, "group": f"dnn_o{size}"},
                )
        quad_meta[sid(quad)] = {
            "indices": list(quad),
            "selection_group": row.selection_group,
            "proxy_child_search_syn": row.child_search_syn,
            "proxy_child_audit_syn": row.child_audit_syn,
            "proxy_any_rank_search": int(row.any_rank_search),
            "proxy_aligned_rank_search": int(row.aligned_rank_search_csearch),
            "proxy_child_search_conf": int(row.child_search_conf),
            "proxy_child_audit_conf": int(row.child_audit_conf),
        }

    (ROOT / "outputs/subsets.json").write_text(
        json.dumps(subsets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "outputs/dnn_quad_meta.json").write_text(
        json.dumps(quad_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"选择四元组 {len(picked)} 个；需要训练唯一子集 {len(subsets)} 个；"
        f"组分布={picked.selection_group.value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
