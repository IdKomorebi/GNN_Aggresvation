#!/usr/bin/env python3
"""区分 any-conf 可达性与 same-conf 层级性。

主评估：
- 父集排名只由 search 分片确定；
- 强高阶集合及其真实最强目标 c* 只由 audit 分片确定；
- any-conf：父集可因任意目标得分高而进入 beam；
- aligned：父集必须在高阶集合的 c* 上得分高。
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N = 44
BEAMS = (200, 500, 1000, 2000, 5000)
TAUS = (0.05, 0.10, 0.15)


def descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def analyze_order(m: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ps = np.load(ROOT / "outputs" / f"syn_search_o{m-1}.npy", mmap_mode="r")
    pa = np.load(ROOT / "outputs" / f"syn_audit_o{m-1}.npy", mmap_mode="r")
    cs = np.load(ROOT / "outputs" / f"syn_search_o{m}.npy", mmap_mode="r")
    ca = np.load(ROOT / "outputs" / f"syn_audit_o{m}.npy", mmap_mode="r")
    nc = ps.shape[1]

    parent_sets = list(combinations(range(N), m - 1))
    parent_index = {key: i for i, key in enumerate(parent_sets)}

    ps_any = np.max(ps, axis=1)
    pa_any = np.max(pa, axis=1)
    rank_s_any = descending_ranks(ps_any)
    rank_a_any = descending_ranks(pa_any)
    rank_s_conf = np.column_stack(
        [descending_ranks(np.asarray(ps[:, c])) for c in range(nc)]
    )
    rank_a_conf = np.column_stack(
        [descending_ranks(np.asarray(pa[:, c])) for c in range(nc)]
    )

    rows = []
    for idx, child in enumerate(combinations(range(N), m)):
        child_s = np.asarray(cs[idx])
        child_a = np.asarray(ca[idx])
        if max(float(child_s.max()), float(child_a.max())) <= min(TAUS):
            continue

        c_s = int(np.argmax(child_s))
        c_a = int(np.argmax(child_a))
        pids = np.array(
            [parent_index[tuple(p)] for p in combinations(child, m - 1)],
            dtype=np.int32,
        )
        score_s = np.asarray(ps[pids])
        score_a = np.asarray(pa[pids])

        flat_s = int(np.argmax(score_s))
        best_parent_pos_s, best_parent_conf_s = np.unravel_index(flat_s, score_s.shape)
        flat_a = int(np.argmax(score_a))
        best_parent_pos_a, best_parent_conf_a = np.unravel_index(flat_a, score_a.shape)

        rows.append(
            {
                "indices": ",".join(map(str, child)),
                "child_search_syn": float(child_s[c_s]),
                "child_audit_syn": float(child_a[c_a]),
                "child_search_conf": c_s,
                "child_audit_conf": c_a,
                "child_conf_stable": c_s == c_a,
                "any_parent_score_search": float(score_s.max()),
                "aligned_parent_score_search_caudit": float(score_s[:, c_a].max()),
                "aligned_parent_score_search_csearch": float(score_s[:, c_s].max()),
                "any_parent_score_audit": float(score_a.max()),
                "aligned_parent_score_audit": float(score_a[:, c_a].max()),
                "any_rank_search": int(rank_s_any[pids].min()),
                "aligned_rank_search_caudit": int(rank_s_conf[pids, c_a].min()),
                "aligned_rank_search_csearch": int(rank_s_conf[pids, c_s].min()),
                "any_rank_audit": int(rank_a_any[pids].min()),
                "aligned_rank_audit": int(rank_a_conf[pids, c_a].min()),
                "best_parent_conf_search": int(best_parent_conf_s),
                "best_parent_conf_audit": int(best_parent_conf_a),
                "best_parent_conf_matches_child_audit": best_parent_conf_s == c_a,
                "best_parent_conf_matches_child_search": best_parent_conf_s == c_s,
                "best_parent_pos_search": int(best_parent_pos_s),
                "best_parent_pos_audit": int(best_parent_pos_a),
            }
        )

    details = pd.DataFrame(rows)
    summary = []
    for tau in TAUS:
        g = details[details.child_audit_syn > tau]
        if not len(g):
            continue
        base = {
            "order": m,
            "tau": tau,
            "n_strong_audit": len(g),
            "child_conf_stability": g.child_conf_stable.mean(),
            "best_parent_conf_match": g.best_parent_conf_matches_child_audit.mean(),
            "median_any_parent_search": g.any_parent_score_search.median(),
            "median_aligned_parent_search": g.aligned_parent_score_search_caudit.median(),
            "parent_above_tau_any": (g.any_parent_score_search > tau).mean(),
            "parent_above_tau_aligned": (
                g.aligned_parent_score_search_caudit > tau
            ).mean(),
        }
        for b in BEAMS:
            any_ok = g.any_rank_search <= b
            aligned_ok = g.aligned_rank_search_caudit <= b
            # 若为 12 个 confidential 各保留一条 aligned lane，
            # 为与 any-conf 总宽 B 近似同预算，每条 lane 只能保留约 B/12。
            aligned_budget = g.aligned_rank_search_caudit <= max(b // ps.shape[1], 1)
            base[f"any_reach_B{b}"] = any_ok.mean()
            base[f"aligned_reach_B{b}"] = aligned_ok.mean()
            base[f"aligned_budget_reach_B{b}"] = aligned_budget.mean()
            base[f"cross_conf_rescue_B{b}"] = (any_ok & ~aligned_ok).mean()
            base[f"blind_any_B{b}"] = (~any_ok).mean()
        summary.append(base)
    return details, pd.DataFrame(summary)


def main() -> None:
    all_summary = []
    for m in (4, 5):
        details, summary = analyze_order(m)
        details.to_csv(ROOT / "outputs" / f"hierarchy_details_o{m}.csv", index=False)
        all_summary.append(summary)
    out = pd.concat(all_summary, ignore_index=True)
    out.to_csv(ROOT / "outputs" / "hierarchy_summary.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    key = [
        "order",
        "tau",
        "n_strong_audit",
        "child_conf_stability",
        "best_parent_conf_match",
        "parent_above_tau_any",
        "parent_above_tau_aligned",
        "any_reach_B1000",
        "aligned_reach_B1000",
        "aligned_budget_reach_B1000",
        "cross_conf_rescue_B1000",
        "blind_any_B1000",
    ]
    print(out[key].round(4).to_string(index=False))
    (ROOT / "outputs" / "hierarchy_protocol.json").write_text(
        json.dumps(
            {
                "selection_split": "search",
                "evaluation_split": "audit",
                "parent_score_any": "max over parent and confidential",
                "parent_score_aligned": "max over parent at audit child argmax-conf",
                "beams": BEAMS,
                "thresholds": TAUS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
