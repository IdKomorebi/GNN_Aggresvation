#!/usr/bin/env python3
"""拆开 v(child)、max v(parent) 与两者差值的分片误差。

目的不是再次定义“真值”，而是确认高阶不稳定主要来自基础 R2，还是来自
两个相近量相减并对 confidential 取最大。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation88/src"))
from query_split import Poly2QuerySplit  # noqa: E402


def metrics(frame: pd.DataFrame, group: str, order: int) -> dict:
    def corr(a: str, b: str) -> float:
        return frame[a].corr(frame[b], method="spearman")

    return {
        "order": order,
        "group": group,
        "n": len(frame),
        "child_v_spearman": corr("child_search", "child_audit"),
        "parent_v_spearman": corr("parent_search", "parent_audit"),
        "syn_spearman": corr("syn_search_aligned", "syn_audit_aligned"),
        "child_v_mae": np.mean(
            np.abs(frame.child_search - frame.child_audit)
        ),
        "parent_v_mae": np.mean(
            np.abs(frame.parent_search - frame.parent_audit)
        ),
        "syn_mae": np.mean(
            np.abs(frame.syn_search_aligned - frame.syn_audit_aligned)
        ),
    }


def main() -> None:
    qs, qa = Poly2QuerySplit("search"), Poly2QuerySplit("audit")
    rng = np.random.RandomState(20260725)
    summary = []
    detail_rows = []
    for order in (4, 5):
        audit_syn = np.load(ROOT / "outputs" / f"syn_audit_o{order}.npy", mmap_mode="r")
        audit_max = np.max(audit_syn, axis=1)
        strong = np.flatnonzero(audit_max > 0.10)
        strong_set = set(strong.tolist())
        remaining = np.setdiff1d(
            np.arange(len(audit_syn), dtype=np.int32), strong, assume_unique=True
        )
        random_ix = rng.choice(remaining, size=1000, replace=False)
        wanted = strong_set | set(random_ix.tolist())
        keys = {}
        for pos, key in enumerate(combinations(range(44), order)):
            if pos in wanted:
                keys[pos] = key
            if len(keys) == len(wanted):
                break

        rows = []
        for pos in sorted(wanted):
            key = keys[pos]
            c = int(np.argmax(audit_syn[pos]))
            vs, va = np.asarray(qs.vhat(key)), np.asarray(qa.vhat(key))
            parents = [tuple(x for x in key if x != drop) for drop in key]
            ps = np.stack([np.asarray(qs.vhat(p)) for p in parents])
            pa = np.stack([np.asarray(qa.vhat(p)) for p in parents])
            row = {
                "order": order,
                "index": pos,
                "indices": ",".join(map(str, key)),
                "group": "audit_strong" if pos in strong_set else "random",
                "audit_conf": c,
                "child_search": vs[c],
                "child_audit": va[c],
                "parent_search": ps[:, c].max(),
                "parent_audit": pa[:, c].max(),
            }
            row["syn_search_aligned"] = row["child_search"] - row["parent_search"]
            row["syn_audit_aligned"] = row["child_audit"] - row["parent_audit"]
            rows.append(row)
        frame = pd.DataFrame(rows)
        detail_rows.append(frame)
        for group, g in frame.groupby("group"):
            summary.append(metrics(g, group, order))

    details = pd.concat(detail_rows, ignore_index=True)
    out = pd.DataFrame(summary)
    details.to_csv(ROOT / "outputs/proxy_error_details.csv", index=False)
    out.to_csv(ROOT / "outputs/proxy_error_decomposition.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
