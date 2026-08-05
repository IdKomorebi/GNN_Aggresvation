#!/usr/bin/env python3
"""Collect comparable low-order, MLP-selected, and legacy GNN-selected sets."""
from __future__ import annotations

import json

import pandas as pd

from common import OUT, R72, low_order_problem


def main():
    problem = low_order_problem()
    fields = problem["fields"]
    orders = problem["orders"]
    candidates: dict[str, dict] = {}

    def add(label, protected, source):
        protected = list(protected)
        if len(set(protected)) != len(protected):
            raise ValueError(f"duplicate field in {label}")
        if not set(protected).issubset(fields):
            raise ValueError(f"unknown field in {label}")
        candidates[label] = {
            "protected": protected,
            "shared": [f for f in fields if f not in set(protected)],
            "k": len(protected),
            "source": source,
        }

    add("low_order_k24", orders["low_order"][:24], "low_order")
    add("degree_k24", orders["degree"][:24], "baseline")
    add("single_k24", orders["single_leak"][:24], "baseline")
    add("random_k24", orders["random"][:24], "baseline")

    curve = pd.read_csv(OUT / "mlp_greedy_curve.csv")
    for strategy, prefix in [("mlp_hybrid", "mlp_hybrid"), ("mlp_direct", "mlp_direct")]:
        # hybrid k=24 is exactly the already-labeled low_order_k24 set.
        ks = list(range(25, 31)) if strategy == "mlp_hybrid" else [24, 27, 30]
        for k in ks:
            row = curve[(curve.strategy == strategy) & (curve.k == k)]
            if len(row) != 1:
                raise ValueError(f"expected one row for {strategy}/k{k}, got {len(row)}")
            add(f"{prefix}_k{k}", json.loads(row.iloc[0].protected), "mlp_search")

    gnn_hybrid = json.loads((R72 / "outputs/protection_hybrid.json").read_text())
    gnn_direct = json.loads((R72 / "outputs/protection_direct.json").read_text())
    add(f"gnn_hybrid_k{len(gnn_hybrid)}", gnn_hybrid, "gnn_search")
    for k in [24, 27, 30]:
        add(f"gnn_direct_k{k}", gnn_direct[:k], "gnn_search")

    # Preserve labels even when two methods happen to select the same set; also record aliases.
    canonical = {}
    for label, meta in candidates.items():
        key = tuple(sorted(meta["protected"]))
        canonical.setdefault(key, []).append(label)
    aliases = [labels for labels in canonical.values() if len(labels) > 1]
    payload = {"candidates": candidates, "equivalent_set_aliases": aliases}
    (OUT / "candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(candidates)} labeled candidates; equivalent groups={aliases}")


if __name__ == "__main__":
    main()
