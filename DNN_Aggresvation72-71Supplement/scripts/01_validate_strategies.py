#!/usr/bin/env python3
"""Re-run D71 protection curves with the more faithful MLP oracle."""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from common import OUT, OracleEvaluator, low_order_problem, write_protocol_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    manifest = write_protocol_manifest()
    problem = low_order_problem()
    evaluator = OracleEvaluator("mlp", args.seed)
    budgets = [0, 4, 8, 12, 16, 20, 24]
    rows = []
    t0 = time.time()
    for strategy, order in problem["orders"].items():
        for k in budgets:
            protected = order[:k]
            shared = [f for f in problem["fields"] if f not in set(protected)]
            r2 = evaluator.eval_fields(shared, manifest["k_cert"])
            rows.append({
                "arch": "mlp", "seed": args.seed, "strategy": strategy,
                "k": k, "n_shared": len(shared), "worst": float(r2.max()),
                "mean": float(r2.mean()), "protected": json.dumps(protected),
            })
            print(f"seed={args.seed} {strategy} k={k}: worst={r2.max():.4f} mean={r2.mean():.4f} [{time.time()-t0:.1f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"validation_mlp_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()

