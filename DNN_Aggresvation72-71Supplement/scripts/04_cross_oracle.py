#!/usr/bin/env python3
"""Evaluate every selected protection set with both oracle families and 3 seeds."""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from common import OUT, OracleEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "gnn"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    payload = json.loads((OUT / "candidates.json").read_text())
    evaluator = OracleEvaluator(args.arch, args.seed)
    rows = []
    t0 = time.time()
    for label, meta in payload["candidates"].items():
        r2 = evaluator.eval_fields(meta["shared"], 200)
        row = {
            "candidate": label, "source": meta["source"], "k": meta["k"],
            "arch": args.arch, "seed": args.seed,
            "worst": float(r2.max()), "mean": float(r2.mean()),
            "per_conf": json.dumps({n: float(v) for n, v in zip(evaluator.conf_names, r2)}),
        }
        rows.append(row)
        print(f"{args.arch}/seed{args.seed} {label}: worst={r2.max():.4f} mean={r2.mean():.4f} [{time.time()-t0:.1f}s]", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"cross_oracle_{args.arch}_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()

