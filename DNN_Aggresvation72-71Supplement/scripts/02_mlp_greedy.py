#!/usr/bin/env python3
"""Re-run D72 direct and hybrid oracle greedy search with the MLP oracle."""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from common import OUT, OracleEvaluator, TAU, low_order_problem, write_protocol_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kcap", type=int, default=30)
    args = ap.parse_args()
    manifest = write_protocol_manifest()
    problem = low_order_problem()
    fields = problem["fields"]
    evaluator = OracleEvaluator("mlp", args.seed)
    rows = []

    def run(initial, strategy):
        protected = list(initial)
        t0 = time.time()
        while True:
            shared = [f for f in fields if f not in set(protected)]
            cert = evaluator.eval_fields(shared, manifest["k_cert"])
            rows.append({
                "strategy": strategy, "seed": args.seed, "k": len(protected),
                "chosen": None if len(protected) == len(initial) else protected[-1],
                "worst_k200": float(cert.max()), "mean_k200": float(cert.mean()),
                "oracle_threshold_met": bool(cert.max() <= TAU),
                "protected": json.dumps(protected), "elapsed": time.time() - t0,
            })
            pd.DataFrame(rows).to_csv(OUT / "mlp_greedy_curve.csv", index=False)
            print(f"[{strategy}] k={len(protected)} worst={cert.max():.4f} mean={cert.mean():.4f}", flush=True)
            if len(protected) >= args.kcap:
                return protected
            candidates = [f for f in fields if f not in set(protected)]
            best_field, best_worst = None, float("inf")
            for f in candidates:
                shared_candidate = [x for x in candidates if x != f]
                score = float(evaluator.eval_fields(shared_candidate, manifest["k_inner"]).max())
                if score < best_worst:
                    best_field, best_worst = f, score
            protected.append(best_field)
            print(f"[{strategy}] choose={best_field} inner_worst={best_worst:.4f}", flush=True)

    low = problem["orders"]["low_order"][:problem["k_cover"]]
    hybrid = run(low, "mlp_hybrid")
    direct = run([], "mlp_direct")
    (OUT / "protection_low_order.json").write_text(json.dumps(low, ensure_ascii=False, indent=2) + "\n")
    (OUT / "protection_mlp_hybrid.json").write_text(json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n")
    (OUT / "protection_mlp_direct.json").write_text(json.dumps(direct, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

