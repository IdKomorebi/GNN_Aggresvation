#!/usr/bin/env python3
"""量化高阶代理协同在 search/audit 半集之间的稳定性。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TAUS = (0.05, 0.10, 0.15)


def main() -> None:
    rows = []
    for m in (3, 4, 5):
        s = np.load(ROOT / "outputs" / f"syn_search_o{m}.npy", mmap_mode="r")
        a = np.load(ROOT / "outputs" / f"syn_audit_o{m}.npy", mmap_mode="r")
        sm, am = np.max(s, axis=1), np.max(a, axis=1)
        sc, ac = np.argmax(s, axis=1), np.argmax(a, axis=1)
        rho = float(spearmanr(sm, am).statistic)
        mae = float(np.mean(np.abs(sm - am)))
        conf_match = float(np.mean(sc == ac))
        for tau in TAUS:
            ss, aa = sm > tau, am > tau
            inter = int(np.sum(ss & aa))
            rows.append(
                {
                    "order": m,
                    "tau": tau,
                    "spearman": rho,
                    "mae": mae,
                    "conf_match": conf_match,
                    "n_search_strong": int(ss.sum()),
                    "n_audit_strong": int(aa.sum()),
                    "intersection": inter,
                    "jaccard": inter / max(int(np.sum(ss | aa)), 1),
                    "search_precision_on_audit": inter / max(int(ss.sum()), 1),
                    "audit_recall_from_search_threshold": inter / max(int(aa.sum()), 1),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/split_stability.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
