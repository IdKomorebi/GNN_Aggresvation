# -*- coding: utf-8 -*-
"""检查82号关键产物的完整性、行数、选择隔离和数值范围。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KGRID = {0, 1, 5, 10, 25, 50}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    schemes = cfg["experiment"]["schemes"]
    for scheme in schemes:
        checkpoint = OUT / f"oracle_{scheme}_seed0.pt"
        k0 = OUT / f"k0_{scheme}_seed0.parquet"
        assert checkpoint.exists(), checkpoint
        assert k0.exists(), k0
        frame = pd.read_parquet(k0)
        assert len(frame) == 14234 * 12, (k0, len(frame))
        assert set(frame["size"]) == {1, 2, 3}
        assert frame.est.between(0, 1).all()

    selection = json.loads(
        (OUT / "k0_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selection_uses_test"] is False
    winners = {
        selection["best_parent_scheme"],
        selection["best_s1_scheme"],
    }
    for scheme in winners:
        assert (OUT / f"oracle_{scheme}_seed1.pt").exists()
        assert (OUT / f"k0_{scheme}_seed1.parquet").exists()
        low = list(
            OUT.glob(f"kgrid_low_{scheme}_seed0_shard*.csv.gz")
        )
        triples = list(
            OUT.glob(f"kgrid_triple_{scheme}_seed0_shard*.csv.gz")
        )
        assert len(low) == 1, (scheme, low)
        assert len(triples) == 2, (scheme, triples)
        low_frame = pd.concat([pd.read_csv(path) for path in low])
        triple_frame = pd.concat([pd.read_csv(path) for path in triples])
        assert len(low_frame) == 990 * 6 * 12
        assert len(triple_frame) == 13244 * 6 * 12
        assert set(low_frame.K) == KGRID
        assert set(triple_frame.K) == KGRID

    metrics = pd.read_csv(OUT / "kgrid_comparison.csv")
    assert not metrics.isna().any().any()
    assert set(metrics.K) == KGRID
    assert {"uniform", *winners}.issubset(set(metrics.scheme))
    for path in (
        ROOT / "figures/k0_strategy_tradeoff.png",
        ROOT / "figures/kgrid_oracle_comparison.png",
    ):
        assert path.exists() and path.stat().st_size > 10_000, path
    result = {
        "status": "PASS",
        "n_seed0_new_schemes": len(schemes),
        "confirmed_schemes": sorted(winners),
        "kgrid_rows": len(metrics),
    }
    (OUT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
