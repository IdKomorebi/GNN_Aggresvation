#!/usr/bin/env python3
"""检查 D89 的重训产物是否完整且口径一致。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RET = OUT / "retrain"


def main() -> None:
    subsets = json.loads((OUT / "subsets.json").read_text(encoding="utf-8"))
    expected = set(subsets)
    report: dict[str, object] = {
        "expected_unique_subsets": len(expected),
        "expected_confidential_targets": 12,
        "seeds": {},
    }
    reference_names = None
    ok = True
    for seed in (0, 1):
        files = sorted(RET.glob(f"*_seed{seed}.json"))
        got = {p.name.removesuffix(f"_seed{seed}.json") for p in files}
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        bad = []
        for p in files:
            d = json.loads(p.read_text(encoding="utf-8"))
            names = tuple(d["per_conf_r2"])
            vals = np.asarray(list(d["per_conf_r2"].values()), dtype=float)
            if reference_names is None:
                reference_names = names
            if (
                names != reference_names
                or len(vals) != 12
                or not np.isfinite(vals).all()
                or d["evaluation_split"] != "audit"
            ):
                bad.append(p.name)
        seed_ok = not missing and not extra and not bad
        ok &= seed_ok
        report["seeds"][str(seed)] = {
            "files": len(files),
            "missing": missing,
            "extra": extra,
            "invalid": bad,
            "ok": seed_ok,
        }
    report["ok"] = ok
    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
