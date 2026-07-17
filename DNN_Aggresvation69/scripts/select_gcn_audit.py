#!/usr/bin/env python3
"""Select a deterministic, stratified GCN retraining audit set.

Full DNN truth and universal-oracle estimates remain exhaustive. Dedicated GCN
training is minutes per subset, so D69 uses all 50 existing D60 broad-size GCN
truth points plus 9 pairs and 9 triples spanning strong, ordinary, and random
synergy regimes. Selection uses DNN truth only and never the D69 GNN oracle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RNG = np.random.RandomState(69)


def load_dnn(folder: str, sid: str) -> dict:
    path = REPO / f"{folder}/outputs/retrain/{sid}_dnn_seed0.json"
    return json.loads(path.read_text())["per_conf_r2"]


def main() -> None:
    registry = json.loads((ROOT / "outputs/subsets.json").read_text())
    singles = {sid: load_dnn("DNN_Aggresvation67", sid)
               for sid, m in registry.items() if m["group"] == "single"}
    pairs = {sid: load_dnn("DNN_Aggresvation67", sid)
             for sid, m in registry.items() if m["group"] == "pair"}
    triples = {sid: load_dnn("DNN_Aggresvation68", sid)
               for sid, m in registry.items() if m["group"] in ["triple_top", "triple_rand"]}
    confs = sorted(next(iter(singles.values())))

    pair_score = {}
    for sid, value in pairs.items():
        i, j = [int(x) for x in sid[1:].split("_")]
        si, sj = f"s{i:02d}", f"s{j:02d}"
        syn = [value[c] - max(singles[si][c], singles[sj][c]) for c in confs]
        pair_score[sid] = max(syn)

    pair_order = sorted(pair_score, key=pair_score.get, reverse=True)
    pair_top = pair_order[:3]
    remaining = [s for s in pair_order[3:] if s != "p00_01"]
    near_zero = sorted(remaining, key=lambda s: abs(pair_score[s]))[:3]
    random_pool = [s for s in remaining if s not in near_zero]
    pair_random = list(RNG.choice(random_pool, size=2, replace=False)) + ["p00_01"]

    triple_score = {}
    for sid, value in triples.items():
        ids = [int(x) for x in sid[1:].split("_")]
        lowers = [f"p{a:02d}_{b:02d}" for pos, a in enumerate(ids) for b in ids[pos + 1:]]
        syn = [value[c] - max(pairs[p][c] for p in lowers) for c in confs]
        triple_score[sid] = max(syn)
    top_population = [s for s in triples if registry[s]["group"] == "triple_top"]
    rand_population = [s for s in triples if registry[s]["group"] == "triple_rand"]
    triple_top = sorted(top_population, key=triple_score.get, reverse=True)[:3]
    top_remaining = [s for s in top_population if s not in triple_top]
    triple_top_random = list(RNG.choice(top_remaining, size=3, replace=False))
    triple_random = list(RNG.choice(rand_population, size=3, replace=False))

    selected: dict[str, dict] = {}
    for reason, sids in [
        ("pair_strong_dnn_synergy", pair_top),
        ("pair_near_zero_dnn_synergy", near_zero),
        ("pair_random", pair_random),
        ("triple_strong_dnn_synergy", triple_top),
        ("triple_top_stratum_random", triple_top_random),
        ("triple_random_stratum", triple_random),
    ]:
        for sid in sids:
            selected[sid] = {
                **registry[sid],
                "selection_reason": reason,
                "dnn_max_synergy": float(pair_score.get(sid, triple_score.get(sid))),
            }

    if len(selected) != 18:
        raise RuntimeError(f"expected 18 unique audit subsets, got {len(selected)}")
    path = ROOT / "outputs/gcn_audit_subsets.json"
    path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    counts = {}
    for meta in selected.values():
        counts[meta["selection_reason"]] = counts.get(meta["selection_reason"], 0) + 1
    print(json.dumps({"n": len(selected), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
