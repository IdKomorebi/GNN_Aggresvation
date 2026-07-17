#!/usr/bin/env python3
"""Build the DNN69 evaluation registry from the three existing truth families.

The registry intentionally keeps three different populations separate:
  * D60 random subsets: broad size coverage, DNN and GCN truth already available.
  * D67 singles/pairs: exhaustive order-1/2 coverage, DNN truth available.
  * D68 certified triples: selected top/random order-3 coverage, DNN truth available.

DNN69 adds GCN retraining for D67/D68 and never silently mixes the populations.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sources = {
        "d60_random": REPO / "DNN_Aggresvation60/outputs/subsets.json",
        "d67_single_pair": REPO / "DNN_Aggresvation67/outputs/subsets.json",
        "d68_triple": REPO / "DNN_Aggresvation68/outputs/subsets.json",
    }
    registry: dict[str, dict] = {}
    source_counts: dict[str, int] = {}

    for source, path in sources.items():
        raw = json.loads(path.read_text())
        if source == "d60_random":
            raw = {sid: meta for sid, meta in raw.items()
                   if meta.get("group") == "random_eval"}
        source_counts[source] = len(raw)
        for sid, meta in raw.items():
            if sid in registry:
                raise ValueError(f"duplicate subset id across sources: {sid}")
            fields = list(meta["fields"])
            if len(fields) != int(meta["size"]):
                raise ValueError(f"size mismatch for {sid}")
            group = {
                "d60_random": "wide_random",
                "d67_single_pair": meta["group"],
                "d68_triple": meta["group"],
            }[source]
            registry[sid] = {
                "source": source,
                "group": group,
                "size": len(fields),
                "fields": fields,
            }

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "subsets.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    )

    # D60 random sampling can land on a singleton/pair that D67 later exhaustively
    # enumerated. Keep both entries as independent retrain replicates, but mark a
    # canonical entry so no report can call them distinct field sets.
    by_field_set: dict[tuple[str, ...], list[str]] = {}
    for sid, meta in registry.items():
        by_field_set.setdefault(tuple(sorted(meta["fields"])), []).append(sid)
    duplicate_groups = []
    priority = {"d67_single_pair": 0, "d68_triple": 0, "d60_random": 1}
    for key, sids in by_field_set.items():
        canonical = sorted(sids, key=lambda s: (priority[registry[s]["source"]], s))[0]
        for sid in sids:
            registry[sid]["canonical_sid"] = canonical
            registry[sid]["is_duplicate_entry"] = sid != canonical
        if len(sids) > 1:
            duplicate_groups.append({"fields": list(key), "sids": sorted(sids),
                                     "canonical_sid": canonical})
    # Rewrite after adding canonical metadata.
    (out / "subsets.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    )

    manifest = {
        "n_subsets": len(registry),
        "n_unique_field_sets": len(by_field_set),
        "duplicate_entry_count": len(registry) - len(by_field_set),
        "duplicate_groups": duplicate_groups,
        "source_counts": source_counts,
        "group_counts": dict(Counter(v["group"] for v in registry.values())),
        "size_counts": dict(Counter(str(v["size"]) for v in registry.values())),
        "source_files": {str(p.relative_to(REPO)): sha256(p) for p in sources.values()},
        "base_yaml_sha256": sha256(ROOT / "base.yaml"),
        "metric_tensor_sha256": sha256(out / "relationship_cache/metric_tensor.npy"),
        "oracle_checkpoints": {
            p.name: sha256(p) for p in sorted(out.glob("oracle_*_seed[0-2].pt"))
        },
        "protocol": {
            "split_seed": 42,
            "retrain_seed": 0,
            "oracle_finetune_seed": 0,
            "k_grid": [0, 10, 50, 200],
            "truth_full_population": "dedicated DNN retrain",
            "truth_structure_audit": "per-confidential max(DNN, GCN) on D60 plus stratified D67/D68 audit",
        },
    }
    (out / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
