#!/usr/bin/env python3
"""Build a leakage-safe task split for subspace learning and synergy audit."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_data  # noqa: E402


def take(rng, values, number):
    values = list(values)
    rng.shuffle(values)
    return values[: min(number, len(values))], values[min(number, len(values)) :]


def main() -> None:
    data, _ = load_data(__import__("torch").device("cpu"))
    name_to_idx = {name: i for i, name in enumerate(data["general"])}

    raw = json.loads(
        (REPO / "DNN_Aggresvation69/outputs/subsets.json").read_text(encoding="utf-8")
    )
    tuple_to_meta = {}
    for sid, meta in raw.items():
        key = tuple(sorted(name_to_idx[name] for name in meta["fields"]))
        # Prefer the exhaustive low-order IDs over duplicated rs IDs.
        priority = 0 if sid.startswith(("s", "p", "t")) else 1
        old = tuple_to_meta.get(key)
        if old is None or priority < old["_priority"]:
            tuple_to_meta[key] = {
                "sid69": sid,
                "source": meta["source"],
                "group": meta["group"],
                "indices": list(key),
                "fields": [data["general"][i] for i in key],
                "size": len(key),
                "_priority": priority,
            }

    def tid(key):
        return "q_" + "_".join(f"{x:02d}" for x in key)

    task_by_key = {}
    for key, meta in tuple_to_meta.items():
        clean = {k: v for k, v in meta.items() if not k.startswith("_")}
        clean["task_id"] = tid(key)
        task_by_key[key] = clean

    singles = sorted(k for k in task_by_key if len(k) == 1)
    pairs = sorted(k for k in task_by_key if len(k) == 2)
    triple_top = sorted(
        k for k, v in task_by_key.items()
        if len(k) == 3 and v["group"] == "triple_top"
    )
    triple_rand = sorted(
        k for k, v in task_by_key.items()
        if len(k) == 3 and v["group"] == "triple_rand"
    )
    wide = sorted(k for k in task_by_key if len(k) >= 4)
    rng = np.random.RandomState(92026)

    # Each triple stratum contributes independently to all three splits.
    def split_triples(values):
        audit, rest = take(rng, values, 30)
        valid, rest = take(rng, rest, 20)
        train, _ = take(rng, rest, 80)
        return train, valid, audit

    tt_train, tt_val, tt_audit = split_triples(triple_top)
    tr_train, tr_val, tr_audit = split_triples(triple_rand)
    triple_train = tt_train + tr_train
    triple_val = tt_val + tr_val
    triple_audit = tt_audit + tr_audit

    def parents(key):
        return [tuple(x for x in key if x != removed) for removed in key]

    held_triple_parents = {
        parent for key in triple_val + triple_audit for parent in parents(key)
    }
    eligible_pairs = [key for key in pairs if key not in held_triple_parents]
    pair_audit, rest = take(rng, eligible_pairs, 50)
    pair_val, rest = take(rng, rest, 30)
    pair_train, _ = take(rng, rest, 200)

    wide_audit, rest = take(rng, wide, 10)
    wide_val, rest = take(rng, rest, 8)
    wide_train = rest

    basis_train = set(triple_train + pair_train + wide_train)
    val_targets = set(triple_val + pair_val + wide_val)
    audit_targets = set(triple_audit + pair_audit + wide_audit)

    def closure(targets):
        result = set(targets)
        for key in list(targets):
            if len(key) == 2:
                result.update((x,) for x in key)
            elif len(key) == 3:
                result.update(parents(key))
        # Pair parents of triples require their singles for syn2 diagnostics.
        for key in list(result):
            if len(key) == 2:
                result.update((x,) for x in key)
        return result

    val_all = closure(val_targets)
    audit_all = closure(audit_targets)
    assert not basis_train.intersection(val_targets | audit_targets)
    assert not basis_train.intersection(held_triple_parents)

    all_keys = sorted(basis_train | val_all | audit_all)
    tasks = []
    for key in all_keys:
        if key not in task_by_key:
            raise KeyError(f"Missing D69 truth/task metadata for {key}")
        task = dict(task_by_key[key])
        roles = []
        if key in basis_train:
            roles.append("basis_train")
        if key in val_targets:
            roles.append("validation_target")
        if key in val_all - val_targets:
            roles.append("validation_parent")
        if key in audit_targets:
            roles.append("audit_target")
        if key in audit_all - audit_targets:
            roles.append("audit_parent")
        task["roles"] = roles
        tasks.append(task)

    def ids(keys):
        return [tid(k) for k in sorted(keys)]

    design = {
        "seed": 92026,
        "oracle": "DNN_Aggresvation69/oracle_mlp_seed0",
        "teacher_steps": 25,
        "gradient_support_sizes": [256, 1024],
        "tasks": tasks,
        "splits": {
            "basis_train": ids(basis_train),
            "validation_targets": ids(val_targets),
            "validation_all": ids(val_all),
            "audit_targets": ids(audit_targets),
            "audit_all": ids(audit_all),
            "validation_triples": ids(triple_val),
            "audit_triples": ids(triple_audit),
            "validation_pairs": ids(pair_val),
            "audit_pairs": ids(pair_audit),
            "validation_wide": ids(wide_val),
            "audit_wide": ids(wide_audit),
        },
        "counts": dict(Counter(role for t in tasks for role in t["roles"])),
        "notes": [
            "validation/audit triple 的全部 pair 父集均未进入 basis_train",
            "single 不进入 basis_train，避免 pair 协同父值发生精确任务泄漏",
            "triple_top 与 triple_rand 各自分层切分",
            "validation/audit 可能共享 single 父节点；调参时会显式排除 audit 闭包中的任务",
        ],
    }
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "n_tasks": len(tasks),
        "counts": design["counts"],
        "basis_by_size": dict(Counter(len(k) for k in basis_train)),
        "validation_targets": len(val_targets),
        "audit_targets": len(audit_targets),
        "held_triple_parents": len(held_triple_parents),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
