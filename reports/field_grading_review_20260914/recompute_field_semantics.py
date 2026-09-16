"""CPU-only descriptive audit of existing D69 records; no new attack training.

The k=1 envelope is constructed and checked on the same singles/pairs.
Its pair coverage is an algebraic check, NOT out-of-sample validation.
"""
from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
subs = json.loads((ROOT / "DNN_Aggresvation69/outputs/subsets.json").read_text())
with (ROOT / "DNN_Aggresvation69/outputs/truth_long.csv").open() as f:
    rows = [r for r in csv.DictReader(f) if r["is_duplicate_entry"] == "False"]
fields = sorted({f for s in subs.values() for f in s["fields"]})
index = {f: i for i, f in enumerate(fields)}
groups = defaultdict(list)
for r in rows:
    ids = tuple(sorted(index[f] for f in subs[r["sid"]]["fields"]))
    groups[r["conf"]].append((ids, r["group"], float(r["truth"])))

def g(v):
    return -0.5 * np.log1p(-np.clip(v, 0, 0.999))

output, detail, union_pairs = {}, [], set()
for conf, data in sorted(groups.items()):
    single = {ids[0]: v for ids, group, v in data if group == "single"}
    pairs = [(ids, v) for ids, group, v in data if group == "pair"]
    sv = np.array([single[i] for i in range(len(fields))])
    raw = np.maximum(sv, 0)
    logscore = g(sv)
    partners = [None] * len(fields)
    raw_partners = [None] * len(fields)
    violations, worst_violation = 0, 0.0
    fit = [(ids, v) for ids, group, v in data if group in ("single", "pair", "triple_rand")]
    X = np.zeros((len(fit), len(fields)))
    for k, (ids, _) in enumerate(fit):
        X[k, list(ids)] = 1
    w, _ = nnls(X, g([v for _, v in fit]))
    n_positive = 0
    for (i, j), v in pairs:
        violation = max(sv[i], sv[j]) - v
        violations += int(violation > 1e-8)
        worst_violation = max(worst_violation, violation)
        if g(v) - w[i] - w[j] > 0.05:
            union_pairs.add((i, j))
            n_positive += 1
        for a, b in ((i, j), (j, i)):
            if v - sv[b] > raw[a]:
                raw[a], raw_partners[a] = v - sv[b], b
            d = g(v) - g(sv[b])
            if d > logscore[a]:
                logscore[a], partners[a] = d, b
    normalized = -np.expm1(-2 * logscore)
    for i, field in enumerate(fields):
        detail.append(dict(target=conf, field=field, singleton_r2=sv[i],
                           nnls_weight=w[i], max_raw_increment_k1=raw[i],
                           max_log_increment_k1=logscore[i],
                           conditional_residual_reduction_k1=normalized[i],
                           raw_witness=fields[raw_partners[i]] if raw_partners[i] is not None else "<empty>",
                           log_witness=fields[partners[i]] if partners[i] is not None else "<empty>"))
    output[conf] = dict(pair_count=len(pairs), nonmonotone_pair_count=violations,
                        largest_pair_drop=worst_violation, positive_nnls_pair_residuals=n_positive,
                        raw_increment_upgraded_4bins=int(np.sum(np.digitize(raw, [.05,.2,.5]) > np.digitize(sv, [.05,.2,.5]))),
                        normalized_high_ge_05=int(np.sum(normalized >= .5)))
    for name, selected in (("pair_construction_check", [(ids, v) for ids, v in pairs]),
                           ("top3_outside_k1_guarantee", [(ids, v) for ids, group, v in data if group == "triple_top"]),
                           ("wide_outside_k1_guarantee", [(ids, v) for ids, group, v in data if group == "wide_random" and len(ids) >= 3])):
        y = np.array([v for _, v in selected])
        p = np.array([-np.expm1(-2 * logscore[list(ids)].sum()) for ids, _ in selected])
        output[conf][name] = dict(n=len(y), mae=float(np.mean(abs(y-p))),
                                 underestimates=int(np.sum(p < y-1e-8)),
                                 false_release=int(np.sum((p <= .7) & (y > .7))),
                                 false_reject=int(np.sum((p > .7) & (y <= .7))))

with (OUT / "field_semantics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(detail[0]))
    writer.writeheader()
    writer.writerows(detail)
result = dict(note=__doc__, fields=len(fields), targets=len(groups),
              distinct_positive_residual_pairs=len(union_pairs), per_target=output,
              wind=[r for r in detail if r["target"] == "total_gen" and r["field"] in ("gen_fuel_wind_mw", "gen_fuel_wind_pct")])
(OUT / "field_semantics_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps({k:v for k,v in result.items() if k != "per_target"}, indent=2, ensure_ascii=False))
for name in ("pair_construction_check", "top3_outside_k1_guarantee", "wide_outside_k1_guarantee"):
    stats = [v[name] for v in output.values()]
    n = sum(v["n"] for v in stats)
    print(name, dict(n=n, mae=sum(v["mae"]*v["n"] for v in stats)/n,
                     false_release=sum(v["false_release"] for v in stats),
                     false_reject=sum(v["false_reject"] for v in stats),
                     underestimates=sum(v["underestimates"] for v in stats)))
print("nonmonotone pairs", sum(v["nonmonotone_pair_count"] for v in output.values()))
print("raw score upgraded", sum(v["raw_increment_upgraded_4bins"] for v in output.values()))
