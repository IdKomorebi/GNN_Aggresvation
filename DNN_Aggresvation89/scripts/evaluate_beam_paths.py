#!/usr/bin/env python3
"""在固定 search beam 路径上区分同-conf 到达与跨-conf 借路。

候选/beam 全部由 search 的 max-conf 分数决定；audit 只负责定义强集合与其 c*。
对每个 audit 强集合：
- found_any：是否被原 beam 触及；
- found_aligned_path：是否至少存在一个进入上一层 beam 的父集，
  且该父集在 search 上的 argmax-conf 等于子集合 audit c*。
两者差值就是在同一查询预算下真正依靠跨-conf 父集“借路”的比例。
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N = 44
B = 1000
TAUS = (0.05, 0.10, 0.15)


def expand(beam: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
    out = set()
    for key in beam:
        members = set(key)
        for x in range(N):
            if x not in members:
                out.add(tuple(sorted(key + (x,))))
    return out


def main() -> None:
    s3 = np.load(ROOT / "outputs/syn_search_o3.npy", mmap_mode="r")
    s4 = np.load(ROOT / "outputs/syn_search_o4.npy", mmap_mode="r")
    a4 = np.load(ROOT / "outputs/syn_audit_o4.npy", mmap_mode="r")
    a5 = np.load(ROOT / "outputs/syn_audit_o5.npy", mmap_mode="r")

    sets3 = list(combinations(range(N), 3))
    sets4 = list(combinations(range(N), 4))
    ix4 = {key: i for i, key in enumerate(sets4)}

    score3 = np.max(s3, axis=1)
    beam3_idx = np.argsort(-score3, kind="stable")[:B]
    beam3 = {sets3[i] for i in beam3_idx}
    conf3 = {sets3[i]: int(np.argmax(s3[i])) for i in beam3_idx}

    touched4 = expand(beam3)
    touched4_idx = np.array([ix4[key] for key in touched4], dtype=np.int32)
    best_local = np.argsort(
        -np.max(s4[touched4_idx], axis=1), kind="stable"
    )[:B]
    beam4_idx = touched4_idx[best_local]
    beam4 = {sets4[i] for i in beam4_idx}
    conf4 = {sets4[i]: int(np.argmax(s4[i])) for i in beam4_idx}
    touched5 = expand(beam4)

    rows = []
    for m, audit, touched, parent_beam, parent_conf in (
        (4, a4, touched4, beam3, conf3),
        (5, a5, touched5, beam4, conf4),
    ):
        counts = {
            tau: {"strong": 0, "found": 0, "aligned": 0, "borrowed": 0}
            for tau in TAUS
        }
        for idx, child in enumerate(combinations(range(N), m)):
            syn = np.asarray(audit[idx])
            c = int(np.argmax(syn))
            value = float(syn[c])
            relevant = [tau for tau in TAUS if value > tau]
            if not relevant:
                continue
            found = child in touched
            aligned = False
            if found:
                for parent in combinations(child, m - 1):
                    p = tuple(parent)
                    if p in parent_beam and parent_conf[p] == c:
                        aligned = True
                        break
            for tau in relevant:
                counts[tau]["strong"] += 1
                counts[tau]["found"] += int(found)
                counts[tau]["aligned"] += int(aligned)
                counts[tau]["borrowed"] += int(found and not aligned)

        for tau, c in counts.items():
            n = c["strong"]
            rows.append(
                {
                    "order": m,
                    "tau": tau,
                    "n_strong_audit": n,
                    "touched": len(touched),
                    "found": c["found"],
                    "recall": c["found"] / max(n, 1),
                    "aligned_path_found": c["aligned"],
                    "aligned_path_recall": c["aligned"] / max(n, 1),
                    "borrowed_path_found": c["borrowed"],
                    "borrowed_fraction_of_found": c["borrowed"]
                    / max(c["found"], 1),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/beam_path_alignment.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(
        f"\nsearch beam: beam3={len(beam3)}, touched4={len(touched4)}, "
        f"beam4={len(beam4)}, touched5={len(touched5)}"
    )


if __name__ == "__main__":
    main()
