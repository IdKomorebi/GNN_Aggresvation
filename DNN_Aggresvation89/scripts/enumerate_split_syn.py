#!/usr/bin/env python3
"""全枚举 search/audit 分片上的逐-conf 不可约增量。

输出按 itertools.combinations 的字典序对齐：
    outputs/syn_{mode}_o{m}.npy, shape=(C(44,m), 12)

这里保留逐-conf 值，避免旧实验过早 max_conf 后无法判断父子目标是否一致。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R88 = REPO / "DNN_Aggresvation88"
sys.path.insert(0, str(R88 / "src"))
from query_split import Poly2QuerySplit  # noqa: E402


def syn_per_conf(q: Poly2QuerySplit, key: tuple[int, ...]) -> np.ndarray:
    v = np.asarray(q.vhat(key), dtype=np.float64)
    parents = [tuple(x for x in key if x != drop) for drop in key]
    best = np.max([np.asarray(q.vhat(p), dtype=np.float64) for p in parents], axis=0)
    return v - best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["search", "audit"])
    ap.add_argument("--orders", default="3,4,5")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    (ROOT / "outputs").mkdir(exist_ok=True)
    q = Poly2QuerySplit(mode=args.mode)
    n, nc = q.n_general, len(q.conf_names)
    metadata = {"mode": args.mode, "n_general": n, "conf_names": q.conf_names, "orders": {}}

    for m in [int(x) for x in args.orders.split(",")]:
        out_path = ROOT / "outputs" / f"syn_{args.mode}_o{m}.npy"
        if out_path.exists() and not args.force:
            arr = np.load(out_path, mmap_mode="r")
            if arr.shape == (comb(n, m), nc):
                print(f"[{args.mode}/o{m}] 已存在，跳过: {arr.shape}", flush=True)
                metadata["orders"][str(m)] = {"count": len(arr), "reused": True}
                continue

        total = comb(n, m)
        arr = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(total, nc)
        )
        t0 = time.perf_counter()
        for i, key in enumerate(combinations(range(n), m)):
            arr[i] = syn_per_conf(q, key)
            if (i + 1) % 20000 == 0 or i + 1 == total:
                rate = (i + 1) / max(time.perf_counter() - t0, 1e-9)
                print(
                    f"[{args.mode}/o{m}] {i+1:,}/{total:,} "
                    f"({rate:,.0f} sets/s)",
                    flush=True,
                )
        arr.flush()
        seconds = time.perf_counter() - t0
        metadata["orders"][str(m)] = {
            "count": total,
            "seconds": seconds,
            "solves": q.n_solve,
            "reused": False,
        }
        print(f"[{args.mode}/o{m}] 完成，用时 {seconds:.1f}s", flush=True)
        q.vhat.cache_clear()

    (ROOT / "outputs" / f"enum_meta_{args.mode}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
