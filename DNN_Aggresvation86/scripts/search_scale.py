# -*- coding: utf-8 -*-
"""86 号：把 beam 搜索扩到 order-5/6，证明"beam 覆盖率保持、触及数不组合爆炸"。

order-4 全枚举只 13.6万(47s)，省 4× 不惊艳；价值在高阶枚举不可行时。
本脚本：order-5 全枚举(1.09M,~10min)作 GT 量 beam 覆盖；order-6 只报 beam 触及成本
(全枚举 705万不跑)，与组合数对比展示"beam 触及≈线性于宽×阶,而枚举指数爆炸"。
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from query import Poly2Query  # noqa: E402
from search_highorder import beam_search  # noqa: E402
from runlog import log  # noqa: E402

TAUS = (0.10, 0.15)


def full_enum(q: Poly2Query, n: int, m: int) -> pd.DataFrame:
    rows = []
    t = time.perf_counter()
    for i, s in enumerate(combinations(range(n), m)):
        rows.append((s, q.syn(s)[0]))
        if (i + 1) % 100000 == 0:
            log("ENUM", "PROG", note=f"order{m} {i+1} {(i+1)/(time.perf_counter()-t):.0f}/s")
    log("ENUM", "DONE", note=f"order{m} {len(rows)} in {time.perf_counter()-t:.0f}s")
    return pd.DataFrame(rows, columns=["indices", "syn"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--enum", action="store_true")
    ap.add_argument("--beams", default="500,1000,2000")
    args = ap.parse_args()
    q = Poly2Query()
    n = q.n_general
    m = args.order

    gt_path = ROOT / f"outputs/enum{m}_syn.parquet"
    strong = None
    if args.enum or gt_path.exists():
        if gt_path.exists() and not args.enum:
            gt = pd.read_parquet(gt_path)
            gt["indices"] = gt["indices"].apply(lambda t: tuple(int(v) for v in t))
        else:
            gt = full_enum(q, n, m)
            gt.to_parquet(gt_path, index=False)
        strong = {tau: set(gt[gt.syn > tau]["indices"]) for tau in TAUS}
        log("SCALE", "GT", note=f"order{m}: " + "; ".join(f">{t}:{len(strong[t])}" for t in TAUS))

    rng = np.random.RandomState(0)
    rows = []
    for B in [int(x) for x in args.beams.split(",")]:
        for key in ["syn", "hybrid"]:
            t = time.perf_counter()
            touched, scored = beam_search(q, n, B, key, target_size=m)
            dt = time.perf_counter() - t
            tm = touched[m]
            row = dict(order=m, beam=B, key=key, touched=len(tm),
                       full_comb=comb(n, m), touch_frac=len(tm) / comb(n, m), seconds=round(dt, 1))
            if strong is not None:
                for tau in TAUS:
                    found = {S for S in tm if scored[S][0] > tau}
                    row[f"cov>{tau}"] = round(len(found & strong[tau]) / max(len(strong[tau]), 1), 4)
                    rnd = set(tuple(sorted(rng.choice(n, m, replace=False))) for _ in range(len(tm)))
                    row[f"rand>{tau}"] = round(len(rnd & strong[tau]) / max(len(strong[tau]), 1), 4)
            rows.append(row)
            log("SCALE", "BEAM", note=f"order{m} B={B} {key} touch={len(tm)}({row['touch_frac']:.4f}) {dt:.0f}s")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"outputs/scale_order{m}.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
