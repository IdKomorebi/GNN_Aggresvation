# -*- coding: utf-8 -*-
"""87 号：高阶 beam 搜索的【覆盖率证书】+ order-6/7/8 成本 benchmark（零 GPU）。

两件事一次做完：

任务①（成本 benchmark）：order-6/7/8 全枚举不可行（705万/1.4亿/1.77亿），
只能报 beam 触及数与耗时，与枚举组合数对比，证明 beam 是唯一可行路径。

任务②（覆盖率保证形式化）：全枚举不可行时如何仍给出覆盖率保证？
用【均匀随机抽样作无偏 GT】——从 C(44,m) 均匀抽 N 个集合算 syn，
落在"强"档的那些是全体强集合的均匀子样本，于是
    coverage_hat = #(抽样强 ∩ beam触及) / #(抽样强)
是真覆盖率的无偏估计，二项分布 → **Wilson 95% 置信下界**即为覆盖率证书。

★严谨性自检：order-4/5 有全枚举 GT，抽样估计必须落在全枚举真值附近，
否则抽样方法本身不可信。本脚本对 4/5 同时报"全枚举真值 vs 抽样估计+CI"。
"""
from __future__ import annotations

import argparse
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from query import Poly2Query  # noqa: E402
from search_highorder import beam_search  # noqa: E402
from runlog import log  # noqa: E402

TAUS = (0.05, 0.10, 0.15)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（小样本下比正态近似可靠）。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sample_sets(n_general: int, m: int, n_sample: int, rng) -> list[tuple[int, ...]]:
    """从 C(n,m) 均匀抽样（去重）。"""
    seen = set()
    while len(seen) < n_sample:
        need = n_sample - len(seen)
        for _ in range(need):
            seen.add(tuple(sorted(rng.choice(n_general, m, replace=False))))
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="6", help="逗号分隔，如 6,7,8")
    ap.add_argument("--beams", default="1000")
    ap.add_argument("--nsample", type=int, default=300000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    q = Poly2Query()
    n = q.n_general
    rng = np.random.RandomState(87)
    rows = []

    for m in [int(x) for x in args.orders.split(",")]:
        # ---- 抽样无偏 GT ----
        t0 = time.perf_counter()
        n_samp = min(args.nsample, comb(n, m))
        samp = sample_sets(n, m, n_samp, rng)
        syns = np.empty(len(samp))
        for i, S in enumerate(samp):
            syns[i] = q.syn(S)[0]
            if (i + 1) % 50000 == 0:
                log("SAMPLE", "PROG", note=f"order{m} {i+1}/{len(samp)} "
                                           f"{(i+1)/(time.perf_counter()-t0):.0f}/s")
        t_samp = time.perf_counter() - t0
        dens = {tau: float((syns > tau).mean()) for tau in TAUS}
        log("SAMPLE", "DONE", note=f"order{m} N={len(samp)} {t_samp:.0f}s; "
            + "; ".join(f">{tau}:{(syns>tau).sum()}({dens[tau]*100:.4f}%)" for tau in TAUS)
            + f"; max_syn={syns.max():.3f}")

        # ---- beam × 各宽度，用抽样 GT 估覆盖率 ----
        for B in [int(x) for x in args.beams.split(",")]:
            t1 = time.perf_counter()
            touched, scored = beam_search(q, n, B, "syn", target_size=m)
            t_beam = time.perf_counter() - t1
            tm = touched[m]
            row = dict(order=m, beam=B, n_sample=len(samp),
                       full_comb=comb(n, m), touched=len(tm),
                       touch_frac=len(tm) / comb(n, m),
                       beam_seconds=round(t_beam, 1), sample_seconds=round(t_samp, 1),
                       max_syn_sampled=round(float(syns.max()), 4))
            for tau in TAUS:
                idx = np.flatnonzero(syns > tau)
                n_strong = len(idx)
                k_hit = sum(1 for i in idx if samp[i] in tm)
                lo, hi = wilson(k_hit, n_strong)
                row[f"dens>{tau}"] = dens[tau]
                row[f"nstrong>{tau}"] = n_strong
                row[f"cov>{tau}"] = round(k_hit / n_strong, 4) if n_strong else float("nan")
                row[f"lo95>{tau}"] = round(lo, 4)
                row[f"hi95>{tau}"] = round(hi, 4)
                # 同预算随机基线（同样用抽样 GT 估）
                rnd = set(sample_sets(n, m, min(len(tm), comb(n, m)), rng))
                k_rnd = sum(1 for i in idx if samp[i] in rnd)
                row[f"rand>{tau}"] = round(k_rnd / n_strong, 4) if n_strong else float("nan")
            rows.append(row)
            log("BEAM", "DONE", note=f"order{m} B={B} touch={len(tm)}({row['touch_frac']:.5f}) "
                                     f"{t_beam:.0f}s cov>0.10={row.get('cov>0.1')}")

    out = pd.DataFrame(rows)
    suffix = args.tag or args.orders.replace(",", "_")
    out.to_csv(ROOT / f"outputs/coverage_cert_{suffix}.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
