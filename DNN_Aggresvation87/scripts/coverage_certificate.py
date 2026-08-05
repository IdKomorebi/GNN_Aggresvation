# -*- coding: utf-8 -*-
"""87 号 任务②：beam 搜索的【覆盖率证书】——给定预算，保证覆盖率至少多少（95% 置信）。

在有全枚举 GT 的 order-4/5 上，扫 beam 宽度 → (成本, 覆盖率, Wilson 95% 下界)。
覆盖率本身在全枚举下是**精确值**（分母是全部强集合）；Wilson 下界的含义是：
把这批强集合视为"该类数据上强协同"的一个有限样本，对**方法在同类数据上的期望覆盖率**
做外推的保守下界。两者都报，并显式区分——这是论文里 recall±CI 的标准做法。

产出"逆查表"：要达到 ≥X% 的覆盖率下界，需要多大 beam / 多少枚举成本。
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
R86 = REPO / "DNN_Aggresvation86"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from query import Poly2Query  # noqa: E402
from search_highorder import beam_search  # noqa: E402
from certify_coverage import wilson  # noqa: E402
from runlog import log  # noqa: E402

TAUS = (0.05, 0.10, 0.15)


def load_exact(m: int):
    p = R86 / f"outputs/enum{m}_syn.parquet"
    df = pd.read_parquet(p)
    col = "syn4" if m == 4 else "syn"
    sets = [tuple(int(v) for v in t) for t in df["indices"]]
    return sets, df[col].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="4,5")
    ap.add_argument("--beams", default="100,200,500,1000,2000")
    args = ap.parse_args()
    q = Poly2Query()
    n = q.n_general

    rows = []
    for m in [int(x) for x in args.orders.split(",")]:
        sets, syns = load_exact(m)
        strong = {tau: [S for S, v in zip(sets, syns) if v > tau] for tau in TAUS}
        for B in [int(x) for x in args.beams.split(",")]:
            t = time.perf_counter()
            touched, _ = beam_search(q, n, B, "syn", target_size=m)
            dt = time.perf_counter() - t
            tm = touched[m]
            row = dict(order=m, beam=B, touched=len(tm), full_comb=comb(n, m),
                       cost_frac=round(len(tm) / comb(n, m), 5), seconds=round(dt, 1))
            for tau in TAUS:
                S_tau = strong[tau]
                if not S_tau:
                    continue
                hit = sum(1 for S in S_tau if S in tm)
                lo, hi = wilson(hit, len(S_tau))
                row[f"n>{tau}"] = len(S_tau)
                row[f"cov>{tau}"] = round(hit / len(S_tau), 4)
                row[f"lo95>{tau}"] = round(lo, 4)
            rows.append(row)
            log("CERT", "ROW", note=f"order{m} B={B} cost={row['cost_frac']:.4f} "
                                    f"cov>0.10={row.get('cov>0.1')} lo95={row.get('lo95>0.1')}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/coverage_certificate.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print(out.to_string(index=False))

    # ---- 逆查表：要达到 ≥X 的 95% 下界，最小成本是多少 ----
    print("\n=== 逆查表：达到覆盖率 95% 置信下界 ≥ X 所需的最小 beam 与成本 ===")
    inv = []
    for m in out.order.unique():
        sub = out[out.order == m].sort_values("beam")
        for tau in TAUS:
            c = f"lo95>{tau}"
            if c not in sub:
                continue
            for target in (0.50, 0.70, 0.80, 0.90):
                ok = sub[sub[c] >= target]
                if len(ok):
                    r = ok.iloc[0]
                    inv.append(dict(order=m, tau=tau, target_lo95=target,
                                    beam=int(r.beam), cost_frac=r.cost_frac,
                                    actual_cov=r[f"cov>{tau}"], actual_lo95=r[c]))
                else:
                    inv.append(dict(order=m, tau=tau, target_lo95=target, beam=None,
                                    cost_frac=None, actual_cov=None, actual_lo95=None))
    dfi = pd.DataFrame(inv)
    dfi.to_csv(ROOT / "outputs/coverage_inverse_table.csv", index=False)
    print(dfi.to_string(index=False))


if __name__ == "__main__":
    main()
