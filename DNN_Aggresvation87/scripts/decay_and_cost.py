# -*- coding: utf-8 -*-
"""87 号：纯高阶协同的【强度衰减曲线】+ beam 覆盖率证书 + order-6/7/8 成本 benchmark。

三件事一次做完（零 GPU）：

A. **衰减曲线（主科学产出）**：order 3..8 的强协同密度与 max_syn。
   order-3/4/5 用全枚举（精确）；order-6/7/8 用均匀随机抽样（无偏 GT）。
   ★抽样方法已在 order-4 验证：三个阈值的 95%CI 全部覆盖全枚举真值。
   高阶密度为 0 时给 **Wilson 置信上界**——这是"高阶没什么可搜"的统计证书。

B. **覆盖率证书（任务②）**：beam 在各阶各宽度下的覆盖率 + Wilson 95% 区间。
   4/5 阶分母来自全枚举强集合（精确覆盖率，CI 作外推区间）；
   高阶若强集合为空则不报覆盖率（诚实：分母为 0 无法估计）。

C. **成本 benchmark（任务①）**：一次 beam 到 order-8 即可拿到各层 touched，
   与 C(44,m) 枚举数对比，证明 beam 触及近线性而枚举指数爆炸。

效率：单次 beam(target=8) 复用所有中间层，省 3 倍时间。
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
from certify_coverage import wilson, sample_sets  # noqa: E402
from runlog import log  # noqa: E402

TAUS = (0.05, 0.10, 0.15, 0.20)


def exact_gt(q: Poly2Query, n: int, m: int) -> tuple[list, np.ndarray]:
    """全枚举（3 阶现算；4/5 阶复用 86 号）。"""
    from itertools import combinations
    if m in (4, 5):
        p = R86 / f"outputs/enum{m}_syn.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            col = "syn4" if m == 4 else "syn"
            sets = [tuple(int(v) for v in t) for t in df["indices"]]
            return sets, df[col].to_numpy()
    t = time.perf_counter()
    sets = list(combinations(range(n), m))
    syns = np.array([q.syn(S)[0] for S in sets])
    log("ENUM", "DONE", note=f"order{m} 全枚举 {len(sets)} in {time.perf_counter()-t:.0f}s")
    return sets, syns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=1000)
    ap.add_argument("--max-order", type=int, default=8)
    ap.add_argument("--nsample", type=int, default=50000)
    args = ap.parse_args()

    q = Poly2Query()
    n = q.n_general
    rng = np.random.RandomState(87)

    # ---- C. 一次 beam 到 max_order，拿所有层 touched ----
    t0 = time.perf_counter()
    touched, scored = beam_search(q, n, args.beam, "syn", target_size=args.max_order)
    t_beam = time.perf_counter() - t0
    log("BEAM", "DONE", note=f"beam到order{args.max_order} B={args.beam} 用时{t_beam:.0f}s; "
        + "; ".join(f"o{m}:{len(touched[m])}" for m in sorted(touched)))

    rows = []
    for m in range(3, args.max_order + 1):
        exact = m <= 5
        if exact:
            sets, syns = exact_gt(q, n, m)
            n_eval = len(sets)
            src = "全枚举"
        else:
            t = time.perf_counter()
            n_eval = min(args.nsample, comb(n, m))
            sets = sample_sets(n, m, n_eval, rng)
            syns = np.empty(n_eval)
            for i, S in enumerate(sets):
                syns[i] = q.syn(S)[0]
                if (i + 1) % 20000 == 0:
                    log("SAMPLE", "PROG", note=f"order{m} {i+1}/{n_eval} "
                                               f"{(i+1)/(time.perf_counter()-t):.0f}/s")
            src = "抽样"
            log("SAMPLE", "DONE", note=f"order{m} N={n_eval} {time.perf_counter()-t:.0f}s "
                                       f"max_syn={syns.max():.4f}")

        tm = touched.get(m, set())
        row = dict(order=m, source=src, n_eval=n_eval, full_comb=comb(n, m),
                   touched=len(tm), touch_frac=len(tm) / comb(n, m),
                   max_syn=round(float(syns.max()), 4),
                   median_syn=round(float(np.median(syns)), 4))
        for tau in TAUS:
            k = int((syns > tau).sum())
            dlo, dhi = wilson(k, n_eval)
            row[f"n>{tau}"] = k
            row[f"dens>{tau}"] = round(k / n_eval, 6)
            row[f"dens_hi95>{tau}"] = round(dhi, 6)          # 密度置信上界（负结果证书）
            row[f"est_total>{tau}"] = int(round(k / n_eval * comb(n, m)))
            if k > 0:
                idx = np.flatnonzero(syns > tau)
                hit = sum(1 for i in idx if sets[i] in tm)
                lo, hi = wilson(hit, k)
                row[f"cov>{tau}"] = round(hit / k, 4)
                row[f"cov_lo95>{tau}"] = round(lo, 4)
                row[f"cov_hi95>{tau}"] = round(hi, 4)
            else:
                row[f"cov>{tau}"] = float("nan")
                row[f"cov_lo95>{tau}"] = float("nan")
                row[f"cov_hi95>{tau}"] = float("nan")
        rows.append(row)
        log("DECAY", "ROW", note=f"order{m} {src} max_syn={row['max_syn']} "
                                 f"n>0.10={row['n>0.1']} cov>0.10={row.get('cov>0.1')}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"outputs/decay_cost_B{args.beam}.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 60)
    key = ["order", "source", "n_eval", "full_comb", "touched", "touch_frac",
           "max_syn", "n>0.1", "dens>0.1", "dens_hi95>0.1", "est_total>0.1",
           "cov>0.1", "cov_lo95>0.1"]
    print(out[key].to_string(index=False))
    print(f"\nbeam 总耗时 {t_beam:.0f}s")


if __name__ == "__main__":
    main()
