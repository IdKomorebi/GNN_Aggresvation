# -*- coding: utf-8 -*-
"""88 号 ★正面交付：【遗漏上界证书】——确定性搜索不可能时的正确替代。

四个搜索方向全部失败（见 diagnose_blindspot / antihier_peel / verify_clique_bound /
paired_check），指向不可能性结论：反层级高阶协同不存在优于枚举的确定性搜索。
于是把目标从"声称找全"改为**"给出遗漏的统计上界"**——这才是审计真正需要的
（风险上界），且理论上完备：要么找到，要么给出上界。

做法（均匀抽样 + 有限总体，Horvitz–Thompson 的简单形式）：
  1. beam 得到已触及集合 T（已发现强协同 = T 中 syn>τ 的）；
  2. 在**补集** V∖T 上均匀抽样 N 个，数出其中强协同 k 个；
  3. 补集强协同总数 M 的点估计 = k·|V∖T|/N，
     上界用 Clopper–Pearson 对密度的 97.5% 单侧上界 × |V∖T|；
  4. 报告：已发现 D 个；遗漏 ≤ M_hi（95% 置信）；召回率 ≥ D/(D+M_hi)。

★验证：order-4/5 有全枚举真值，可检验"真实遗漏是否 ≤ 上界"（名义 95%）。
"""
from __future__ import annotations

import argparse
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R86 = REPO / "DNN_Aggresvation86"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from query import Poly2Query  # noqa: E402
from search_highorder import beam_search  # noqa: E402
from runlog import log  # noqa: E402


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper–Pearson 单侧上界（比 Wilson 在 k=0 时更保守、更可靠）。"""
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(conf, k + 1, n - k))


def load_enum(m: int) -> dict:
    df = pd.read_parquet(R86 / f"outputs/enum{m}_syn.parquet")
    col = "syn4" if m == 4 else "syn"
    return {tuple(int(v) for v in t): s for t, s in zip(df["indices"], df[col])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="4,5")
    ap.add_argument("--beam", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--nsample", type=int, default=50000)
    args = ap.parse_args()

    q = Poly2Query()
    n = q.n_general
    rng = np.random.RandomState(88)
    rows = []

    for m in [int(x) for x in args.orders.split(",")]:
        total = comb(n, m)
        t = time.perf_counter()
        touched, scored = beam_search(q, n, args.beam, "syn", target_size=m)
        T = touched[m]
        found = {S for S in T if scored[S][0] > args.tau}
        log("MISS", "BEAM", note=f"order{m} touched={len(T)} found={len(found)} "
                                 f"{time.perf_counter()-t:.0f}s")

        # ---- 在补集上均匀抽样 ----
        comp_size = total - len(T)
        N = min(args.nsample, comp_size)
        seen, k = set(), 0
        t = time.perf_counter()
        while len(seen) < N:
            S = tuple(sorted(rng.choice(n, m, replace=False)))
            if S in T or S in seen:
                continue
            seen.add(S)
            if q.syn(S)[0] > args.tau:
                k += 1
            if len(seen) % 20000 == 0:
                log("MISS", "PROG", note=f"order{m} {len(seen)}/{N} k={k} "
                                         f"{len(seen)/(time.perf_counter()-t):.0f}/s")

        dens_hat = k / N
        dens_hi = cp_upper(k, N)
        miss_hat = dens_hat * comp_size
        miss_hi = dens_hi * comp_size
        D = len(found)
        row = dict(order=m, tau=args.tau, beam=args.beam, total=total,
                   touched=len(T), comp_size=comp_size, n_sample=N, k_strong_in_sample=k,
                   found=D, miss_hat=round(miss_hat, 1), miss_hi95=round(miss_hi, 1),
                   recall_hat=round(D / max(D + miss_hat, 1e-9), 4),
                   recall_lo95=round(D / max(D + miss_hi, 1e-9), 4),
                   sample_seconds=round(time.perf_counter() - t, 1))

        # ---- 验证（仅 4/5 阶有全枚举真值）----
        if m in (4, 5):
            enum = load_enum(m)
            strong_all = {S for S, v in enum.items() if v > args.tau}
            true_miss = len(strong_all - T)
            row["true_total_strong"] = len(strong_all)
            row["true_miss"] = true_miss
            row["true_recall"] = round(len(strong_all & T) / len(strong_all), 4)
            row["bound_holds"] = bool(true_miss <= miss_hi)
        rows.append(row)
        log("MISS", "ROW", note=f"order{m} found={D} miss_hat={miss_hat:.1f} "
                                f"miss_hi={miss_hi:.1f} true_miss={row.get('true_miss')} "
                                f"holds={row.get('bound_holds')}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/miss_bound.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(out.to_string(index=False))
    print("\n=== 证书形式（可直接写进审计报告）===")
    for _, r in out.iterrows():
        print(f"  order-{int(r['order'])} (τ={r['tau']}): 已发现强协同 {int(r['found'])} 个；"
              f"未搜索区域({int(r['comp_size']):,}个集合)中遗漏 ≤ {r['miss_hi95']:.0f} 个(95%置信)；"
              f"⟹ 召回率 ≥ {r['recall_lo95']:.1%}"
              + (f"　[真值核验：实际遗漏 {int(r['true_miss'])}，上界"
                 f"{'成立 ✓' if r['bound_holds'] else '失效 ✗'}]" if 'true_miss' in r and pd.notna(r.get('true_miss')) else ""))


if __name__ == "__main__":
    main()
