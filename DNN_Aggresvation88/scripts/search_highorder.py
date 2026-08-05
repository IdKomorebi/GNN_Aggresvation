# -*- coding: utf-8 -*-
"""86 号：高阶协同的非枚举 beam 搜索（骑在 85 号 ms 级结构化查询上，零 GPU）。

GPT 85 号解决了"单次 v̂(S) 查询太贵"（缓存充分统计量，ms 级），但明确留下
"候选数量太多"——4 阶 C(44,4)=13.6万、6 阶 705万——仍需非枚举搜索。本脚本补这块。

方法：从二阶起逐层 beam：
  beam_m（宽 B）→ 每个集合加一个字段 → order-(m+1) 候选 → 打分 → 取 top-B → beam_{m+1}
"触及候选数" = 搜索实际查询的集合数（成本）。用 order-4 全枚举作 ground truth，
量化"beam 用多少枚举成本覆盖多少真强四阶"，并对比多种 beam key 与基线。

beam key（决定 beam 里保留谁；对应 84 号"该用哪种信号"的搜索层版本）：
  vmax   —— v̂(S) 最大（syn 的上界，82 号：擅长嵌套在强低阶上的高阶）
  syn    —— 纯高阶增量最大（擅长纯高阶，但增量估计噪声大）
  hybrid —— 两者各取 top-B 的并集（覆盖两类）

强协同阈值 τ：syn_m > 0.10 / 0.15 / 0.20 分档报告。
诚实：结构化查询是"攻击者下界"，非无条件 sup_f；真强四阶的独立认证在 certify_ho.py（另配 GPU）。
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query import Poly2Query  # noqa: E402
from runlog import log  # noqa: E402

TAUS = (0.10, 0.15, 0.20)


def full_enum_order4(q: Poly2Query, n: int) -> pd.DataFrame:
    """全枚举 C(n,4)，算每个四元组 syn4。作为 ground truth 参照。"""
    rows = []
    t = time.perf_counter()
    for i, quad in enumerate(combinations(range(n), 4)):
        s, c = q.syn(quad)
        rows.append((quad, s))
        if (i + 1) % 20000 == 0:
            log("ENUM4", "PROGRESS", note=f"{i+1} quads, {(i+1)/(time.perf_counter()-t):.0f}/s")
    df = pd.DataFrame(rows, columns=["indices", "syn4"])
    log("ENUM4", "DONE", note=f"{len(df)} quads in {time.perf_counter()-t:.1f}s")
    return df


def beam_search(q: Poly2Query, n: int, B: int, key: str, target_size: int = 4):
    """从 order-2 beam 到 order-target_size，返回 (touched_by_order, found_syn)。"""
    touched = {2: set(combinations(range(n), 2))}          # order-2 全看（946，便宜）
    scored = {tuple(sorted(p)): q.syn(p) for p in touched[2]}
    beam = _select(q, touched[2], B, key)
    for m in range(2, target_size):
        cand = set()
        for S in beam:
            for k in range(n):
                if k not in S:
                    cand.add(tuple(sorted(S + (k,))))
        touched[m + 1] = cand
        for S in cand:
            if S not in scored:
                scored[S] = q.syn(S)
        beam = _select(q, cand, B, key)
    return touched, scored


def _select(q: Poly2Query, cand, B: int, key: str):
    cand = list(cand)
    if key == "vmax":
        pri = [q.vmax(S) for S in cand]
        order = np.argsort(pri)[::-1][:B]
        return [cand[i] for i in order]
    if key == "syn":
        pri = [q.syn(S)[0] for S in cand]
        order = np.argsort(pri)[::-1][:B]
        return [cand[i] for i in order]
    if key == "hybrid":
        vm = np.argsort([q.vmax(S) for S in cand])[::-1][:B]
        sy = np.argsort([q.syn(S)[0] for S in cand])[::-1][:B]
        keep = list(dict.fromkeys(list(vm) + list(sy)))
        return [cand[i] for i in keep]
    raise ValueError(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum4", action="store_true", help="跑 order-4 全枚举 ground truth")
    ap.add_argument("--beams", default="50,100,200,500,1000")
    args = ap.parse_args()
    q = Poly2Query()
    n = q.n_general

    # ground truth（全枚举 order-4）
    gt_path = ROOT / "outputs/enum4_syn.parquet"
    if args.enum4 or not gt_path.exists():
        gt = full_enum_order4(q, n)
        gt.to_parquet(gt_path, index=False)
    else:
        gt = pd.read_parquet(gt_path)
        gt["indices"] = gt["indices"].apply(lambda t: tuple(int(v) for v in t))
    total4 = len(gt)
    strong = {tau: set(gt[gt.syn4 > tau]["indices"]) for tau in TAUS}
    log("SEARCH", "GT", note="; ".join(f">{tau}:{len(strong[tau])}" for tau in TAUS))

    rng = np.random.RandomState(0)
    rows = []
    for B in [int(x) for x in args.beams.split(",")]:
        for key in ["vmax", "syn", "hybrid"]:
            t = time.perf_counter()
            touched, scored = beam_search(q, n, B, key, target_size=4)
            dt = time.perf_counter() - t
            touched4 = touched[4]
            found = {tau: {S for S in touched4 if scored[S][0] > tau} for tau in TAUS}
            row = dict(beam=B, key=key, touched4=len(touched4),
                       touch_frac=len(touched4) / total4, seconds=round(dt, 1))
            for tau in TAUS:
                cov = len(found[tau] & strong[tau]) / max(len(strong[tau]), 1)
                row[f"cov>{tau}"] = round(cov, 4)
            # 同触及预算随机基线（覆盖对照）
            rand = set(map(lambda _: tuple(sorted(rng.choice(n, 4, replace=False))),
                           range(len(touched4))))
            for tau in TAUS:
                row[f"rand_cov>{tau}"] = round(len(rand & strong[tau]) / max(len(strong[tau]), 1), 4)
            rows.append(row)
            log("SEARCH", "BEAM", note=f"B={B} key={key} touch={len(touched4)}({row['touch_frac']:.3f}) "
                                       f"cov>0.15={row['cov>0.15']} in {dt:.0f}s")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/beam_frontier.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print(f"\n全枚举 order-4: {total4} 个; 强四阶 " +
          "; ".join(f">{tau}:{len(strong[tau])}" for tau in TAUS))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
