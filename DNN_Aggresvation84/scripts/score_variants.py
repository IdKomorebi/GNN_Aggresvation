# -*- coding: utf-8 -*-
"""84 号 杠杆A 打分：对比各 oracle 变体在设计三元组上的 K=0 强档召回与机械倒挂。

核心问句（80 号）：单调/增量约束能否让最强档 syn3_true>0.20 的 S1 不再变负、召回抬起来，
且不牺牲整体？逐变体报告：
- 负 S1 数 / 强档负 S1 数（机械倒挂指纹）
- 父低估：mean(vijk − best_pair) @ 强档（<0 = 倒挂未修）
- Spearman(S1, syn3_true)
- Top-30% 加权召回：整体 + 分强度档（对齐 80 号口径）
- S2(上界筛) 强档召回（对照）

无模型选择用到真值 → 用全部 2197 评估合规。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BANDS = [(0.10, 0.12), (0.12, 0.15), (0.15, 0.20), (0.20, 1.01)]


def wrecall_top(df: pd.DataFrame, key: str, lo: float, hi: float, frac: float = 0.30) -> float:
    """按 key 降序取 top frac，返回该强度档 (lo,hi] 真强三元组落入 top 的加权召回。"""
    n_top = int(round(len(df) * frac))
    top = set(df.sort_values(key, ascending=False).head(n_top)["indices"])
    band = df[(df.syn3_true > lo) & (df.syn3_true <= hi)]
    if band.weight.sum() == 0:
        return float("nan")
    hit = band[band["indices"].isin(top)]
    return float(hit.weight.sum() / band.weight.sum())


def summarize(name: str, path: Path) -> dict:
    df = pd.read_parquet(path)
    df["indices"] = df["indices"].apply(lambda t: tuple(int(v) for v in t))
    strong = df[df.syn3_true > 0.20]
    row = {
        "variant": name,
        "S1<0_all": int((df.S1 < 0).sum()),
        "S1<0_strong": int((strong.S1 < 0).sum()),
        "n_strong": len(strong),
        "parent_gap_strong": float((strong.vijk_at_cstar - strong.bestpair_at_cstar).mean()),
        "spearman_S1": float(spearmanr(df.S1, df.syn3_true).correlation),
        "rec_all": wrecall_top(df, "S1", 0.10, 1.01),
    }
    for lo, hi in BANDS:
        row[f"recS1_{lo:g}-{hi:g}"] = wrecall_top(df, "S1", lo, hi)
    row["recS2_>0.2"] = wrecall_top(df, "S2", 0.20, 1.01)
    return row


def main() -> None:
    variants = {
        "baseline_uniform(75)": ROOT / "outputs/eval_baseline_uniform.parquet",
        "none(sanity)": ROOT / "outputs/eval_none_lam0.parquet",
        "mono_lam2": ROOT / "outputs/eval_mono_lam2.parquet",
        "mono_lam10": ROOT / "outputs/eval_mono_lam10.parquet",
        "incr_lam2": ROOT / "outputs/eval_incr_lam2.parquet",
        "incr_lam10": ROOT / "outputs/eval_incr_lam10.parquet",
        "mono_incr_lam5": ROOT / "outputs/eval_mono_incr_lam5.parquet",
    }
    rows = [summarize(n, p) for n, p in variants.items() if p.exists()]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/variant_scores.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.3f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
