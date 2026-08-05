# -*- coding: utf-8 -*-
"""97 号大样本判决：4 个臂族 × 3 种子 × 24 元组，池化后做统计检验。

为什么要池化：6 元组时中位百分位在 1-99 之间跳（tanhboth 在 k=6 的三个种子给
95/1/99），单种子差异不可信。本表每格 72 个元组，并给出与基线的 Fisher 精确检验。

基线 = r93x（真·复刻 93 号，含 support/query 元组错配），即现役 φ 的对照。
命中 = 百分位 >= 95（400 个同阶随机集合中排进前 5%）。随机基线命中率 5%。
"""
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

R96 = Path("../DNN_Aggresvation96/outputs")
FAM = {"r93x": ["r93x", "r93x_s1", "r93x_s2"],
       "noaug": ["noaug", "noaug_s1", "noaug_s2"],
       "r93(修错配)": ["r93", "r93_s1", "r93_s2"],
       "tanhboth": ["tanhboth", "tanhboth_s1", "tanhboth_s2"]}

data = {}
for fam, arms in FAM.items():
    ds = []
    for a in arms:
        p = R96 / f"inject_search_b{a}_pjm_last_s0of1.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["arm"] = a
            ds.append(d)
    if ds:
        data[fam] = pd.concat(ds, ignore_index=True)

base = data.get("r93x")
for r2 in (0.20, 0.35):
    print(f"\n===== 注入 R² = {r2:.2f} =====")
    print(f"{'臂':<14}{'k=5':>18}{'k=6':>18}{'k=7':>18}")
    for fam, d in data.items():
        cells = []
        for k in (5, 6, 7):
            s = d[(d.r2_inject == r2) & (d.order == k)].percentile
            if len(s) == 0:
                cells.append("—")
                continue
            hit, n = int((s >= 95).sum()), len(s)
            txt = f"{hit/n*100:.0f}%({hit}/{n}) 中{s.median():.0f}"
            if fam != "r93x" and base is not None:
                b = base[(base.r2_inject == r2) & (base.order == k)].percentile
                if len(b):
                    bh = int((b >= 95).sum())
                    p = fisher_exact([[hit, n - hit], [bh, len(b) - bh]],
                                     alternative="greater")[1]
                    txt += f" p={p:.3f}"
            cells.append(txt)
        print(f"{fam:<14}" + "".join(f"{c:>18}" for c in cells))
print("\n命中 = 百分位≥95（400 个同阶随机集合中的前 5%）；随机基线命中率 5%。")
print("p 值 = 对 r93x 基线的 Fisher 精确检验（单侧，备择：本臂命中率更高）。")
