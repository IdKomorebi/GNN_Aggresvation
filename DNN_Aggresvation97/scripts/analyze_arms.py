# -*- coding: utf-8 -*-
"""97 号判决表：五个训练臂 × (高阶搜索能力, 真实 conf 尾部保真度)。

参照系（96 号已测，同一批注入元组、同一 400 随机对照）：
  现役 φ（93 号 aug8-raw-2..4）  k=6 中位百分位 12.0，k=7 为 0.4
  配基 full 字典                 k=6/7 均为 100  ← 上限参照，非可外推目标
随机基线 = 50。
"""
import glob
from pathlib import Path

import pandas as pd
import torch

R96 = Path("../DNN_Aggresvation96")
ARMS = ["r93x", "noaug", "noaug_s1", "noaug_s2", "r93", "rawhi", "tanh",
        "tanhboth", "tanhboth_s1", "tanhboth_s2"]

rows = []
for a in ARMS:
    import glob as _g
    ck = torch.load(_g.glob(f"outputs/oracle_l1_{a}_seed*.pt")[0], map_location="cpu",
                    weights_only=False)
    rec = {"臂": a, "val_conf": round(ck["val_conf"], 5),
           "val_syn": round(ck["val_syn"], 5), "ep": ck["epochs_run"]}
    p = R96 / f"outputs/inject_search_a{a}_pjm_last_s0of1.csv"
    if p.exists():
        d = pd.read_csv(p)
        for r2 in (0.20, 0.35):
            s = d[d.r2_inject == r2]
            for k in sorted(s.order.unique()):
                v = s[s.order == k].percentile
                rec[f"k{k}@{r2:.2f}"] = f"{v.median():.0f}({(v>=95).sum()}/{len(v)})"
    # ★精确匹配：早先用通配符导致 tanh 匹到 tanhboth、r93 匹到 r93x
    for q in glob.glob(f"outputs/*_a{a}.csv"):
        e = pd.read_csv(q)
        if "rho_tail" in e.columns:
            rec["二阶ρ"] = round(float(e.rho.iloc[0]), 4)
            rec["二阶尾部ρ"] = round(float(e.rho_tail.iloc[0]), 4)
            break
    rows.append(rec)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\n注：格式 = 中位百分位(命中数/总数)，命中 = 百分位≥95；随机基线 50。")
print("参照：现役 φ k6=12 k7=0.4；配基 full k6=k7=100（上限参照，不可外推）。")
