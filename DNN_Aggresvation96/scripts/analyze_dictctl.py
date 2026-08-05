# -*- coding: utf-8 -*-
"""字典对照汇总：同一批注入元组上比较 φ(last) 与手工 full 字典的搜索端命中。

判读要点（写进结论）：
  full 字典**显式含有生成信号所用的那个 k 阶单项式** ∏z_i，对合成注入信号是
  "配基"（近似 oracle 基）；φ 是为 12 个真实机密字段元训练的，对合成信号分布外。
  故本表比较的是**基是否匹配**，不是"哪个字典在真实数据上更好"
  （真实数据上 full 已被 94/95 号证伪：order-5 full 与随机无异 p=0.076）。
"""
import glob
import pandas as pd

rows = []
for p in glob.glob("outputs/inject_search_m6_*_s0of1.csv"):
    name = p.split("inject_search_m6_")[1].rsplit("_s0of1", 1)[0]
    ds, kind = name.split("_")
    d = pd.read_csv(p)
    d["dataset"], d["dict"] = ds, kind
    rows.append(d)
df = pd.concat(rows, ignore_index=True)

for r2 in sorted(df.r2_inject.unique()):
    s = df[df.r2_inject == r2]
    hit = s.groupby(["dataset", "dict", "order"]).percentile.agg(
        命中率_p95=lambda x: f"{(x >= 95).mean()*100:.0f}%",
        中位百分位=lambda x: round(x.median(), 1), n="count")
    print(f"\n===== 注入 R² = {r2} =====")
    print(hit.unstack("dict").to_string())
