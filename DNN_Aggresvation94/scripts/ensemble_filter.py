# -*- coding: utf-8 -*-
"""94 号 分析③：★跨 seed 集成过滤——把 top 精确率从 ~48% 提到 ~100%。

发现：seed0 认证的 9 个真 o4 协同在 seed1/seed2 扫描里也排最前
（跨 seed 几何平均排名中位 105/135,751），而 11 个假候选跨 seed 漂移
（中位 27,647）——**分离 263 倍**。
⟹ 用"多 seed 一致性"做免费的候选过滤：cross_rank = √(rank_s1·rank_s2) ≤ 500
在认证样本上 9/9 真全保留、11/11 假全剔除。

⚠ 诚实标注：阈值在同一批认证集合上选的（20 个），需前瞻验证；
但 263× 的分离幅度使结论对阈值极不敏感（300–500 之间结果相同）。
机制上与 84 号"σ 在最强档失明"不矛盾：那是**同一模型**集成成员的方差
（共享偏差），这里是**独立训练**的排序一致性（打破共享偏差）。
"""
from __future__ import annotations

import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R93 = ROOT.parent / "DNN_Aggresvation93"


def load(prefix):
    fs = glob.glob(str(R93 / f"outputs/{prefix}_s*of*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs])
    d["S"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    return d.set_index("S")["syn"]


def main() -> None:
    s1 = load("scan_o4_oracle_l1_aug8_seed1_last")
    s2 = load("scan_o4_oracle_l1_aug8_seed2_last")
    cert = pd.concat([pd.read_csv(f) for f in
                      glob.glob(str(R93 / "outputs/certify_o4_l1top_s*.csv"))])
    cert = cert[cert.group == "strong"].drop_duplicates("S").copy()
    cert["Stup"] = cert["S"].map(lambda s: tuple(ast.literal_eval(s)))
    r1 = {S: i + 1 for i, S in enumerate(s1.sort_values(ascending=False).index)}
    r2 = {S: i + 1 for i, S in enumerate(s2.sort_values(ascending=False).index)}
    cert["rank_s1"] = cert.Stup.map(r1)
    cert["rank_s2"] = cert.Stup.map(r2)
    cert["cross_rank"] = np.sqrt(cert.rank_s1 * cert.rank_s2)
    cert["true"] = cert.syn_true_audit > 0.10
    t, f = cert[cert.true], cert[~cert.true]
    print(f"真协同({len(t)}) 跨seed排名中位 {t.cross_rank.median():.0f}   "
          f"假候选({len(f)}) 中位 {f.cross_rank.median():.0f}   "
          f"分离 {f.cross_rank.median()/t.cross_rank.median():.0f}×")
    for thr in (300, 500, 1000):
        sel = cert[cert.cross_rank <= thr]
        print(f"  过滤 cross_rank<={thr}: 精确率 {sel.true.mean()*100:.0f}% "
              f"({int(sel.true.sum())}/{len(sel)})，召回 {sel.true.sum()}/{len(t)}")
    cert.sort_values("cross_rank").to_csv(ROOT / "outputs/ensemble_filter_o4.csv", index=False)


if __name__ == "__main__":
    main()
