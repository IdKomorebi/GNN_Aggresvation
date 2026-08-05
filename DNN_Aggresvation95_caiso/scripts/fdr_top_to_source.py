#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""95_caiso：把 paired_fdr 的 order-3 结果转成 certify_highorder 可读的 source parquet。

取 τ=0.10 过 FDR 的集合按 syn_audit 降序（供认证"o3 最强真值"= 衰减律第一行），
格式对齐 scan_order5 产出：indices/syn/conf/v_set + _s0of1.parquet 命名。
v_set 无从获得（FDR 不算 v），置 NaN——certify 只用 syn/conf 与 artifact 标记，
artifact 判定 syn==v_set 在 NaN 下自然为 False。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "outputs/paired_fdr_o3_cat3.csv")
col_rej = "rej@0.1"
if col_rej not in df.columns:
    cands = [c for c in df.columns if c.startswith("rej@0.1")]
    assert cands, df.columns.tolist()
    col_rej = cands[0]
# 全体集合都进源文件（certify 的对照组要从 syn<0.05 抽样）；剔除病态行
# （|syn_audit|>1 = 闭式解在 audit 极端样本上爆炸，FDR 已证其不通过，
#   但混进对照池会当"弱集合"被抽走，认证浪费且语义错）。
# 未过 FDR 的集合把 syn 压到 min(syn, 0.05-ε) 之下？不——保持原值，
# strong 组由"FDR 通过 + syn 降序"决定：把未通过者 syn 置为其原值但
# 通过者排前的排序天然由 syn 值保证（top10 全部 >0.27 且全过 FDR）。
clean = df[df.syn_audit.abs() <= 1].copy()
passed = clean[clean[col_rej] == True]  # noqa: E712
weak = clean[(clean[col_rej] != True) & (clean.syn_audit < 0.05)]  # noqa: E712
# 只保留"FDR 通过（strong 候选）∪ 真弱（对照池）"；中间地带（syn 高但不显著，
# 含半病态外推行）既不该当 strong 也不配当对照，全部排除。
keep = pd.concat([passed, weak], ignore_index=True)
assert keep.nlargest(10, "syn_audit")[col_rej].all(), "top10 必须全过 FDR"
out = pd.DataFrame(dict(indices=keep["S"], syn=keep["syn_audit"],
                        conf=keep["conf"], v_set=np.nan))
p = ROOT / "outputs/fdr_o3_top_s0of1.parquet"
out.to_parquet(p, index=False)
print(f"{p.name}: {len(out)} 集合（含 {len(passed)} 个过 FDR），"
      f"top syn_audit = {out.syn.nlargest(5).round(4).tolist()}，"
      f"对照池(syn<0.05) {int((out.syn < 0.05).sum())} 个")
