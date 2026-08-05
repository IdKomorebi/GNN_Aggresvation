# -*- coding: utf-8 -*-
"""DNN70：低阶代理 + 层级性统计检验（全部用现成重训真值，不重训）。

核心问题：集合敏感度 v_c(S) 是否由其低阶子结构决定？
- 若"最强低阶子集"预测器能在留出的宽尺寸集合上贴合真值，则任意 S 的评估
  可塌缩为"检查它是否包含危险的小子集"，即 O(n^2)~O(n^3)，回应可扩展性质疑。
- 残差 = 真值 - 低阶预测，量化高阶(≥4)贡献；按 |S| 分层看它是否随规模增长。

真值来源（DNN_Aggresvation69/outputs）：
  subsets.json     : 1437 子集 → fields/group/canonical
  truth_long.csv   : 每 (sid, conf) 的重训真值 dnn（攻击者最优响应，本文真值口径）
可用低阶真值：44 单字段 + 全部 946 字段对（完备）+ 397 认证三元组（部分）。
留出验证集：50 个宽尺寸随机集合（|S| 1..43）。
"""
import json, itertools, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SRC = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/outputs"
OUT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation70/outputs"
NOISE = 0.004  # 60 号实测噪声底上界（σ≈0.002-0.004）

reg = json.load(open(f"{SRC}/subsets.json"))
tl = pd.read_csv(f"{SRC}/truth_long.csv")
# 每 (sid,conf) 的真值（去重条目仍保留真值，一致）
truth = {(r.sid, r.conf): r.dnn for r in tl.itertuples()}
CONFS = sorted(tl.conf.unique())

# ---- 构建低阶真值表（按字段集合索引）----
single_v = {c: {} for c in CONFS}   # single_v[conf][field] = v
pair_v   = {c: {} for c in CONFS}   # pair_v[conf][frozenset(fi,fj)] = v
triple_v = {c: {} for c in CONFS}   # triple_v[conf][frozenset(3)] = v

for sid, meta in reg.items():
    g, fields = meta["group"], meta["fields"]
    fs = frozenset(fields)
    for c in CONFS:
        v = truth.get((sid, c))
        if v is None:
            continue
        if g == "single":
            single_v[c][fields[0]] = v
        elif g == "pair":
            pair_v[c][fs] = v
        elif g.startswith("triple"):
            triple_v[c][fs] = v

n_fields = len(single_v[CONFS[0]])
n_pairs = len(pair_v[CONFS[0]])
n_triples = len(triple_v[CONFS[0]])
print(f"低阶真值表: 单字段 {n_fields}, 字段对 {n_pairs}, 三元组 {n_triples}")

# ---- 留出集：宽尺寸随机集合 ----
wide = {sid: meta for sid, meta in reg.items() if meta["group"] == "wide_random"}
print(f"宽尺寸留出集: {len(wide)} 个, 规模范围 {min(len(m['fields']) for m in wide.values())}-{max(len(m['fields']) for m in wide.values())}")

# ---- 代理预测器（v_c 单调 → 均为下界估计）----
def predict(fields, conf, order):
    fset = list(fields)
    best = 0.0
    # order>=1: 最强单字段
    for f in fset:
        best = max(best, single_v[conf].get(f, 0.0))
    if order >= 2:
        for i, j in itertools.combinations(fset, 2):
            best = max(best, pair_v[conf].get(frozenset((i, j)), 0.0))
    if order >= 3:
        # 只有认证过的三元组有真值；其余回退到 order-2（即不提升）
        for tri in itertools.combinations(fset, 3):
            v = triple_v[conf].get(frozenset(tri))
            if v is not None:
                best = max(best, v)
    return best

# 加性 Möbius 二阶（对照：若可加则应准，实测应因饱和而崩）
def predict_additive(fields, conf):
    fset = list(fields)
    s = sum(single_v[conf].get(f, 0.0) for f in fset)
    for i, j in itertools.combinations(fset, 2):
        vij = pair_v[conf].get(frozenset((i, j)))
        if vij is None:
            continue
        m2 = vij - single_v[conf].get(i, 0.0) - single_v[conf].get(j, 0.0)  # 二阶 Möbius 系数
        s += m2
    return float(np.clip(s, 0.0, 1.0))

# ---- 在留出集上评测（只用 |S|>=5 才是真正的外推）----
rows = []
for sid, meta in wide.items():
    fields = meta["fields"]; sz = len(fields)
    for c in CONFS:
        t = truth.get((sid, c))
        if t is None:
            continue
        rows.append(dict(sid=sid, size=sz, conf=c, truth=t,
                         m1=predict(fields, c, 1),
                         m2=predict(fields, c, 2),
                         m3=predict(fields, c, 3),
                         add=predict_additive(fields, c)))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/surrogate_predictions.csv", index=False)

def metrics(sub, col):
    e = sub[col] - sub["truth"]
    sp = spearmanr(sub[col], sub["truth"]).correlation
    return dict(n=len(sub), mae=e.abs().mean(), bias=e.mean(),
                spearman=sp, resid_mean=(sub["truth"] - sub[col]).mean(),
                resid_p90=(sub["truth"] - sub[col]).quantile(0.9))

# 全体 & 仅 |S|>=5（真正外推）
summ = []
for label, sub in [("all_wide", df), ("size>=5", df[df["size"] >= 5]),
                   ("size>=9", df[df["size"] >= 9]), ("size>=17", df[df["size"] >= 17])]:
    for col, name in [("m1", "最强单字段"), ("m2", "最强字段对"),
                      ("m3", "最强三元组"), ("add", "加性Möbius二阶")]:
        m = metrics(sub, col); m["subset"] = label; m["model"] = name
        summ.append(m)
summ = pd.DataFrame(summ)[["subset", "model", "n", "mae", "bias", "spearman", "resid_mean", "resid_p90"]]
summ.to_csv(f"{OUT}/surrogate_metrics.csv", index=False)
print("\n=== 代理保真度（留出宽尺寸集合）===")
print(summ.to_string(index=False))

# ---- 残差按 |S| 分层（高阶贡献是否随规模增长）----
band = pd.cut(df["size"], [4, 8, 16, 32, 44], labels=["5-8", "9-16", "17-32", "33-44"])
df["band"] = band
res_by_size = df[df["size"] >= 5].groupby("band", observed=True).apply(
    lambda s: pd.Series({
        "n": len(s),
        "truth_mean": s["truth"].mean(),
        "resid_m2_mean": (s["truth"] - s["m2"]).mean(),
        "resid_m2_p90": (s["truth"] - s["m2"]).quantile(0.9),
        "resid_m3_mean": (s["truth"] - s["m3"]).mean(),
        "frac_resid_gt_noise": ((s["truth"] - s["m2"]) > 3 * NOISE).mean(),
    })).reset_index()
res_by_size.to_csv(f"{OUT}/residual_by_size.csv", index=False)
print("\n=== 残差(真值 - 最强字段对)按规模分层 ===")
print(res_by_size.to_string(index=False))

print("\n完成，输出见 outputs/")
