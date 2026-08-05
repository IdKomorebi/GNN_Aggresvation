# -*- coding: utf-8 -*-
"""DNN70 续：(1) 等距校准 —— 低阶预测排序是否"就是全部故事"；
(2) Apriori 层级性检验 —— 强 k 阶协同是否含强 (k-1) 阶子集（剪枝合法性前提）。"""
import json, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

SRC = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/outputs"
S68 = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation68/outputs"
OUT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation70/outputs"
NOISE = 0.004

df = pd.read_csv(f"{OUT}/surrogate_predictions.csv")

# ========== (1) 留一子集 等距校准 ==========
# 问题：最强字段对预测系统性低估，但若真值是其预测的单调函数，
# 则一维等距映射即可复原绝对值 —— 说明"排序即全部信息"。
# 用留一(按 sid)交叉验证避免自欺。
def loo_isotonic(df, col):
    preds, truths = [], []
    sids = df.sid.unique()
    for held in sids:
        tr = df[df.sid != held]; te = df[df.sid == held]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(tr[col].values, tr["truth"].values)
        preds.extend(iso.predict(te[col].values)); truths.extend(te["truth"].values)
    preds, truths = np.array(preds), np.array(truths)
    e = preds - truths
    return dict(mae=np.abs(e).mean(), bias=e.mean(),
                spearman=spearmanr(preds, truths).correlation)

rows = []
for col, name in [("m1", "最强单字段"), ("m2", "最强字段对"), ("m3", "最强三元组")]:
    for label, sub in [("all_wide", df), ("size>=5", df[df["size"] >= 5])]:
        raw_mae = (sub[col] - sub["truth"]).abs().mean()
        cal = loo_isotonic(sub, col)
        rows.append(dict(subset=label, model=name, raw_mae=raw_mae,
                         cal_mae=cal["mae"], cal_bias=cal["bias"], cal_spearman=cal["spearman"]))
cal_df = pd.DataFrame(rows)
cal_df.to_csv(f"{OUT}/isotonic_calibration.csv", index=False)
print("=== 留一等距校准（rank→value）===")
print(cal_df.to_string(index=False))
print(f"\n参考：噪声底 σ≈0.002-0.004，3σ≈{3*NOISE}")

# ========== (2) Apriori 层级性检验 ==========
# 前提：若强 k 阶协同的成员总是包含强 (k-1) 阶协同/子集，
# 则 (k+1) 阶候选只需在强 k 阶超集中生成（先验式剪枝合法）。
reg = json.load(open(f"{SRC}/subsets.json"))
tl = pd.read_csv(f"{SRC}/truth_long.csv")
truth = {(r.sid, r.conf): r.dnn for r in tl.itertuples()}
CONFS = sorted(tl.conf.unique())

# 二阶协同真值（68 号 build_synergy2 口径：syn2 = vij - max(vi,vj)）
syn2 = pd.read_csv(f"{S68}/synergy2_perconf.csv")  # cols: fi,fj,conf,vi,vj,vij,synergy
# 三阶认证真值（68 号）：syn3_true = v_ijk - best_pair
tri = pd.read_csv(f"{S68}/triples_certified.csv")
tri = tri[tri.group == "triple_top"]  # 只看 top 组（估计器认为强的）

DELTA = 0.10
# 三阶显著协同：syn3_true > DELTA
strong3 = tri[tri.syn3_true > DELTA].copy()
# 对每个强三元组(i,j,k,conf)，看它的三个二元子集是否有 syn2>DELTA（对同一 conf）
syn2_lookup = {(min(r.fi, r.fj), max(r.fi, r.fj), r.conf): r.synergy for r in syn2.itertuples()}

def pair_syn(a, b, c):
    return syn2_lookup.get((min(a, b), max(a, b), c), np.nan)

contains_strong_pair = []
best_child_syn2 = []
for r in strong3.itertuples():
    fs = [r.fi, r.fj, r.fk]
    child = [pair_syn(fs[a], fs[b], r.conf) for a, b in [(0, 1), (0, 2), (1, 2)]]
    child = [x for x in child if not np.isnan(x)]
    if not child:
        continue
    best_child_syn2.append(max(child))
    contains_strong_pair.append(max(child) > DELTA)

frac = np.mean(contains_strong_pair) if contains_strong_pair else float("nan")
print("\n=== Apriori 层级性检验（三阶 → 二阶）===")
print(f"显著三阶协同(syn3_true>{DELTA})记录数: {len(strong3)}, 可查子对的: {len(contains_strong_pair)}")
print(f"其中含显著二阶子对(syn2>{DELTA})的比例: {frac:.3f}")
print(f"强三元组的最强二元子对 syn2 中位数: {np.median(best_child_syn2):.3f}, 均值: {np.mean(best_child_syn2):.3f}")

# 反向：随机三元组里强二元子对能否预示三阶（选择性检查，作为对照）
hier = pd.DataFrame(dict(best_child_syn2=best_child_syn2, syn3=strong3.syn3_true.values[:len(best_child_syn2)]))
hier.to_csv(f"{OUT}/hierarchy_test.csv", index=False)

# 也报告：以"含显著二元子对"作为三阶候选筛选器的覆盖率/命中率
all_top = tri.copy()
all_top["best_pair_syn2"] = [
    max([x for x in [pair_syn(r.fi, r.fj, r.conf), pair_syn(r.fi, r.fk, r.conf), pair_syn(r.fj, r.fk, r.conf)]
         if not np.isnan(x)] + [0.0]) for r in all_top.itertuples()]
# 若只保留"含强二元子对"的三元组，能捕获多少真强三阶？
kept = all_top[all_top.best_pair_syn2 > DELTA]
recall = (kept.syn3_true > DELTA).sum() / max(1, (all_top.syn3_true > DELTA).sum())
print(f"\n以'含显著二元子对'为剪枝器：保留 {len(kept)}/{len(all_top)} 条，"
      f"捕获真显著三阶 {recall:.3f}")
print("完成。")
