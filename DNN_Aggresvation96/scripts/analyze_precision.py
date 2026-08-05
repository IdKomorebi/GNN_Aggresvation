# -*- coding: utf-8 -*-
"""96 号：高阶发现**准确率**分析（k=6,7,8，并回看 k=3,4,5）。

在 k>=6 上召回率不可测（无法枚举真值），但以下四个量都可测，且能把
"方法报假"与"信号本身消失"区分开：

  A 精确率  precision@τ：自选 top-K 中认证真值 >τ 的比例
  B 校准    结构化估计均值 vs 认证均值（偏差方向与幅度）、逐点 |est-cert|
  C 判别力  top 组 vs 同池随机对照组的 Mann-Whitney 单边 p、AUC
  D 强度    最强认证真值（把认证版衰减律延伸到 8 阶）

★读法：若 A 低但 C 显著且 B 无正偏 ⟹ 是"该阶确实没有强协同"（方法诚实）；
       若 A 低且 B 正偏大 ⟹ 是"方法报假"（方法失效）。两者结论完全相反。
"""
from __future__ import annotations

import ast
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R93, R94 = REPO / "DNN_Aggresvation93", REPO / "DNN_Aggresvation94"


def load_cert(pattern: str, root: Path = ROOT) -> pd.DataFrame | None:
    fs = sorted(glob.glob(str(root / f"outputs/{pattern}")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    d["Stup"] = d["S"].map(lambda s: tuple(ast.literal_eval(s)))
    return d


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def summarize(name: str, d: pd.DataFrame, tau: float = 0.10) -> dict:
    s = d[d.group == "strong"]
    c = d[d.group == "control"] if (d.group == "control").any() else d.iloc[:0]
    row = dict(阶=name, n_top=len(s), n_ctrl=len(c))
    row["估计均值"] = round(float(s.syn_struct.mean()), 4)
    row["认证均值"] = round(float(s.syn_true_audit.mean()), 4)
    row["偏差(估-认)"] = round(float((s.syn_struct - s.syn_true_audit).mean()), 4)
    row["最强认证"] = round(float(s.syn_true_audit.max()), 4)
    row[f"精确率@{tau}"] = f"{int((s.syn_true_audit > tau).sum())}/{len(s)}"
    row["精确率@0.05"] = f"{int((s.syn_true_audit > 0.05).sum())}/{len(s)}"
    if len(c):
        row["对照均值"] = round(float(c.syn_true_audit.mean()), 4)
        row[f"对照@{tau}"] = f"{int((c.syn_true_audit > tau).sum())}/{len(c)}"
        try:
            row["MW_p"] = round(float(mannwhitneyu(
                s.syn_true_audit, c.syn_true_audit, alternative="greater").pvalue), 5)
        except ValueError:
            row["MW_p"] = float("nan")
        row["AUC"] = round(auc(s.syn_true_audit.to_numpy(),
                               c.syn_true_audit.to_numpy()), 3)
    allsets = pd.concat([s, c])
    if len(allsets) > 3 and allsets.syn_struct.nunique() > 1:
        row["ρ(估计,认证)"] = round(float(spearmanr(
            allsets.syn_struct, allsets.syn_true_audit).statistic), 3)
    row["子集反超"] = f"{int((s.maxsub_true_audit > s.v_true_audit).sum())}/{len(s)}"
    return row


def main() -> None:
    pd.set_option("display.width", 250)
    rows = []

    # ---- 回看已有的 k=4,5（93 号，同协议）----
    for k, pat, root in [(4, "certify_o4_l1top_s*.csv", R93),
                         (5, "certify_o5_l1top_s*.csv", R93)]:
        d = load_cert(pat, root)
        if d is None:
            continue
        if "group" not in d:
            d["group"] = "strong"
        rows.append(summarize(f"k={k}（93号）", d))

    # ---- 本号 k=6,7,8 ----
    for k in (6, 7, 8):
        d = load_cert(f"certify_o{k}_hi_s*.csv")
        if d is None:
            print(f"  （k={k} 认证结果尚未就绪）")
            continue
        rows.append(summarize(f"k={k}（本号）", d))

    if not rows:
        print("暂无认证结果")
        return
    tab = pd.DataFrame(rows)
    print("=" * 110)
    print("高阶发现准确率汇总（认证真值口径；tau=0.10）")
    print("=" * 110)
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "outputs/precision_summary.csv", index=False)

    # ---- 扫描侧统计：估计器在各阶报出了多少强候选 ----
    print()
    print("=" * 110)
    print("扫描侧：估计器自身报出的强候选数（未经认证）")
    print("=" * 110)
    srows = []
    for f in sorted(glob.glob(str(ROOT / "outputs/hi_o*_*.json"))):
        st = json.loads(Path(f).read_text(encoding="utf-8"))
        srows.append(dict(阶=st["order"], 模式=st["mode"], 已扫=st["n_scanned"],
                          空间=st["space"],
                          覆盖=f"{st['n_scanned']/st['space']*100:.2f}%",
                          最大估计=round(st["max_syn"], 4),
                          n_gt_010=st["n_gt_010"], n_gt_015=st["n_gt_015"],
                          n_gt_020=st["n_gt_020"], 分钟=round(st["sec"] / 60, 1)))
    if srows:
        sc = pd.DataFrame(srows).groupby(["阶", "模式", "空间"], as_index=False).agg(
            已扫=("已扫", "sum"), 最大估计=("最大估计", "max"),
            n_gt_010=("n_gt_010", "sum"), n_gt_015=("n_gt_015", "sum"),
            n_gt_020=("n_gt_020", "sum"), 分钟=("分钟", "max"))
        sc["覆盖"] = (sc.已扫 / sc.空间 * 100).round(2).astype(str) + "%"
        print(sc.to_string(index=False))
        sc.to_csv(ROOT / "outputs/scan_summary.csv", index=False)

    # ---- 认证版衰减律延伸表 ----
    print()
    print("=" * 110)
    print("认证版衰减律（延伸至高阶）")
    print("=" * 110)
    decay = [dict(阶=3, 最强认证=0.455, 来源="82号重训 / FDR 0.461 互证"),
             dict(阶=4, 最强认证=0.196, 来源="93号 L1 top-20 认证"),
             dict(阶=5, 最强认证=0.154, 来源="93号 L1 top-21 认证")]
    for r in rows:
        if "本号" in r["阶"]:
            decay.append(dict(阶=int(r["阶"].split("=")[1][0]),
                              最强认证=r["最强认证"], 来源="本号认证"))
    dd = pd.DataFrame(decay)
    print(dd.to_string(index=False))
    dd.to_csv(ROOT / "outputs/decay_extended.csv", index=False)


if __name__ == "__main__":
    main()
