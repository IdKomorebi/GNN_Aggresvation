# -*- coding: utf-8 -*-
"""93 号 任务A 分析：90 号 full 字典发现的"强五阶"经 DNN 重训后还剩多少？

判据（预注册）
--------------
· 若强组认证后 syn 普遍塌到对照组水平 ⟹ 90 号那批发现是**结构化查询的假阳性**，
  87→90 的"衰减律被推翻"需要第二次修正；
· 若强组显著高于对照组 ⟹ 90 号结论成立，高阶风险确实被低估；
· 分 artifact / non-artifact 两栏看，才能区分"clamp 伪影"与"字典本身的差分偏差"。

★同时给出一个 90 号没做、也做不到的诊断：
  `maxsub_true > v_true` 的比例——即**四阶子集的真值反而高于五元组自己**。
  结构化查询把这类集合判成强协同，说明差分偏差的方向是系统性的，不是随机噪声。
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    fs = sorted(glob.glob(str(ROOT / "outputs/certify_o5_s*of*.csv")))
    if not fs:
        print("尚无认证结果")
        return
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    d["retained"] = d.syn_true_audit / d.syn_struct.clip(lower=1e-9)
    d["sub_exceeds"] = d.maxsub_true_audit > d.v_true_audit
    print(f"=== 已认证 {len(d)} 个五元组 "
          f"(strong {int((d.group=='strong').sum())} / control {int((d.group=='control').sum())}) ===\n")

    rows = []
    for grp, g in d.groupby("group"):
        for art in ([False, True] if grp == "strong" else [False]):
            gg = g[g.artifact == art] if grp == "strong" else g
            if not len(gg):
                continue
            rows.append(dict(
                组=f"{grp}{'(伪影)' if art else ''}", n=len(gg),
                结构化syn=round(float(gg.syn_struct.mean()), 4),
                认证syn=round(float(gg.syn_true_audit.mean()), 4),
                认证syn最大=round(float(gg.syn_true_audit.max()), 4),
                保留率=round(float(gg.retained.median()), 3),
                过0_2的=int((gg.syn_true_audit > 0.2).sum()),
                过0_1的=int((gg.syn_true_audit > 0.1).sum()),
                子集反超=int(gg.sub_exceeds.sum())))
    summ = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))

    s = d[d.group == "strong"]
    c = d[d.group == "control"]
    print(f"\n=== 强组 vs 对照组（认证后）===")
    if len(s) and len(c):
        u = mannwhitneyu(s.syn_true_audit, c.syn_true_audit, alternative="greater")
        print(f"  强组认证 syn 中位 {s.syn_true_audit.median():.4f}   "
              f"对照组 {c.syn_true_audit.median():.4f}")
        print(f"  Mann-Whitney U 单边 p = {u.pvalue:.4f}  "
              f"⟹ {'仍显著高于对照' if u.pvalue < 0.05 else '★与对照组无显著差异'}")
    if len(s) > 3:
        r = spearmanr(s.syn_struct, s.syn_true_audit)
        print(f"  强组内 结构化syn ↔ 认证syn 的 Spearman = {r.statistic:.4f} (p={r.pvalue:.3f})")

    print(f"\n=== ★子集反超（maxsub_true > v_true，即根本不存在增量）===")
    print(f"  强组 {int(s.sub_exceeds.sum())}/{len(s)} "
          f"({s.sub_exceeds.mean()*100 if len(s) else 0:.0f}%)   "
          f"对照组 {int(c.sub_exceeds.sum())}/{len(c)} "
          f"({c.sub_exceeds.mean()*100 if len(c) else 0:.0f}%)")

    print(f"\n=== 认证后 syn 最大的 8 个（无论来自哪组）===")
    cols = ["S", "group", "artifact", "syn_struct", "syn_true_audit",
            "v_true_audit", "maxsub_true_audit"]
    print(d.nlargest(8, "syn_true_audit")[cols].round(4).to_string(index=False))

    d.to_csv(ROOT / "outputs/certify_merged.csv", index=False)
    summ.to_csv(ROOT / "outputs/certify_summary.csv", index=False)


if __name__ == "__main__":
    main()
