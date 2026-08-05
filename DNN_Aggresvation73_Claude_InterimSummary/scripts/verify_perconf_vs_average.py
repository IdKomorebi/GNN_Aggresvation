# -*- coding: utf-8 -*-
"""核查 B：协同必须逐 confidential 统计，不能对 12 个 conf 取平均——量化到底差多少。

背景
----
56 号已把"逐 conf vs 平均"记为方法论教训，68 号 build_synergy2 也确实是逐 (pair, conf)
算的。但项目里一直没有一个"平均口径会漏掉多少"的定量数字。本脚本给出这个数字，
并统计"一个强协同对通常对几个 conf 成立"。

数据来源（只读）
--------------
DNN_Aggresvation68/outputs/synergy2_perconf.csv
    列：i, j, fi, fj, conf, vi, vj, vij, max_single, synergy
    946 个字段对 × 12 个 confidential = 11352 条，全部是重训真值。

输出
----
outputs/q5b_perconf_vs_average.csv     不同阈值下两种口径的命中/漏报
outputs/q5b_nconf_distribution.csv     强协同对同时超阈的 conf 个数分布
outputs/q5b_perconf_vs_average.txt     可读报告
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "DNN_Aggresvation68" / "outputs" / "synergy2_perconf.csv"
OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    p = pd.read_csv(SRC)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    g = p.groupby(["fi", "fj"])
    per_conf_max = g.synergy.max()       # 逐 conf 口径：只要对某一个 conf 强就算强
    avg_over_12 = g.synergy.mean()       # 平均口径：先对 12 个 conf 平均再判定
    n_pairs = len(per_conf_max)

    log("=" * 78)
    log(f"核查 B：逐 conf vs 12-conf 平均（{n_pairs} 个字段对，重训真值）")
    log("=" * 78)

    rows = []
    for th in [0.05, 0.10, 0.20, 0.30]:
        hit_perconf = int((per_conf_max > th).sum())
        hit_avg = int((avg_over_12 > th).sum())
        missed = int(((per_conf_max > th) & (avg_over_12 <= th)).sum())
        rows.append(dict(threshold=th, n_pairs=n_pairs, hit_perconf=hit_perconf,
                         hit_average=hit_avg, missed_by_average=missed,
                         miss_rate=missed / max(hit_perconf, 1)))
        log(f"阈值 {th:.2f}: 逐conf口径命中 {hit_perconf:>3} 对 | "
            f"平均口径命中 {hit_avg:>3} 对 | 平均口径漏报 {missed:>3} 对 "
            f"({100 * missed / max(hit_perconf, 1):.0f}%)")
    pd.DataFrame(rows).to_csv(OUT / "q5b_perconf_vs_average.csv", index=False)

    # 强协同对（max_c syn > 0.2）中，有几个 conf 同时超阈
    TH = 0.20
    strong_idx = set(per_conf_max[per_conf_max > TH].index)
    sub = p[[(r.fi, r.fj) in strong_idx for r in p.itertuples()]]
    n_conf_over = sub.groupby(["fi", "fj"]).synergy.apply(lambda x: int((x > TH).sum()))
    dist = n_conf_over.value_counts().sort_index()
    dist.rename_axis("n_conf_over_threshold").rename("n_pairs").to_frame().to_csv(
        OUT / "q5b_nconf_distribution.csv")

    log(f"\n强协同对（max_c syn>{TH}，共 {len(strong_idx)} 对）中，同时超阈的 conf 个数分布：")
    for k, v in dist.items():
        log(f"  同时对 {k:>2} 个 conf 强 : {v:>3} 对  ({100 * v / len(strong_idx):.0f}%)")
    log(f"→ 其中只对【唯一一个】conf 成立的占 {100 * dist.get(1, 0) / len(strong_idx):.0f}%，"
        f"这类协同在平均口径下被 12 倍稀释，必然漏报。")

    (OUT / "q5b_perconf_vs_average.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写出 -> {OUT}")


if __name__ == "__main__":
    main()
