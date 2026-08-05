# -*- coding: utf-8 -*-
"""核查 C：70 号 Apriori 剪枝的 "85%" 存在选择偏差，用无偏随机样本重算。

背景
----
70 号 calibration_and_hierarchy.py 报告："314 条显著三阶协同中，85% 含显著二阶子对；
以'含显著二元子对'为剪枝器，保留 993/2364，捕获真显著三阶 85%，2.4× 加速。"

问题：那 2364 条全部来自 68 号的 `triple_top` 组——即 MLP 估计器在 13244 个三元组里
挑出的 top200 再重训认证的结果。用"被筛选器挑过的池子"去评估"筛选器的漏检率"，
是循环论证（selection bias）。

68 号同时认证了 200 个【随机】三元组（triple_rand 组，2400 条 pair×conf），
那才是对全空间的无偏抽样。本脚本在两个池子上分别重算，并给出无偏口径的
recall / 保留率 / 加速比，附 Wilson 95% 置信区间。

名词
----
recall（召回率）：真正的强三阶协同里，有多少条会被剪枝器保留下来（不被误剪）。
保留率：剪枝器保留的候选占全部候选的比例；加速比 = 1 / 保留率。

数据来源（只读）
--------------
DNN_Aggresvation68/outputs/triples_certified.csv   列：group, fi, fj, fk, conf,
                                                    v_ijk_true, best_pair, syn3_true, syn3_est
DNN_Aggresvation68/outputs/synergy2_perconf.csv    二阶协同重训真值

输出
----
outputs/q6_apriori_selection_bias.csv
outputs/q6_apriori_selection_bias.txt
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S68 = ROOT / "DNN_Aggresvation68" / "outputs"
OUT = Path(__file__).resolve().parents[1] / "outputs"

DELTA = 0.10          # "显著协同" 判定阈值，与 70 号一致
N_TRIPLES_ALL = 13244  # C(44,3)
N_CONF = 12


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间——小样本比例的置信区间（比正态近似可靠）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    syn2 = pd.read_csv(S68 / "synergy2_perconf.csv")
    lk = {(min(r.fi, r.fj), max(r.fi, r.fj), r.conf): r.synergy for r in syn2.itertuples()}
    tri = pd.read_csv(S68 / "triples_certified.csv")

    def best_pair_syn2(r) -> float:
        fs = [r.fi, r.fj, r.fk]
        vals = [lk.get((min(fs[a], fs[b]), max(fs[a], fs[b]), r.conf), np.nan)
                for a, b in [(0, 1), (0, 2), (1, 2)]]
        vals = [v for v in vals if not np.isnan(v)]
        return max(vals) if vals else np.nan

    tri["best_child_syn2"] = [best_pair_syn2(r) for r in tri.itertuples()]

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 78)
    log(f"核查 C：Apriori 剪枝器（'含显著二阶子对 syn2>{DELTA}'）的 recall —— 有偏 vs 无偏")
    log("=" * 78)

    rows = []
    for group, desc in [("triple_top", "估计器挑出的 top200（有偏，70 号原口径）"),
                        ("triple_rand", "随机抽样 200 个三元组（无偏）")]:
        s = tri[(tri.group == group) & tri.best_child_syn2.notna()]
        strong = s[s.syn3_true > DELTA]
        n_keep = int((s.best_child_syn2 > DELTA).sum())
        n_rec = int((strong.best_child_syn2 > DELTA).sum())
        recall = n_rec / len(strong) if len(strong) else float("nan")
        keep_rate = n_keep / len(s)
        lo, hi = wilson(n_rec, len(strong))
        rows.append(dict(group=group, n_entries=len(s), n_strong=len(strong),
                         base_rate=len(strong) / len(s), recall=recall,
                         recall_ci_lo=lo, recall_ci_hi=hi,
                         keep_rate=keep_rate, speedup=1 / max(keep_rate, 1e-9)))
        log(f"\n[{group}] {desc}")
        log(f"  条目数(pair×conf) = {len(s)}")
        log(f"  强三阶协同(syn3_true>{DELTA}) = {len(strong)}  "
            f"（基率 {100 * len(strong) / len(s):.2f}%）")
        log(f"  剪枝器 recall = {n_rec}/{len(strong)} = {recall:.3f}   "
            f"Wilson 95% CI [{lo:.2f}, {hi:.2f}]")
        log(f"  保留率 = {keep_rate:.3f}  →  加速比 ≈ {1 / keep_rate:.1f}×")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q6_apriori_selection_bias.csv", index=False)

    top_n = int(df[df.group == "triple_top"].n_entries.iloc[0])
    log(f"\n选择偏差的量级：triple_top 只占全空间 {top_n}/{N_TRIPLES_ALL * N_CONF} "
        f"= {100 * top_n / (N_TRIPLES_ALL * N_CONF):.3f}% 的 (三元组, conf) 条目，"
        f"且这 {top_n} 条正是被 MLP 估计器判为最强的那批。")
    log("\n结论：")
    log("  1) 70 号的 85% 是【recall】不是 Spearman；0.795 才是 Spearman(估计 vs 认证)。")
    log("  2) 85% 在有偏池子上算得；无偏随机池上 recall 更高（见上表），但 n 太小、CI 很宽。")
    log("  3) 正确写法：报无偏口径的点估计 + CI + 样本量，并配上保留率/加速比；")
    log("     若要收紧 CI，需要把随机三元组认证样本从 200 扩到 2000~3000 个。")

    (OUT / "q6_apriori_selection_bias.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写出 -> {OUT}")


if __name__ == "__main__":
    main()
