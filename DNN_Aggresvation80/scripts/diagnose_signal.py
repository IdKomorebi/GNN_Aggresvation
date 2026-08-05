# -*- coding: utf-8 -*-
"""80 号 第 4 步：两个决定性诊断（零 GPU）。

诊断 A：信号上限——用【真实】父泄露 v_ijk_true 当排序键，能否定位最强协同？
   若能 → 问题在"估计不准"（微调/更好 oracle 可救）；
   若不能 → v(ijk) 信号本身不足以区分"协同"与"三个强字段"，必须靠减子集，
            而减子集又依赖父估计准 → 只能靠微调（微调不可替代的证据）。
   用 68 号 397 认证集（有 v_ijk_true, best_pair 真值, syn3_true）。

诊断 B（问题 2）：层级候选生成的覆盖率——用【强二阶协同】扩展生成三阶候选，
   能覆盖多少已知强三阶？若覆盖率高 → 高阶发现可绕过全空间扫描（Apriori 正确用法）；
   顺带量化候选数的塌缩倍数。用 68 号 synergy2 真值 + 三阶真值。
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

R80 = Path(__file__).resolve().parents[1]
R68 = R80.parent / "DNN_Aggresvation68" / "outputs"
sys.path.insert(0, str(R80 / "src"))
from runlog import log  # noqa: E402


def diag_A():
    print("=" * 96)
    print("诊断 A：信号上限——真实 v_ijk_true / 真实 syn3_true 谁能定位强协同")
    print("（68 号 397 认证三元组，per-triple max over conf）")
    print("=" * 96)
    c = pd.read_csv(R68 / "triples_certified.csv")
    # per triple：取真值最强的 conf
    g = c.groupby(["fi", "fj", "fk"])
    tri = g.agg(v_ijk_true=("v_ijk_true", "max"),
                best_pair_true=("best_pair", "max"),
                syn3_true=("syn3_true", "max")).reset_index()
    strong = tri[tri.syn3_true > 0.10]
    print(f"认证三元组 {len(tri)}，其中强三阶（syn3>0.1）{len(strong)}")

    # 分档：各排序键（真值）的 Top-k 召回
    for lo, hi, bn in [(0.10, 0.15, "0.10-0.15"), (0.15, 0.20, "0.15-0.20"), (0.20, 1.01, ">0.20")]:
        band = tri[(tri.syn3_true >= lo) & (tri.syn3_true < hi)]
        if len(band) == 0:
            continue
        # 用真实 v_ijk_true 排序，这些强三元组能排多前？
        tri_sorted_v = tri.sort_values("v_ijk_true", ascending=False).reset_index(drop=True)
        ranks_v = [tri_sorted_v[(tri_sorted_v.fi == r.fi) & (tri_sorted_v.fj == r.fj) &
                                (tri_sorted_v.fk == r.fk)].index[0] for r in band.itertuples()]
        pct_v = np.median(ranks_v) / len(tri) * 100
        print(f"  档 {bn}（n={len(band)}）：用【真实 v_ijk】排序，全局百分位中位 = {pct_v:.0f}%"
              f"（越小越靠前）；真实 v_ijk 中位 = {band.v_ijk_true.median():.3f}")

    # 核心：真实 v_ijk_true 在"强协同"和"非协同高泄露"上能否分开
    hi_leak = tri[tri.v_ijk_true > 0.5]      # 联合泄露很高的三元组
    print(f"\n  联合泄露 v_ijk_true>0.5 的三元组共 {len(hi_leak)}，"
          f"其中真有强三阶协同（syn3>0.1）的占 {(hi_leak.syn3_true>0.10).mean():.0%}")
    print(f"  → 说明 'v(ijk) 高' {'能' if (hi_leak.syn3_true>0.10).mean()>0.6 else '不能'}"
          f"可靠推出'有协同'；{'v(ijk)信号足够' if (hi_leak.syn3_true>0.10).mean()>0.6 else 'v(ijk)信号不足，必须减子集'}")


def diag_B():
    print("\n" + "=" * 96)
    print("诊断 B（问题2）：层级候选生成——强二阶协同扩展能覆盖多少强三阶？")
    print("=" * 96)
    import yaml
    sys.path.insert(0, str(R80.parent / "DNN_Aggresvation69"))
    from src.data_processing import prepare_data
    cfg = yaml.safe_load((R80 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R80.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    n2i = {n: i for i, n in enumerate(di["general"])}
    nG = len(di["general"])

    # 强二阶协同对（68 号真值）
    syn2 = pd.read_csv(R68 / "synergy2_perconf.csv")
    syn2t = syn2.groupby(["fi", "fj"]).synergy.max().reset_index()
    for th2 in [0.05, 0.10, 0.15]:
        strong_pairs = set()
        for r in syn2t[syn2t.synergy > th2].itertuples():
            strong_pairs.add(tuple(sorted((n2i[r.fi], n2i[r.fj]))))
        # 候选三阶 = 每个强二阶对 + 任意第三字段
        cand = set()
        for (a, b) in strong_pairs:
            for c in range(nG):
                if c not in (a, b):
                    cand.add(tuple(sorted((a, b, c))))
        # 已知强三阶（79 号 split）
        sp = pd.read_csv(R80.parent / "DNN_Aggresvation79/outputs/truth_design_split.csv")
        sp["ix"] = sp["indices"].apply(eval)
        strong3 = sp[sp.strong]
        covered = sum(1 for r in strong3.itertuples() if tuple(sorted(r.ix)) in cand)
        # 加权覆盖
        cov_w = strong3[strong3.ix.apply(lambda x: tuple(sorted(x)) in cand)].weight.sum()
        tot_w = strong3.weight.sum()
        print(f"  强二阶阈值 {th2}：强二阶对 {len(strong_pairs)} 个 → 三阶候选 {len(cand)} 个"
              f"（全空间 {nG*(nG-1)*(nG-2)//6}，塌缩 {nG*(nG-1)*(nG-2)//6/max(len(cand),1):.1f}×）")
        print(f"      覆盖已知强三阶：{covered}/{len(strong3)}（加权 {cov_w/tot_w:.1%}）")
    log("DIAG", "DONE", note="信号上限 + 层级生成覆盖率诊断完成")


if __name__ == "__main__":
    diag_A()
    diag_B()
