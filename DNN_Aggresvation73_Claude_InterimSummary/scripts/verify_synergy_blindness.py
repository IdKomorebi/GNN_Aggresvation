# -*- coding: utf-8 -*-
"""核查 A：不微调（K=0）的 oracle 是否具备"发现二阶协同"的能力。

背景
----
67/68 号建立了二阶协同的重训真值：syn_c(i,j) = v_c({i,j}) - max(v_c({i}), v_c({j}))。
69 号同时记录了 MLP/GNN oracle 在 K∈{0,10,50,200} 步微调下对同一批 (pair, conf)
条目的估计值。69 号 RESULTS 报告的是全局 Spearman（K=0 时 MLP 0.671），
但"全局 Spearman"会被大量真值≈0 的条目撑起来，无法回答一个更尖锐的问题：

    对于我们真正在意的、已被重训认证的【最强】协同，K=0 的 oracle 看得见吗？

本脚本按真值强度分层，并单独追踪真值 top-N 条目在估计中的排名。

数据来源（只读，不重算）
----------------------
DNN_Aggresvation69/outputs/synergy_detail.csv
    列：order, sid, group, arch, K, conf, truth_syn, est_syn
    order=2 即二阶；truth_syn 为重训真值协同，est_syn 为 oracle 估计协同。

输出
----
outputs/q5_synergy_blindness_bystrength.csv   分层统计
outputs/q5_synergy_blindness_toprank.csv      真值 top-N 的估计排名
outputs/q5_synergy_top10_detail.csv           真值 top10 逐条明细
outputs/q5_synergy_blindness.txt              可读报告
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "DNN_Aggresvation69" / "outputs" / "synergy_detail.csv"
OUT = Path(__file__).resolve().parents[1] / "outputs"

# 分层阈值：0.2 以上视为"强协同"（68 号 top pairs 大多在 0.4~0.77）；
# 0.02 以下视为"无协同"（60 号噪声底 3σ≈0.012，取 0.02 更保守）。
STRONG, MID_LO, NULL = 0.20, 0.10, 0.02
DETECT = 0.10  # 判定 oracle "报出了协同" 的阈值


def main() -> None:
    df = pd.read_csv(SRC)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 78)
    log("核查 A：K=0 oracle 对二阶协同的检出能力（分层 + top-N 排名）")
    log("=" * 78)

    rows_layer, rows_rank, top10_detail = [], [], None

    for arch in ["mlp", "gnn"]:
        d2 = df[(df.order == 2) & (df.arch == arch)]
        if d2.empty:
            continue
        log(f"\n---------- arch = {arch}  (n={len(d2[d2.K == d2.K.min()])} 条 pair×conf) ----------")
        for K in sorted(d2.K.unique()):
            s = d2[d2.K == K]
            strong = s[s.truth_syn > STRONG]
            mid = s[(s.truth_syn > MID_LO) & (s.truth_syn <= STRONG)]
            null = s[s.truth_syn <= NULL]
            rho = spearmanr(s.truth_syn, s.est_syn).correlation
            rec_strong = (strong.est_syn > DETECT).mean()
            rec_mid = (mid.est_syn > DETECT).mean()
            fpr = (null.est_syn > DETECT).mean()
            rows_layer.append(dict(
                arch=arch, K=K, n=len(s), spearman=rho,
                n_strong=len(strong), strong_truth_mean=strong.truth_syn.mean(),
                strong_est_mean=strong.est_syn.mean(), strong_recall=rec_strong,
                n_mid=len(mid), mid_est_mean=mid.est_syn.mean(), mid_recall=rec_mid,
                n_null=len(null), null_est_mean=null.est_syn.mean(), false_positive_rate=fpr,
            ))
            log(f"K={K:>3}  rho={rho:.3f} | 强(>{STRONG}) n={len(strong)} "
                f"真均值={strong.truth_syn.mean():.3f} 估均值={strong.est_syn.mean():.3f} "
                f"检出率={100 * rec_strong:.0f}% | 中({MID_LO}~{STRONG}) n={len(mid)} "
                f"估均值={mid.est_syn.mean():.3f} 检出率={100 * rec_mid:.0f}% | "
                f"无(<={NULL}) 估均值={null.est_syn.mean():.4f} 误报率={100 * fpr:.2f}%")

        # top-N 排名追踪：真值最强的 N 条，在 est 的全局排名里排第几
        for K in sorted(d2.K.unique()):
            s = d2[d2.K == K].reset_index(drop=True)
            s = s.assign(rank_true=s.truth_syn.rank(ascending=False),
                         rank_est=s.est_syn.rank(ascending=False))
            for N in [10, 20, 50]:
                top = s.nsmallest(N, "rank_true")
                rows_rank.append(dict(arch=arch, K=K, N=N, n_total=len(s),
                                      median_est_rank=top.rank_est.median(),
                                      worst_est_rank=top.rank_est.max(),
                                      hit_in_est_topN=int((top.rank_est <= N).sum())))
            if K == 0 and arch == "mlp":
                top10_detail = s.nsmallest(10, "rank_true")[
                    ["sid", "conf", "truth_syn", "est_syn", "rank_est"]].copy()
                top10_detail["n_total"] = len(s)

    layer = pd.DataFrame(rows_layer)
    rank = pd.DataFrame(rows_rank)
    layer.to_csv(OUT / "q5_synergy_blindness_bystrength.csv", index=False)
    rank.to_csv(OUT / "q5_synergy_blindness_toprank.csv", index=False)

    log("\n---------- 真值 top-N 条目在【估计】中的排名（越大越说明看不见）----------")
    for r in rank[rank.arch == "mlp"].itertuples():
        log(f"mlp K={r.K:>3} 真值top{r.N:>2}: 估计排名中位数={r.median_est_rank:.0f}/{r.n_total}, "
            f"落入估计top{r.N}的={r.hit_in_est_topN}/{r.N}")

    if top10_detail is not None:
        top10_detail.to_csv(OUT / "q5_synergy_top10_detail.csv", index=False)
        log("\n---------- MLP oracle, K=0：真值 top10 协同的逐条估计 ----------")
        log(top10_detail.to_string(index=False))

    (OUT / "q5_synergy_blindness.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写出 -> {OUT}")


if __name__ == "__main__":
    main()
