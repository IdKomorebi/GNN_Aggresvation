# -*- coding: utf-8 -*-
"""77 号主结果图。

图 1 两级协议召回：L1-only vs L1+L2 在扩容无偏池上的召回（带 Wilson CI），
     以及成本对比。核心结论——第二级不改召回。
图 2 微调改的是绝对值不是排序：K=0 vs K=25 的 ŝyn3 vs 真值散点。
"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R77 = Path(__file__).resolve().parents[1]
for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
          "Source Han Sans CN", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    prot = pd.read_csv(R77 / "outputs/h2_protocol_recall.csv")
    l1 = prot[prot.stage == "L1"].sort_values("n_keep")

    # ---------------- 图 1：召回曲线 ----------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(l1.n_keep, l1.recall,
                yerr=[l1.recall - l1.ci_lo, l1.ci_hi - l1.recall],
                fmt="o-", color="#c0392b", lw=2.2, ms=6, capsize=4,
                label="仅第一级 K=0（成本 2 s）")
    if (prot.stage == "L1+L2").any():
        l2 = prot[prot.stage == "L1+L2"].sort_values("n_keep")
        ax.errorbar(l2.n_keep, l2.recall,
                    yerr=[l2.recall - l2.ci_lo, l2.ci_hi - l2.recall],
                    fmt="s--", color="#2980b9", lw=2, ms=5, capsize=4,
                    label="两级 K=0→K*=25（成本 +225 s）")
    ax.set_xscale("log")
    ax.set_xlabel("保留的候选三元组数（共 13244 个）")
    ax.set_ylabel("真强三阶召回率（n=130，syn3>0.1）")
    ax.set_title("DNN77 图1：两级高阶扫描协议的召回（扩容无偏池，误差棒=Wilson 95% CI）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    ax.annotate("第二级微调没有提升召回：\nK=0 的排序已把真强三元组\n排进 top 30%（召回 85%）",
                xy=(3973, 0.85), xytext=(500, 0.55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    fig.savefig(R77 / "figures/h1_recall_zh.png", dpi=150)

    # ---------------- 图 2：微调改绝对值不改排序 ----------------
    import yaml
    sys.path.insert(0, str(R77.parent / "DNN_Aggresvation69"))
    from src.data_processing import prepare_data
    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R77.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)

    pool = pd.read_csv(R77 / "outputs/h2_unbiased_pool.csv")
    pool["ix"] = pool["ix"].apply(eval)
    s3 = pd.read_csv(R77 / "outputs/triples_syn3_k0.csv")
    k0 = {(r.i, r.j, r.k, r.conf): r.syn3_est_k0 for r in s3.itertuples()}
    d2 = pd.concat([pd.read_csv(p) for p in sorted((R77 / "outputs").glob("triples_kstar_shard*.csv"))])
    low = pd.read_csv(R77 / "outputs/lowfour_k0.csv")
    v2 = low[low["size"] == 2].set_index(["key", "conf"]).est.to_dict()

    def key(*ix):
        return "_".join(f"{x:02d}" for x in sorted(ix))
    k25 = {}
    for r in d2.itertuples():
        p = [v2.get((key(r.i, r.j), r.conf)), v2.get((key(r.i, r.k), r.conf)),
             v2.get((key(r.j, r.k), r.conf))]
        if all(x is not None for x in p):
            k25[(r.i, r.j, r.k, r.conf)] = r.est - max(p)

    rows = []
    for r in pool.itertuples():
        rows.append((r.syn3_true, k0.get((*r.ix, r.conf)), k25.get((*r.ix, r.conf))))
    df = pd.DataFrame(rows, columns=["t", "e0", "e25"]).dropna(subset=["e0", "e25"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, col, lab, K in [(axes[0], "e0", "K=0（免费，第一级）", 0),
                            (axes[1], "e25", "K*=25（第二级微调）", 25)]:
        ax.scatter(df.t, df[col], s=10, alpha=0.4, color="#2980b9")
        ax.plot([-0.05, 0.5], [-0.05, 0.5], "k--", lw=1, alpha=0.6)
        ax.axhline(0.1, color="#e67e22", ls=":", lw=1)
        ax.axvline(0.1, color="#e67e22", ls=":", lw=1)
        from scipy.stats import spearmanr
        rho = spearmanr(df.t, df[col]).correlation
        ax.set_title(f"{lab}\nSpearman={rho:.3f}")
        ax.set_xlabel("三阶协同真值 syn3_true")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("oracle 估计 ŝyn3")
    fig.suptitle("DNN77 图2：微调改善的是绝对值与相关性，第一级排序已足够做筛选", fontsize=13)
    fig.tight_layout()
    fig.savefig(R77 / "figures/h2_finetune_zh.png", dpi=150)
    print("已写出 figures/h1_recall_zh.png、h2_finetune_zh.png")


if __name__ == "__main__":
    main()
