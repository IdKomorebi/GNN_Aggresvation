# -*- coding: utf-8 -*-
"""DNN78 修正版主图。只读取 78 号已生成的 CSV。"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnn78-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from runlog import log, section  # noqa: E402

for font in (
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Source Han Sans CN",
    "DejaVu Sans",
):
    if any(font in item.name for item in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [font]
        break
plt.rcParams["axes.unicode_minus"] = False


def f2_figure():
    data = pd.read_csv(OUT / "f2_query_level.csv")
    test = data[
        (data.scope == "test")
        & data.rule.isin(("fixed_k50", "query_maxconf_tuned"))
    ].copy()
    labels = ["固定 K=50", "逐字段对自适应"]
    colors = ["#7f8c8d", "#2e86c1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, test.mean_steps, color=colors)
    axes[0].set_ylabel("平均微调步数")
    axes[0].set_title("实际查询成本：每个字段对只有一个停止 K")
    for idx, value in enumerate(test.mean_steps):
        axes[0].text(idx, value + 1, f"{value:.1f}", ha="center")

    axes[1].bar(labels, test.recall, color=colors)
    # 柱状图从 0 起，避免把 1.2 个百分点的差异视觉放大。
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("强二阶协同召回")
    axes[1].set_title("独立字段对测试集（55 条强协同）")
    for idx, value in enumerate(test.recall):
        axes[1].text(idx, value + 0.004, f"{value:.1%}", ha="center")
    fig.suptitle("DNN78 图1：F-2 从逐 entry 停机修正为逐字段对停机")
    fig.tight_layout()
    fig.savefig(FIG / "f2_query_stop_zh.png", dpi=170)
    plt.close(fig)


def recall_figure():
    current = pd.read_csv(OUT / "syn3_recall_current_pool.csv")
    weighted = pd.read_csv(OUT / "syn3_recall_stratified.csv")
    labels = {
        "k0": "K=0 直接扫描",
        "k25_mixed": "旧 mixed-K（仅作错误对照）",
        "k25_consistent": "同保真 K=25",
    }
    colors = {
        "k0": "#c0392b",
        "k25_mixed": "#95a5a6",
        "k25_consistent": "#2471a3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for estimator, group in current.groupby("estimator"):
        axes[0].plot(
            group.retention * 100,
            group.recall,
            marker="o",
            label=labels[estimator],
            color=colors[estimator],
        )
    axes[0].set_xlabel("最终保留候选比例（%）")
    axes[0].set_ylabel("真强三元组召回")
    axes[0].set_title("现有2000池：同保真微调改善中等预算重排")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    k0 = weighted[weighted.estimator == "k0"]
    axes[1].plot(k0.retention * 100, k0.recall, "o-", color="#c0392b")
    axes[1].axhline(0.95, color="#7f8c8d", ls="--", label="95%召回")
    axes[1].set_xlabel("K=0保留候选比例（%）")
    axes[1].set_ylabel("分层加权召回")
    axes[1].set_title("总体口径：高召回需要保留大部分候选")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("DNN78 图2：三阶筛选召回的同K与抽样修正")
    fig.tight_layout()
    fig.savefig(FIG / "syn3_recall_corrected_zh.png", dpi=170)
    plt.close(fig)


def value_figure():
    pool = pd.read_csv(REPO / "DNN_Aggresvation77/outputs/h2_unbiased_pool.csv")
    pool["indices"] = pool.ix.map(ast.literal_eval)
    k0 = pd.read_csv(REPO / "DNN_Aggresvation77/outputs/triples_syn3_k0.csv")
    k0_map = {
        (row.i, row.j, row.k, row.conf): row.syn3_est_k0 for row in k0.itertuples()
    }
    k25 = pd.read_csv(OUT / "syn3_k25_consistent_entries.csv")
    k25_map = {
        (row.i, row.j, row.k, row.conf): row.syn3_k25_consistent
        for row in k25.itertuples()
    }
    rows = []
    for row in pool.itertuples():
        key = (*row.indices, row.conf)
        if key in k25_map:
            rows.append((row.syn3_true, k0_map[key], k25_map[key]))
    data = pd.DataFrame(rows, columns=["truth", "k0", "k25"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, column, title in (
        (axes[0], "k0", "K=0"),
        (axes[1], "k25", "同保真 K=25"),
    ):
        axis.scatter(data.truth, data[column], s=9, alpha=0.35, color="#2e86c1")
        axis.plot([-0.15, 0.5], [-0.15, 0.5], "k--", lw=1)
        axis.axhline(0.1, color="#d35400", ls=":")
        axis.axvline(0.1, color="#d35400", ls=":")
        rho = spearmanr(data.truth, data[column]).correlation
        mae = np.abs(data[column] - data.truth).mean()
        axis.set_title(f"{title}\nSpearman={rho:.3f}, MAE={mae:.4f}")
        axis.set_xlabel("重训三阶协同真值")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("oracle三阶协同估计")
    fig.suptitle("DNN78 图3：K=25必须同时微调三元组和三个二元子集")
    fig.tight_layout()
    fig.savefig(FIG / "syn3_value_corrected_zh.png", dpi=170)
    plt.close(fig)


def main():
    section("阶段 5：修正版可视化")
    f2_figure()
    recall_figure()
    value_figure()
    log(
        "FIGURES",
        "DONE",
        note="生成 F2 查询级早停、三阶召回、三阶同保真数值三张中文图",
    )
    print("已生成 3 张 DNN78 修正版图")


if __name__ == "__main__":
    main()
