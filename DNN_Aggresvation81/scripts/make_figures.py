# -*- coding: utf-8 -*-
"""生成81号关键静态图。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from runlog import Timer, log  # noqa: E402

BLUE = "#2F6B9A"
ORANGE = "#D47A34"
GOLD = "#C6A648"
INK = "#263238"
GREY = "#9AA6A8"


def setup():
    cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(cjk_path)
    cjk_name = font_manager.FontProperties(fname=cjk_path).get_name()
    plt.rcParams.update(
        {
            "font.family": cjk_name,
            "font.sans-serif": [cjk_name],
            "axes.unicode_minus": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.dpi": 130,
        }
    )


def plot_single_heatmaps(single):
    base = single[
        (single.family == "s1_only") & (single.target_budget == 0.30)
    ].set_index("K")
    data = single[
        (single.family == "force_fill")
        & (single.s2_variant == "aligned")
        & (single.target_budget == 0.30)
    ].copy()
    data["delta_recall_pp"] = [
        100 * (row.all_recall - base.loc[row.K, "all_recall"])
        for row in data.itertuples()
    ]
    data["delta_extreme_raw"] = [
        row.all_b_gt20_raw_hit - base.loc[row.K, "all_b_gt20_raw_hit"]
        for row in data.itertuples()
    ]
    ks = sorted(data.K.unique())
    params = sorted(data.parameter.unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for ax, col, title, cmap, fmt in (
        (
            axes[0],
            "delta_recall_pp",
            "固定Top30：S2-aligned救援带来的总体召回变化",
            "RdBu_r",
            "{:+.1f}",
        ),
        (
            axes[1],
            "delta_extreme_raw",
            "固定Top30：最强三阶原始命中数变化（共15个）",
            "RdBu_r",
            "{:+.0f}",
        ),
    ):
        matrix = (
            data.pivot(index="K", columns="parameter", values=col)
            .reindex(index=ks, columns=params)
            .to_numpy()
        )
        vmax = np.nanmax(np.abs(matrix))
        image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    fmt.format(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
        ax.set_xticks(range(len(params)), [f"{100*p:g}%" for p in params])
        ax.set_yticks(range(len(ks)), [f"K{k}" for k in ks])
        ax.set_xlabel("固定总预算中给S2的救援比例")
        ax.set_ylabel("先微调步数")
        ax.set_title(title, fontsize=12)
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle(
        "S2不是越多越好：K5/K10出现有效区，过强融合会挤掉S1候选",
        fontsize=15,
    )
    fig.savefig(FIG / "single_stage_fixed30_heatmaps_zh.png", dpi=180)
    plt.close(fig)


def plot_k10_top60(single):
    q = single[
        (single.K == 10)
        & (single.target_budget == 0.60)
        & (single.family == "rank_fusion")
        & (single.s2_variant == "aligned")
    ].sort_values("parameter")
    baseline = single[
        (single.K == 10)
        & (single.target_budget == 0.60)
        & (single.family == "s1_only")
    ].iloc[0]
    x = 100 * q.parameter.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(x, 100 * q.tune_recall, marker="o", color=BLUE, label="tune")
    axes[0].plot(x, 100 * q.test_recall, marker="s", color=ORANGE, label="test")
    axes[0].axhline(
        100 * baseline.test_recall,
        color=GREY,
        linestyle="--",
        label="纯S1 test",
    )
    axes[0].set(
        xlabel="S2-aligned融合权重",
        ylabel="Top60强三阶召回（%）",
        title="K10宽进阶段：S2提升候选覆盖",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#E5E8EA")

    axes[1].plot(
        x,
        q.all_b_gt20_raw_hit,
        marker="o",
        color=BLUE,
        label="最强档命中",
    )
    axes[1].axhline(
        baseline.all_b_gt20_raw_hit,
        color=GREY,
        linestyle="--",
        label="纯S1",
    )
    axes[1].set(
        xlabel="S2-aligned融合权重",
        ylabel="命中数量（总计15个）",
        title="最强三阶：8/15最高提升到12/15",
        ylim=(0, 15),
    )
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#E5E8EA")
    fig.suptitle("先微调K10再用S1+S2，比K0并集更有信息", fontsize=15)
    fig.savefig(FIG / "k10_top60_fusion_zh.png", dpi=180)
    plt.close(fig)


def plot_protected_lane(protected):
    q = protected[
        (protected.first_k == 10)
        & (protected.first_retention == 0.60)
        & (protected.first_family == "force_fill")
        & (protected.first_s2 == "any")
        & (protected.first_parameter == 0.20)
        & (protected.final_k == 25)
        & (protected.final_retention == 0.30)
    ].sort_values("protect_fraction")
    truth = pd.read_csv(
        ROOT.parent
        / "DNN_Aggresvation79"
        / "outputs"
        / "truth_design_split.csv"
    )
    estimated_total_strong = truth.loc[
        truth.strong.astype(bool), "weight"
    ].sum()
    q = q.copy()
    q["all_precision_design"] = (
        estimated_total_strong
        * q.all_recall
        / round(13244 * 0.30)
    )
    x = 100 * q.protect_fraction.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].plot(x, 100 * q.tune_recall, marker="o", color=BLUE, label="tune")
    axes[0].plot(x, 100 * q.test_recall, marker="s", color=ORANGE, label="test")
    axes[0].axvspan(2, 3, color=GOLD, alpha=0.18, label="稳定区")
    axes[0].set(
        xlabel="最终Top30中保护S2救回项的比例",
        ylabel="强三阶召回（%）",
        title="保护太少收益没传下来，太多则挤掉S1",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#E5E8EA")

    axes[1].plot(
        x,
        100 * q.all_precision_design,
        marker="s",
        color=ORANGE,
        label="总体精度",
    )
    axes[1].axvspan(2, 3, color=GOLD, alpha=0.18)
    axes[1].set(
        xlabel="保护比例",
        ylabel="Top30精度（%）",
        title="2%～3%时精度不降（纵轴局部放大）",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#E5E8EA")

    axes[2].plot(
        x,
        q.all_b_gt20_raw_hit,
        marker="o",
        color=BLUE,
        label="最强档命中",
    )
    axes[2].axvspan(2, 3, color=GOLD, alpha=0.18)
    axes[2].set(
        xlabel="保护比例",
        ylabel="命中数量（总计15个）",
        title="2%～3%命中11/15",
        ylim=(0, 15),
    )
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", color="#E5E8EA")
    fig.suptitle("S2适合做小型保护通道，不适合接管最终全局排名", fontsize=15)
    fig.savefig(FIG / "protected_lane_sweep_zh.png", dpi=180)
    plt.close(fig)


def plot_final(summary):
    base = summary[summary.protocol == "D79_S1_baseline"].iloc[0]
    selected = summary[summary.protocol == "D81_tune_F1_selected"].iloc[0]
    labels = ["tune", "test", "全部加权"]
    base_recall = 100 * np.array(
        [base.tune_recall, base.test_recall, base.all_recall]
    )
    sel_recall = 100 * np.array(
        [selected.tune_recall, selected.test_recall, selected.all_recall]
    )
    base_precision = 100 * np.array(
        [base.tune_precision, base.test_precision, base.all_precision]
    )
    sel_precision = 100 * np.array(
        [selected.tune_precision, selected.test_precision, selected.all_precision]
    )
    x = np.arange(3)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, before, after, title, ylabel in (
        (
            axes[0],
            base_recall,
            sel_recall,
            "Top30召回（纵轴局部放大）",
            "召回率（%）",
        ),
        (
            axes[1],
            base_precision,
            sel_precision,
            "Top30精度（纵轴局部放大）",
            "精度（%）",
        ),
    ):
        ax.bar(
            x - width / 2,
            before,
            width,
            color="#DDE3E5",
            edgecolor=GREY,
            label="79号纯S1",
        )
        ax.bar(
            x + width / 2,
            after,
            width,
            color=BLUE,
            edgecolor="#234E70",
            label="81号S2保护通道",
        )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color="#E5E8EA", zorder=0)
        ax.legend(frameon=False)
        low = min(before.min(), after.min())
        ax.set_ylim(max(0, low - 5), max(before.max(), after.max()) + 3)
        for xpos, value in zip(x - width / 2, before):
            ax.text(xpos, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)
        for xpos, value in zip(x + width / 2, after):
            ax.text(xpos, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)
    fig.suptitle(
        "同为平均19步、最终Top30：S2保护通道小幅提高总体指标",
        fontsize=15,
    )
    fig.savefig(FIG / "final_protocol_comparison_zh.png", dpi=180)
    plt.close(fig)


def main():
    setup()
    log("FIGURES", "START", note="生成参数与最终方案图")
    with Timer() as timer:
        single = pd.read_csv(ROOT / "outputs" / "single_stage_sweep.csv")
        protected = pd.read_csv(ROOT / "outputs" / "protected_lane_sweep.csv")
        summary = pd.read_csv(ROOT / "outputs" / "selected_protocols.csv")
        plot_single_heatmaps(single)
        plot_k10_top60(single)
        plot_protected_lane(protected)
        plot_final(summary)
    log(
        "FIGURES",
        "DONE",
        note="完成4张中文图",
        elapsed_s=timer.elapsed,
    )
    print("完成4张图")


if __name__ == "__main__":
    main()
