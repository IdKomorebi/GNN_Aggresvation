# -*- coding: utf-8 -*-
"""生成 79 号的中文结果图。"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
CHINESE_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(CHINESE_FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def recall_by_k() -> None:
    data = pd.read_csv(OUT / "kgrid_recall_precision.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for retention, group in data[data.retention.isin([0.10, 0.20, 0.30])].groupby(
        "retention"
    ):
        axes[0].plot(group.K, group.recall, marker="o", label=f"保留 {retention:.0%}")
        axes[1].plot(group.K, group.precision, marker="o", label=f"保留 {retention:.0%}")
    axes[0].set(title="微调几步后第一次筛选更可靠？", xlabel="K（mini-batch 更新步）", ylabel="召回率")
    axes[1].set(title="同一候选预算下的精确率", xlabel="K（mini-batch 更新步）", ylabel="精确率")
    for axis in axes:
        axis.legend()
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "kgrid_recall_precision_zh.png", dpi=180)
    plt.close(fig)


def value_metrics() -> None:
    data = pd.read_csv(OUT / "kgrid_triple_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(data.K, data.average_precision, marker="o", color="#2b6cb0")
    axes[0].set(title="发现能力（PR-AUC）", xlabel="K", ylabel="Average Precision")
    axes[1].plot(data.K, data.mae, marker="o", color="#c05621", label="MAE")
    axes[1].plot(data.K, data.bias.abs(), marker="s", color="#805ad5", label="|Bias|")
    axes[1].set(title="三元组最大协同的数值误差", xlabel="K", ylabel="误差")
    axes[1].legend()
    axes[2].plot(data.K, data.threshold_recall, marker="o", label="召回率")
    axes[2].plot(data.K, data.threshold_precision, marker="s", label="精确率")
    axes[2].set(title="直接以估计值 > 0.10 报警", xlabel="K", ylabel="比例")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIG / "kgrid_value_metrics_zh.png", dpi=180)
    plt.close(fig)


def pareto() -> None:
    data = pd.read_csv(OUT / "rescue_rank_metrics.csv")
    data = data[data.retention == 0.30]
    chosen = pd.read_csv(OUT / "final_fidelity_tiers.csv")
    chosen = chosen[
        chosen.tier.isin(
            [
                "old_k0",
                "fast",
                "balanced",
                "recommended_hierarchy",
                "high_assurance",
                "upper_bound",
            ]
        )
    ]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter(
        data.max_k,
        data.test_recall,
        s=12,
        alpha=0.18,
        color="#718096",
        label="其他方案（隔离测试集）",
    )
    colors = {
        "old_k0": "#718096",
        "fast": "#2f855a",
        "balanced": "#2b6cb0",
        "recommended_hierarchy": "#d69e2e",
        "high_assurance": "#805ad5",
        "upper_bound": "#c53030",
    }
    labels = {
        "old_k0": "K0基线",
        "fast": "K5快速档",
        "balanced": "K10均衡档",
        "recommended_hierarchy": "推荐分层（平均19步）",
        "high_assurance": "K25全体上限",
        "upper_bound": "K50参考",
    }
    for row in chosen.itertuples():
        ax.scatter(
            row.avg_updates_continuation,
            row.test_recall,
            s=90,
            marker="X",
            color=colors[row.tier],
            label=labels[row.tier],
        )
    ax.set_title("Top 30%发现池：保真度—召回折中")
    ax.set_xlabel("平均每个三元组的更新步数（保存状态后续训）")
    ax.set_ylabel("强三阶召回率")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    fig.savefig(FIG / "fidelity_tiers_top30_zh.png", dpi=180)
    plt.close(fig)


def strength_heatmap() -> None:
    data = pd.read_csv(OUT / "recall_by_truth_strength.csv")
    strong = data[data.strength_bin != "非强"].copy()
    order = ["0.10–0.12", "0.12–0.15", "0.15–0.20", ">0.20"]
    table = strong.pivot(index="strength_bin", columns="K", values="recall_top30").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(table.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), labels=table.columns)
    ax.set_yticks(np.arange(len(table.index)), labels=table.index)
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            value = table.iloc[row, column]
            ax.text(column, row, f"{value:.1%}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="召回率")
    ax.set(title="Top 30%召回：哪些强度区间被少步微调救回？", xlabel="K", ylabel="真实三阶协同")
    fig.tight_layout()
    fig.savefig(FIG / "recall_strength_heatmap_zh.png", dpi=180)
    plt.close(fig)


def main() -> None:
    recall_by_k()
    value_metrics()
    pareto()
    strength_heatmap()


if __name__ == "__main__":
    main()
