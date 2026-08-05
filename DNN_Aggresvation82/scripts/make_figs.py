# -*- coding: utf-8 -*-
"""82号静态结果图。

图约定：
1) K0散点回答不同训练分布在父集合保真与S1发现之间是否有权衡；
2) K曲线回答专用oracle的起点和微调收敛是否优于uniform。
输出为项目内可复现PNG；蓝/橙为两个专用方案，灰为uniform，另配线型和点型。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "uniform": "#6B7280",
    "parent": "#2563A6",
    "s1": "#C46A22",
}
MARKERS = {"uniform": "o", "parent": "s", "s1": "^"}
LINES = {"uniform": "--", "parent": "-", "s1": "-."}


def k0_tradeoff() -> None:
    data = pd.read_csv(OUT / "k0_scheme_summary.csv")
    data = data[data["mode"] == "single"].copy()
    selection = json.loads(
        (OUT / "k0_selection.json").read_text(encoding="utf-8")
    )
    parent = selection["best_parent_scheme"]
    s1 = selection["best_s1_scheme"]
    reference = {"uniform", "small80", "workload_hard", "logunif"}
    name = {
        "uniform": "uniform",
        "small80": "small80",
        "workload_hard": "workload_hard",
        "logunif": "logunif",
        "triple_only": "100%三元",
        "triple90_pair10": "90%三元+10%二元",
        "triple70_pair20_uniform10": "70%三元+20%二元+10%全尺度",
        "triple50_pair40_uniform10": "50%三元+40%二元+10%全尺度",
        "triple40_pair30_uniform30": "40%三元+30%二元+30%全尺度",
        "local234_20_60_20": "2/3/4局部(20/60/20)",
        "local234_uniform10": "2/3/4局部+10%全尺度",
        "warm_triple70": "uniform暖启动→70%三元",
        "warm_local234": "uniform暖启动→2/3/4局部",
        "triple70_group8": "70%三元（8样本/掩码）",
        "triple70_group32": "70%三元（32样本/掩码）",
        "local234_group8": "2/3/4局部（8样本/掩码）",
        "local234_group32": "2/3/4局部（32样本/掩码）",
    }
    offsets = {
        "local234_group32": (-75, -30),
        "triple90_pair10": (8, 12),
        "local234_group8": (-115, -5),
        "logunif": (8, 8),
    }
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for row in data.itertuples():
        if row.scheme == parent:
            role = "parent"
        elif row.scheme == s1:
            role = "s1"
        else:
            role = "uniform"
        alpha = 1.0 if row.scheme in {parent, s1, "uniform"} else 0.55
        fill = COLORS[role] if row.scheme not in reference else "#9CA3AF"
        ax.scatter(
            row.tune_parent_mae,
            row.tune_top30_f1,
            s=105 if alpha == 1 else 55,
            marker=MARKERS[role],
            color=fill,
            edgecolor="#28323C",
            linewidth=0.7,
            alpha=alpha,
        )
        ax.annotate(
            name.get(row.scheme, row.scheme),
            (row.tune_parent_mae, row.tune_top30_f1),
            xytext=offsets.get(row.scheme, (5, 4)),
            textcoords="offset points",
            fontsize=8.2,
            alpha=max(alpha, 0.75),
        )
    ax.set_title("K=0训练分布对照")
    ax.set_xlabel("调参集三元父集合 MAE（越低越好）")
    ax.set_ylabel("调参集 S1 Top-30% F1（越高越好）")
    ax.text(
        0.01,
        0.01,
        "每个新方案先跑 seed 0；蓝/橙分别为父集合与S1预选方案",
        transform=ax.transAxes,
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(FIG / "k0_strategy_tradeoff.png", dpi=190)
    plt.close(fig)


def kgrid_curves() -> None:
    data = pd.read_csv(OUT / "kgrid_comparison.csv")
    selection = json.loads(
        (OUT / "k0_selection.json").read_text(encoding="utf-8")
    )
    chosen = [
        ("uniform", "single", "uniform", "uniform"),
        (
            selection["best_parent_scheme"],
            selection["best_parent_mode"],
            "parent",
            "父集合预选",
        ),
        (
            selection["best_s1_scheme"],
            selection["best_s1_mode"],
            "s1",
            "S1预选",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    definitions = [
        ("test_parent_mae", "隔离测试：三元父集合 MAE", "MAE"),
        ("test_top30_recall", "隔离测试：S1 Top-30%召回", "召回率"),
        ("extreme_parent_error_mean", "极强协同：父集合平均误差", "估计值−真值"),
        ("extreme_syn_est_mean", "极强协同：估计三阶增量", "估计增量"),
    ]
    for scheme, mode, role, label in chosen:
        group = data[
            data.scheme.eq(scheme) & data["mode"].eq(mode)
        ].sort_values("K")
        for axis, (column, _, _) in zip(axes.flat, definitions):
            axis.plot(
                group.K,
                group[column],
                color=COLORS[role],
                marker=MARKERS[role],
                linestyle=LINES[role],
                linewidth=2,
                label=f"{label}: {scheme}",
            )
    for axis, (_, title, ylabel) in zip(axes.flat, definitions):
        axis.set_title(title)
        axis.set_xlabel("微调更新步数 K")
        axis.set_ylabel(ylabel)
        axis.axhline(0, color="#374151", linewidth=0.8, alpha=0.7)
    truth_mean = data.extreme_syn_true_mean.dropna().iloc[0]
    axes[1, 1].axhline(
        truth_mean,
        color="#374151",
        linewidth=1.2,
        linestyle=":",
        label=f"重训真值均值 {truth_mean:.3f}",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    truth_handle, truth_label = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles + truth_handle[-1:],
        labels + truth_label[-1:],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("通用与三元组专用 oracle 的 K 步微调对照", y=0.985)
    fig.subplots_adjust(top=0.84, hspace=0.34, wspace=0.24)
    fig.savefig(FIG / "kgrid_oracle_comparison.png", dpi=190)
    plt.close(fig)


def main() -> None:
    k0_tradeoff()
    kgrid_curves()


if __name__ == "__main__":
    main()
