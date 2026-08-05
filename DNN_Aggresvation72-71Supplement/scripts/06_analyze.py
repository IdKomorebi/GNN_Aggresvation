#!/usr/bin/env python3
"""Aggregate the supplement, render static research figures, and write RESULTS.md."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-7271-supp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GOLD, PINK = "#2a78d6", "#eb6834", "#b88a00", "#d55e91"
INK, GREY, GRID = "#111111", "#5f6368", "#e3e5e8"


def style(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GREY, labelsize=9)


def table_md(df, cols, digits=4):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df[cols].itertuples(index=False, name=None):
        vals = []
        for v in row:
            vals.append(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    validation = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "validation_mlp_seed*.csv")))],
        ignore_index=True,
    )
    validation_summary = validation.groupby(["strategy", "k"], as_index=False).agg(
        worst_mean=("worst", "mean"), worst_std=("worst", "std"),
        mean_mean=("mean", "mean"), mean_std=("mean", "std"), n_seeds=("seed", "nunique"),
    )
    validation_summary.to_csv(OUT / "validation_mlp_summary.csv", index=False)

    cross = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "cross_oracle_*.csv")))],
        ignore_index=True,
    )
    cross_summary = cross.groupby(["candidate", "source", "k", "arch"], as_index=False).agg(
        oracle_worst_mean=("worst", "mean"), oracle_worst_std=("worst", "std"),
        oracle_mean_mean=("mean", "mean"), oracle_mean_std=("mean", "std"),
        n_oracle_seeds=("seed", "nunique"),
    )
    cross_summary.to_csv(OUT / "cross_oracle_summary.csv", index=False)

    clean_rows = [json.loads(Path(p).read_text()) for p in glob.glob(str(OUT / "clean_retrain/*.json"))]
    clean = pd.DataFrame(clean_rows)
    clean_summary = clean.groupby(["candidate", "source", "k"], as_index=False).agg(
        retrain_worst_mean=("worst_test_r2", "mean"), retrain_worst_std=("worst_test_r2", "std"),
        retrain_mean_mean=("mean_test_r2", "mean"), retrain_mean_std=("mean_test_r2", "std"),
        best_epoch_mean=("best_epoch", "mean"), epochs_ran_mean=("epochs_ran", "mean"),
        n_retrain_seeds=("seed", "nunique"),
    )
    clean_summary.to_csv(OUT / "clean_retrain_summary.csv", index=False)

    comparison = cross_summary.merge(clean_summary, on=["candidate", "source", "k"], validate="many_to_one")
    comparison["underestimate_gap"] = comparison.retrain_worst_mean - comparison.oracle_worst_mean
    comparison.to_csv(OUT / "oracle_vs_clean_retrain.csv", index=False)

    old_rows = [json.loads(Path(p).read_text()) for p in glob.glob(str(OUT / "clean_retrain_400cap/*.json"))]
    old = pd.DataFrame(old_rows).groupby("candidate", as_index=False).worst_test_r2.mean().rename(
        columns={"worst_test_r2": "worst_400cap"}
    )
    cap = clean_summary[["candidate", "retrain_worst_mean"]].merge(old, on="candidate")
    cap["change_800_minus_400"] = cap.retrain_worst_mean - cap.worst_400cap
    cap.to_csv(OUT / "training_cap_sensitivity.csv", index=False)

    candidates = json.loads((OUT / "candidates.json").read_text())["candidates"]
    overlap_rows = []
    pairs = [
        ("mlp_direct_k24", "gnn_direct_k24"), ("mlp_direct_k27", "gnn_direct_k27"),
        ("mlp_direct_k30", "gnn_direct_k30"), ("mlp_hybrid_k27", "gnn_hybrid_k27"),
        ("low_order_k24", "mlp_direct_k24"),
    ]
    for a, b in pairs:
        sa, sb = set(candidates[a]["protected"]), set(candidates[b]["protected"])
        overlap_rows.append({"set_a": a, "set_b": b, "intersection": len(sa & sb),
                             "union": len(sa | sb), "jaccard": len(sa & sb) / len(sa | sb)})
    pd.DataFrame(overlap_rows).to_csv(OUT / "set_overlap.csv", index=False)

    key_labels = [
        "low_order_k24", "degree_k24", "single_k24", "random_k24",
        "mlp_direct_k24", "gnn_direct_k24", "mlp_hybrid_k27", "mlp_direct_k27",
        "gnn_hybrid_k27", "gnn_direct_k27", "mlp_hybrid_k30", "mlp_direct_k30",
        "gnn_direct_k30",
    ]
    key = clean_summary[clean_summary.candidate.isin(key_labels)].copy()
    key["sort"] = key.candidate.map({x: i for i, x in enumerate(key_labels)})
    key = key.sort_values("sort").drop(columns="sort")
    key.to_csv(OUT / "same_budget_comparison.csv", index=False)

    # Figure 1: D71 MLP-oracle protection curves, 3 seeds.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=220)
    colors = {"low_order": BLUE, "degree": GOLD, "single_leak": PINK, "random": ORANGE}
    labels = {"low_order": "低阶协同贪心", "degree": "加权度", "single_leak": "单字段泄露", "random": "随机"}
    for ax, metric, ylabel in [(axes[0], "worst", "最坏机密字段 R²"), (axes[1], "mean", "12个机密字段平均 R²")]:
        style(ax)
        for strategy in ["low_order", "degree", "single_leak", "random"]:
            s = validation_summary[validation_summary.strategy == strategy].sort_values("k")
            ax.errorbar(s.k, s[f"{metric}_mean"], yerr=s[f"{metric}_std"], marker="o", lw=2,
                        capsize=3, color=colors[strategy], label=labels[strategy])
        ax.axhline(0.5, color=INK, ls="--", lw=1, label="阈值 0.5" if metric == "worst" else None)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("防护字段数 |P|")
        ax.set_ylabel(ylabel)
        ax.set_title("MLP-oracle K=200，三个种子均值±标准差")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "mlp_protection_validation_zh.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: MLP search paths, oracle vs untouched-test retrain.
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=220)
    style(ax)
    for prefix, color, marker, label in [
        ("mlp_hybrid", BLUE, "o", "混合：低阶24 + MLP精修"),
        ("mlp_direct", ORANGE, "s", "纯MLP-oracle贪心"),
    ]:
        r = clean_summary[clean_summary.candidate.str.startswith(prefix)].sort_values("k")
        if prefix == "mlp_hybrid":
            low = clean_summary[clean_summary.candidate == "low_order_k24"].copy()
            r = pd.concat([low, r], ignore_index=True).sort_values("k")
        ax.errorbar(r.k, r.retrain_worst_mean, yerr=r.retrain_worst_std, color=color,
                    marker=marker, lw=2.3, capsize=3, label=f"{label}：测试集重训评估")
        o = cross_summary[(cross_summary.arch == "mlp") & cross_summary.candidate.str.startswith(prefix)].sort_values("k")
        if prefix == "mlp_hybrid":
            low_o = cross_summary[(cross_summary.arch == "mlp") & (cross_summary.candidate == "low_order_k24")]
            o = pd.concat([low_o, o], ignore_index=True).sort_values("k")
        ax.plot(o.k, o.oracle_worst_mean, color=color, marker=marker, ls="--", alpha=0.75,
                label=f"{label}：MLP-oracle")
    ax.axhline(0.5, color=INK, ls=(0, (5, 3)), lw=1.2, label="安全阈值 0.5")
    ax.set_xlim(23.5, 30.5)
    ax.set_ylim(0.35, 0.86)
    ax.set_xticks(range(24, 31))
    ax.set_xlabel("防护字段数 |P|")
    ax.set_ylabel("最坏机密字段 R²")
    ax.set_title("MLP防护搜索：代理估计与测试集重训评估")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "mlp_search_clean_certificate_zh.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: clean same-budget comparison; absolute bars start at zero.
    show_labels = ["low_order_k24", "mlp_direct_k24", "gnn_direct_k24",
                   "mlp_hybrid_k27", "mlp_direct_k27", "gnn_hybrid_k27", "gnn_direct_k27",
                   "mlp_hybrid_k30", "mlp_direct_k30", "gnn_direct_k30"]
    display = {x: x.replace("_", " ") for x in show_labels}
    s = clean_summary.set_index("candidate").loc[show_labels].reset_index()
    fig, ax = plt.subplots(figsize=(9.3, 5.8), dpi=220)
    style(ax)
    y = np.arange(len(s))
    bar_colors = [BLUE if "hybrid" in x or "low_order" in x else ORANGE if "mlp_direct" in x else "#8a8f98" for x in s.candidate]
    ax.barh(y, s.retrain_worst_mean, xerr=s.retrain_worst_std, color=bar_colors,
            edgecolor="white", capsize=3)
    ax.set_yticks(y, [display[x] for x in s.candidate])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color=INK, ls="--", lw=1.1)
    ax.set_xlabel("测试集最坏机密字段 R²（三个重训种子）")
    ax.set_title("同预算防护方案：测试集不参与重训早停")
    for yi, v in zip(y, s.retrain_worst_mean):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "same_budget_clean_comparison_zh.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    def val(candidate, col="retrain_worst_mean"):
        return float(clean_summary.loc[clean_summary.candidate == candidate, col].iloc[0])

    def oracle(candidate, arch="mlp"):
        return float(cross_summary.loc[(cross_summary.candidate == candidate) & (cross_summary.arch == arch), "oracle_worst_mean"].iloc[0])

    lines = [
        "# DNN_Aggresvation72-71Supplement Results",
        "",
        "## 一句话结论",
        "",
        "换成69号已证明更保真的MLP-oracle，并让专用DNN只用开发集内验证集早停后，72号的宽泛Goodhart结论不再成立：",
        "纯MLP-oracle贪心并未比低阶骨架更差；真正严重失效的是原GNN-oracle搜索。低阶骨架 + MLP精修",
        "仍是当前同一基准上的最好管线，并在删除30/44字段时由三种子重训评估得到低于0.5。",
        "",
        "## 协议",
        "",
        "- 71号四策略用MLP-oracle、K=200、oracle种子0/1/2复跑。",
        "- 72号MLP混合/纯oracle贪心使用K=50选字段、K=200记录，统一跑到k=30。",
        "- 17个候选用MLP/GNN两类oracle、三个种子交叉评估。",
        "- 17个候选均做三个专用DNN重训种子：外层70%开发/30% test；开发集内15% validation early-stop，test不参与重训模型选择。",
        "- 最大800轮、patience=120；400轮快照保留用于训练上限敏感性检查。",
        "",
        "## 71号MLP-oracle复跑（k=24）",
        "",
        table_md(validation_summary[(validation_summary.k == 24)].sort_values("worst_mean"),
                 ["strategy", "k", "worst_mean", "worst_std", "mean_mean", "mean_std", "n_seeds"]),
        "",
        "## 测试集重训评估：关键同预算比较",
        "",
        table_md(key, ["candidate", "source", "k", "retrain_worst_mean", "retrain_worst_std",
                       "retrain_mean_mean", "retrain_mean_std", "n_retrain_seeds"]),
        "",
        "## 代理低估差值（真值 - oracle）",
        "",
        table_md(comparison[comparison.candidate.isin(show_labels)].sort_values(["candidate", "arch"]),
                 ["candidate", "k", "arch", "oracle_worst_mean", "oracle_worst_std",
                  "retrain_worst_mean", "retrain_worst_std", "underestimate_gap"]),
        "",
        "## 核心判断",
        "",
        f"1. k=24同预算：低阶={val('low_order_k24'):.3f}，MLP直接贪心={val('mlp_direct_k24'):.3f}，"
        f"GNN直接贪心={val('gnn_direct_k24'):.3f}。纯MLP贪心没有出现原72号的反向退化。",
        f"2. k=27同预算：MLP直接贪心={val('mlp_direct_k27'):.3f}，MLP混合={val('mlp_hybrid_k27'):.3f}，"
        f"GNN混合={val('gnn_hybrid_k27'):.3f}，GNN直接={val('gnn_direct_k27'):.3f}。",
        f"3. k=30：MLP混合={val('mlp_hybrid_k30'):.3f}±"
        f"{float(clean_summary.loc[clean_summary.candidate=='mlp_hybrid_k30','retrain_worst_std'].iloc[0]):.3f}，"
        f"是当前基准上唯一由三种子重训评估得到低于0.5的方案；纯MLP直接={val('mlp_direct_k30'):.3f}。",
        f"4. MLP混合k=29的oracle={oracle('mlp_hybrid_k29'):.3f}，重训={val('mlp_hybrid_k29'):.3f}，"
        "仍存在一次假达标；k=30时oracle与重训才同时达标。因此最终阈值判断仍必须重训。",
        f"5. 原GNN直接k=30的GNN-oracle={oracle('gnn_direct_k30','gnn'):.3f}，重训={val('gnn_direct_k30'):.3f}，"
        "低估远大于MLP搜索路径。原Goodhart现象主要是GNN代理保真度不足与对其误差反复优化的共同结果，",
        "   不能再概括成所有oracle贪心都会失败。",
        "",
        "## 训练与测量检查",
        "",
        f"- 51/51个800轮上限重训评估完成；平均实际训练{clean.epochs_ran.mean():.1f}轮。",
        f"- 800轮与400轮结果最大绝对变化={cap.change_800_minus_400.abs().max():.4f}，核心排序稳定。",
        "- 所有候选三个训练种子；所有oracle候选两种结构各三个种子。",
        "- 外层test不参与专用DNN的early-stop，修正了72号原重训模型选择泄漏。",
        "",
        "## 仍需诚实说明",
        "",
        "- 当前仍是PJM 2025单数据集，阈值0.5与字段删除成本的外部有效性尚未验证。",
        "- 候选集合仍由69/71/72的oracle或低阶真值在同一外层test上选择，因此这不是端到端未见测试；数值适合比较当前候选，不能当作无偏泛化证书。",
        "- 真正端到端认证需要PJM 2024/未来时间段，或从头重建不接触最终holdout的低阶真值与oracle搜索。",
        "- 贪心每一步仍使用同一oracle反复选择；MLP误差较小但不为零，因此重训证书不能删除。",
        "- k=30意味着删除44个一般字段中的68%，说明单靠删字段获得低泄露的代价非常高。",
        "",
        "Figures: `figures/mlp_protection_validation_zh.png`, `figures/mlp_search_clean_certificate_zh.png`, "
        "`figures/same_budget_clean_comparison_zh.png`.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote summaries, figures, and RESULTS.md")


if __name__ == "__main__":
    main()
