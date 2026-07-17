#!/usr/bin/env python3
"""Generate the figures used by the Advanced Graph Theory course paper.

All numeric panels are rebuilt from DNN_Aggresvation60/67/68/69 artifacts.
The framework panel is a vector-style schematic drawn with matplotlib so the
paper directory remains self-contained and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parents[2]
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK_FONT).exists():
    font_manager.fontManager.addfont(_CJK_FONT)
    _CJK_FAMILY = font_manager.FontProperties(fname=_CJK_FONT).get_name()
else:
    _CJK_FAMILY = "Droid Sans Fallback"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [_CJK_FAMILY, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

BLUE = "#3B6FB6"
CYAN = "#4FA3A5"
RED = "#D85852"
ORANGE = "#E6903A"
PURPLE = "#8B6BB1"
YELLOW = "#F2C14E"
GRAY = "#5C6670"
LIGHT = "#F4F7FA"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, width, height, text, fc, ec="#334155", fontsize=11, lw=1.4):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, linespacing=1.35)
    return patch


def arrow(ax, start, end, color=GRAY, rad=0.0, lw=1.6, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_framework() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Panel backgrounds and titles.
    for x0, width in [(0.1, 4.35), (4.7, 6.0), (10.95, 4.4)]:
        ax.add_patch(FancyBboxPatch((x0, 0.18), width, 5.75,
                                   boxstyle="round,pad=0.02,rounding_size=0.08",
                                   facecolor="#FCFDFE", edgecolor="#CBD5E1", linewidth=1.2))
    ax.text(0.35, 5.62, "(a) 字段推断图", weight="bold", fontsize=14)
    ax.text(4.95, 5.62, "(b) 任意集合敏感度 Oracle", weight="bold", fontsize=14)
    ax.text(11.2, 5.62, "(c) 协同图与高阶结构", weight="bold", fontsize=14)

    # (a) Bipartite inference graph.
    general = [(1.0, 4.45), (2.1, 4.72), (3.2, 4.35),
               (0.85, 2.92), (2.15, 3.15), (3.35, 2.85)]
    confidential = [(1.35, 1.2), (3.0, 1.2)]
    for i, j, w in [(0, 1, 1.4), (1, 2, 2.2), (0, 4, 1.1), (1, 4, 2.0),
                    (2, 5, 1.5), (3, 4, 1.7), (4, 5, 1.2), (1, 3, 0.9)]:
        (x1, y1), (x2, y2) = general[i], general[j]
        ax.plot([x1, x2], [y1, y2], color="#9FB3C8", lw=w, zorder=1)
    for gi, ci, w in [(0, 0, 1.3), (1, 0, 2.5), (3, 0, 1.0),
                      (2, 1, 1.1), (4, 1, 2.3), (5, 1, 1.8)]:
        arrow(ax, general[gi], confidential[ci], color="#8C9BAB", lw=w)
    labels = ["负荷", "风电", "风电占比", "电价", "备用需求", "核电"]
    for (x, y), label in zip(general, labels):
        ax.add_patch(Circle((x, y), 0.34, facecolor=BLUE, edgecolor="white", lw=1.5, zorder=3))
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=9, zorder=4)
    for (x, y), label in zip(confidential, ["总发电", "净交换"]):
        ax.add_patch(Circle((x, y), 0.39, facecolor=RED, edgecolor="white", lw=1.5, zorder=3))
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=9, zorder=4)
    ax.text(2.2, 0.45,
            r"$G=(V_g\cup V_c,E,W)$：蓝色为可见字段，红色为机密字段",
            ha="center", color="#334155", fontsize=10.5)

    # (b) Scoring and certification pipeline.
    box(ax, (5.0, 3.92), 1.45, 0.95, "字段集合 $S$\n掩码 $m_S$", "#E8F0FB")
    box(ax, (6.95, 4.18), 1.75, 1.08, "专用攻击者\nDNN / GCN\n逐集合重训", "#FCE9E7", fontsize=10.5)
    box(ax, (6.95, 2.72), 1.75, 1.08, "通用 Oracle\n$[x\odot m,m]$\n一次前向扫描", "#E7F5F4", fontsize=10.5)
    box(ax, (9.05, 3.45), 1.28, 1.18,
        "集合敏感度\n$\widehat v_c(S)$\n$=[R_c^2]_+$", "#FFF3C4", fontsize=11.5)
    arrow(ax, (6.45, 4.40), (6.95, 4.68))
    arrow(ax, (6.45, 4.28), (6.95, 3.26))
    arrow(ax, (8.70, 4.68), (9.05, 4.15))
    arrow(ax, (8.70, 3.25), (9.05, 3.85))
    box(ax, (5.15, 0.95), 4.95, 0.88,
        "全空间扫描  →  少步微调  →  候选集合重训认证",
        "#F7F1FB", ec="#9A7BB8", fontsize=11)
    arrow(ax, (9.68, 3.42), (8.63, 1.86), color=PURPLE, rad=0.08)
    ax.text(7.68, 2.16, "精度—成本分层", color=PURPLE, ha="center", fontsize=10)
    ax.text(7.72, 0.46,
            r"理论目标：$v_c(S)=\sup_{f\in\mathcal{A}}[R_c^2(f;S)]_+$",
            ha="center", fontsize=11, color="#334155")

    # (c) Pairwise graph and a third-order hyperedge.
    pair_nodes = {"风电MW": (11.55, 4.32), "风电占比": (13.02, 4.67),
                  "水电MW": (14.45, 4.20), "水电占比": (13.85, 3.05),
                  "光伏MW": (12.25, 2.86), "光伏占比": (11.42, 2.02)}
    pair_edges = [("风电MW", "风电占比", 0.765),
                  ("水电MW", "水电占比", 0.665),
                  ("光伏MW", "光伏占比", 0.596)]
    for a, b, value in pair_edges:
        p1, p2 = pair_nodes[a], pair_nodes[b]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ORANGE,
                lw=1.5 + 5 * value, alpha=0.78, solid_capstyle="round")
        xm, ym = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(xm, ym + 0.13, f"{value:.3f}", color="#9A4E15",
                ha="center", fontsize=9, weight="bold")
    for label, (x, y) in pair_nodes.items():
        ax.add_patch(Circle((x, y), 0.37, facecolor=CYAN, edgecolor="white", lw=1.4, zorder=3))
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=8.7, zorder=4)
    ax.text(13.2, 5.13, r"二阶边权：$\Delta_c(i,j)$", ha="center", color="#334155")

    tri = np.array([[12.03, 1.12], [13.18, 1.48], [14.15, 1.05]])
    ax.add_patch(Polygon(tri, closed=True, facecolor="#EFE8F7", edgecolor=PURPLE,
                         lw=2.2, alpha=0.95))
    for (x, y), label in zip(tri, ["燃气MW", "燃气占比", "负荷预测"]):
        ax.add_patch(Circle((x, y), 0.32, facecolor=PURPLE, edgecolor="white", lw=1.2, zorder=3))
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=8.2, zorder=4)
    ax.text(13.10, 0.58, r"三阶超边：$\Delta_c^{(3)}=0.455$",
            ha="center", color="#5E477F", weight="bold", fontsize=10.5)

    save(fig, "framework.png")


def band_label(size: int) -> str:
    for lo, hi in [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]:
        if lo <= size <= hi:
            return f"{lo}--{hi}"
    raise ValueError(size)


def draw_attacker_scale() -> None:
    data = pd.read_csv(REPO / "DNN_Aggresvation60/outputs/random_eval.csv")
    data["band"] = data["size"].map(band_label)
    order = ["1--4", "5--8", "9--16", "17--32", "33--44"]
    summary = data.groupby("band").agg(
        dnn=("dnn", "mean"), gcn=("gcn", "mean"), best=("v_best", "mean")
    ).loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(order))
    axes[0].plot(x, summary["dnn"], "o-", lw=2.2, color=ORANGE, label="专用 DNN")
    axes[0].plot(x, summary["gcn"], "s-", lw=2.2, color=BLUE, label="专用 GCN")
    axes[0].plot(x, summary["best"], "D--", lw=1.7, color=GRAY, label="两结构逐目标上包络")
    axes[0].set_xticks(x, order)
    axes[0].set_xlabel("可见字段集合规模 $|S|$")
    axes[0].set_ylabel("12 个机密字段的平均 $R^2$")
    axes[0].set_ylim(0.15, 0.92)
    axes[0].set_title("(a) 集合越大，可达到的推断能力越高")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    delta = (summary["gcn"] - summary["dnn"]).to_numpy()
    colors = [BLUE if v > 0 else ORANGE for v in delta]
    bars = axes[1].bar(x, delta, color=colors, width=0.68)
    axes[1].axhline(0, color="#334155", lw=1)
    axes[1].axhline(0.02, color="#7C8794", lw=1, ls="--", label="强差异参考线 0.02")
    axes[1].axhline(-0.02, color="#7C8794", lw=1, ls="--")
    axes[1].set_xticks(x, order)
    axes[1].set_xlabel("可见字段集合规模 $|S|$")
    axes[1].set_ylabel("GCN $R^2$ $-$ DNN $R^2$")
    axes[1].set_title("(b) 图结构优势集中在中等规模集合")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    for rect, value in zip(bars, delta):
        axes[1].text(rect.get_x() + rect.get_width() / 2,
                     value + (0.0018 if value >= 0 else -0.0022),
                     f"{value:+.4f}", ha="center",
                     va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    save(fig, "attacker_scale.png")


def draw_oracle_fidelity() -> None:
    fidelity = pd.read_csv(REPO / "DNN_Aggresvation69/outputs/oracle_fidelity.csv")
    fidelity = fidelity[(fidelity["reference"] == "dnn_full") &
                        (fidelity["cohort"] == "wide_all")]
    synergy = pd.read_csv(REPO / "DNN_Aggresvation69/outputs/synergy_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.35))
    for arch, color, marker in [("mlp", ORANGE, "o"), ("gnn", BLUE, "s")]:
        d = fidelity[fidelity["arch"] == arch].sort_values("K")
        axes[0].plot(d["K"], d["mae_per_conf"], marker=marker, color=color,
                     lw=2.2, label=arch.upper())
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xticks([0, 10, 50, 200], ["0", "10", "50", "200"])
    axes[0].set_xlabel("单集合微调步数 $K$")
    axes[0].set_ylabel("逐目标 MAE（相对专用重训）")
    axes[0].set_title("(a) 通用 Oracle 的绝对保真度")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    style = {2: "-", 3: "--"}
    marker = {2: "o", 3: "^"}
    for order in [2, 3]:
        for arch, color in [("mlp", ORANGE), ("gnn", BLUE)]:
            d = synergy[(synergy["order"] == order) & (synergy["arch"] == arch)].sort_values("K")
            axes[1].plot(d["K"], d["spearman"], ls=style[order], marker=marker[order],
                         color=color, lw=2.0, label=f"{arch.upper()} / {order}阶")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xticks([0, 10, 50, 200], ["0", "10", "50", "200"])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("单集合微调步数 $K$")
    axes[1].set_ylabel("与重训协同值的 Spearman 相关")
    axes[1].set_title("(b) 二阶与三阶协同排序保真度")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, "oracle_fidelity.png")


NAME_MAP = {
    "gen_fuel_wind_mw": "风电功率",
    "gen_fuel_wind_pct": "风电占比",
    "gen_fuel_hydro_mw": "水电功率",
    "gen_fuel_hydro_pct": "水电占比",
    "gen_fuel_solar_mw": "光伏功率",
    "gen_fuel_solar_pct": "光伏占比",
    "gen_fuel_multiple_fuels_mw": "多燃料功率",
    "gen_fuel_multiple_fuels_pct": "多燃料占比",
    "gen_fuel_other_renewables_mw": "其他可再生功率",
    "gen_fuel_other_renewables_pct": "其他可再生占比",
    "gen_fuel_nuclear_mw": "核电功率",
    "system_energy_price_da": "日前系统电价",
    "da_as_as_req_mw_primary_reserve": "一次备用需求",
    "da_as_nsr_mw_primary_reserve": "一次备用非同步量",
}


def draw_synergy_graph() -> None:
    pairs = pd.read_csv(REPO / "DNN_Aggresvation68/outputs/synergy2_top_pairs.csv")
    # A field pair can leak several targets. Keep the strongest target per pair.
    pairs["key"] = pairs.apply(lambda r: tuple(sorted((r["fi"], r["fj"]))), axis=1)
    # The clearest physical family is fuel output + fuel share. Restricting the
    # visualization to that family avoids turning a small course-paper figure
    # into an unreadable hairball while retaining the exact certified weights.
    fuel_pair = pairs.apply(
        lambda r: r["fi"].endswith("_mw") and
        r["fj"] == r["fi"][:-3] + "_pct", axis=1
    )
    strongest = (pairs[fuel_pair].sort_values("synergy", ascending=False)
                  .drop_duplicates("key").head(5))

    graph = nx.Graph()
    for _, row in strongest.iterrows():
        a, b = NAME_MAP.get(row["fi"], row["fi"]), NAME_MAP.get(row["fj"], row["fj"])
        graph.add_edge(a, b, weight=float(row["synergy"]), target=row["conf"])

    # Group related MW/% pairs by hand to make the graph readable and deterministic.
    pos = {
        "风电功率": (-1.3, 0.95), "风电占比": (-0.2, 1.25),
        "水电功率": (0.65, 0.72), "水电占比": (1.55, 1.12),
        "光伏功率": (-1.5, -0.50), "光伏占比": (-0.35, -0.86),
        "多燃料功率": (0.55, -0.55), "多燃料占比": (1.60, -0.25),
        "其他可再生功率": (0.42, -1.40), "其他可再生占比": (1.58, -1.37),
    }
    fallback = nx.spring_layout(graph, seed=11)
    for node in graph:
        pos.setdefault(node, fallback[node])

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.set_title("重训认证得到的高权重二阶协同子图", pad=13, weight="bold")
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    nx.draw_networkx_edges(graph, pos, ax=ax, width=2.0 + 7.0 * weights,
                           edge_color=weights, edge_cmap=plt.cm.YlOrRd,
                           edge_vmin=0, edge_vmax=0.8, alpha=0.85)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=2200, node_color="#E7F5F4",
                           edgecolors=CYAN, linewidths=2.0)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_family=_CJK_FAMILY, font_size=10)
    edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=9, font_color="#9A3412",
                                 bbox={"fc": "white", "ec": "none", "alpha": 0.82})
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=mpl.colors.Normalize(vmin=0, vmax=0.8))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"最大目标协同 $\max_c\Delta_c(i,j)$")
    ax.text(0.5, -0.04,
            "边越粗、颜色越深表示两个单独风险较低的字段合并后带来的额外推断能力越强",
            transform=ax.transAxes, ha="center", color="#475569", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "synergy_graph.png")


def draw_triple_certification() -> None:
    data = pd.read_csv(REPO / "DNN_Aggresvation68/outputs/triples_certified.csv")
    top = data[data["group"] == "triple_top"]["syn3_true"].to_numpy()
    random = data[data["group"] == "triple_rand"]["syn3_true"].to_numpy()
    paired = data.dropna(subset=["syn3_est"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    bins = np.linspace(-0.05, 0.48, 35)
    axes[0].hist(random, bins=bins, density=True, alpha=0.68, color="#9CA3AF", label="随机三元组")
    axes[0].hist(top, bins=bins, density=True, alpha=0.72, color=PURPLE, label="Oracle 筛选候选")
    axes[0].axvline(0.1, color=RED, ls="--", lw=1.5, label="强协同阈值 0.1")
    axes[0].set_xlabel(r"重训认证三阶增量协同 $\Delta_c^{(3)}$")
    axes[0].set_ylabel("密度")
    axes[0].set_title("(a) 候选组与随机组的信号分离")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].scatter(paired["syn3_est"], paired["syn3_true"], s=18,
                    color=PURPLE, alpha=0.55, edgecolor="none")
    lo = min(paired["syn3_est"].min(), paired["syn3_true"].min())
    hi = max(paired["syn3_est"].max(), paired["syn3_true"].max())
    axes[1].plot([lo, hi], [lo, hi], "--", color="#64748B", lw=1.2)
    rho = paired[["syn3_est", "syn3_true"]].corr(method="spearman").iloc[0, 1]
    axes[1].text(0.04, 0.92, f"Spearman = {rho:.3f}", transform=axes[1].transAxes,
                 bbox={"fc": "white", "ec": "#CBD5E1", "boxstyle": "round,pad=0.3"})
    axes[1].set_xlabel("Oracle 扫描估计")
    axes[1].set_ylabel("逐集合重训认证")
    axes[1].set_title("(b) 三阶候选筛选保真度")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "triple_certification.png")


def main() -> None:
    draw_framework()
    draw_attacker_scale()
    draw_oracle_fidelity()
    draw_synergy_graph()
    draw_triple_certification()
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
