# -*- coding: utf-8 -*-
"""V4_en 论文图的统一样式：期刊印刷风格（白底、Helvetica、细线、淡网格），色板为已验证的分类色板。
本文方法固定用蓝色；基线按固定顺序取色，不随图变化。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#111111", "#4a4a4a", "#8c8c8c", "#e3e3e3"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
LIGHTBLUE, PALEBLUE = "#9fc3ea", "#dbe9f8"
GRAYS = ["#bdbdbd", "#969696", "#737373", "#525252"]
TEXTW = 6.9   # 版心宽度（英寸），elsarticle 单栏

# 各打分方法的固定颜色（117 号定义对比）
METHOD_STYLE = {
    "pearson": ("Pearson $r^2$", "#c8c8c8"), "spearman": ("Spearman $\\rho^2$", "#b0b0b0"),
    "mi": ("Mutual information", "#989898"), "dcor": ("Distance corr.", "#7f7f7f"),
    "M0": ("Single-field $V(\\{i\\})$", ORANGE),
    "loco": ("LOCO", YELLOW), "perm": ("Permutation imp.", MAGENTA), "sage": ("SAGE", VIOLET),
    "graph": ("Inference-graph prop.", AQUA),
    "M2_est": ("$\\hat M^{(2)}$ (amortized)", LIGHTBLUE), "M2_cert": ("$M^{(2)}$ scan–certify", BLUE),
    "M2": ("$M^{(2)}$ exact", "#0d366b"),
}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4,
    })


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=220); plt.close(fig)
