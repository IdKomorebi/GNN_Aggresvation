# -*- coding: utf-8 -*-
"""Schematic figures 1 (audit pipeline) and 2 (two-stage estimator), drawn with matplotlib
so that they can be inspected locally and need no TikZ at compile time."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

F = Path(__file__).resolve().parents[1] / "figures"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif", "savefig.dpi": 300,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GRAY, INK = "#2a6fb5", "#d9622b", "#1b9e77", "#c0392b", "#8d8d88", "#222220"
LIGHT = {"blue": "#dce9f6", "orange": "#fae3d6", "green": "#d8f0e6", "gray": "#eeeeec"}


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight=weight, linespacing=1.35)


def arrow(ax, p, q, color=INK, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


# ------------------------------------------------------------------ Fig 1
fig, ax = plt.subplots(figsize=(7.2, 2.45))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
box(ax, .05, 1.35, 1.55, .85, "candidate\nfield set $S$", LIGHT["gray"], GRAY)
box(ax, 2.00, 1.35, 1.95, .85, "amortized\nestimator\n$\\hat V_y(S)$", LIGHT["blue"], BLUE,
    fs=8, weight="bold")
box(ax, 4.35, 1.35, 2.25, .85, "budgeted marginal capability\n$M^{(K)}_{i\\to y}$\n+ witness $T^*$",
    LIGHT["green"], GREEN, fs=7.2, weight="bold")
box(ax, 7.05, 2.05, 2.85, .72, "risk grading table\n(escalation profile, tier)",
    LIGHT["orange"], ORANGE, fs=8)
box(ax, 7.05, .78, 2.85, .72, "release rule\n$U^{(K)}(A)\\leq\\tau$", LIGHT["orange"], ORANGE,
    fs=8, weight="bold")
arrow(ax, (1.62, 1.77), (1.98, 1.77))
arrow(ax, (3.97, 1.77), (4.33, 1.77))
arrow(ax, (6.62, 1.95), (7.03, 2.35))
arrow(ax, (6.62, 1.60), (7.03, 1.18))
# workloads
ax.add_patch(Rectangle((1.62, 2.46), 2.72, 1.02, fill=False, edgecolor=BLUE, lw=.7,
                       linestyle=":", zorder=0))
ax.text(2.98, 3.28, "audit workloads", ha="center", fontsize=8, color=BLUE, style="italic")
for i, t in enumerate(["(i) ad-hoc query", "(ii) low-order sweep", "(iii) context search"]):
    ax.text(2.98, 3.02 - .21 * i, t, ha="center", fontsize=7.0, color="#6f6f6b")
arrow(ax, (2.98, 2.44), (2.98, 2.22), color=BLUE, lw=.9)
# certification loop
arrow(ax, (5.47, 1.33), (5.47, .72), color=RED, lw=1.0)
ax.text(5.60, .44, "certify witness by retraining\n$\\Rightarrow$ lower bound $L$",
        fontsize=7.0, color=RED, ha="left", linespacing=1.3)
fig.savefig(F / "fig1_framework.pdf"); fig.savefig(F / "fig1_framework.png", dpi=170)
plt.close(fig); print("fig1 done")

# ------------------------------------------------------------------ Fig 2
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
ax.text(2.35, 3.24, "Stage 1: random-mask pre-learning", ha="center", fontsize=8.2,
        weight="bold", color=INK)
ax.text(2.35, 3.02, "(once per market)", ha="center", fontsize=7.2, color="#6f6f6b")
ax.text(7.50, 3.24, "Stage 2: subset-specific adaptation", ha="center",
        fontsize=8.2, weight="bold", color=INK)
ax.text(7.50, 3.02, "(per query, closed form)", ha="center", fontsize=7.2, color="#6f6f6b")
ax.plot([4.85, 4.85], [.15, 3.05], color=GRAY, lw=.8, ls="--")
# stage 1
box(ax, .1, 2.15, 1.35, .62, "$x \\odot m$\n$\\oplus\; m$", LIGHT["gray"], GRAY, fs=8)
box(ax, 1.75, 2.15, 1.35, .62, "backbone\n$\\phi_\\theta$", LIGHT["blue"], BLUE, fs=8, weight="bold")
box(ax, 3.4, 2.15, 1.25, .62, "all 12\ntargets", LIGHT["gray"], GRAY, fs=8)
arrow(ax, (1.47, 2.46), (1.73, 2.46)); arrow(ax, (3.12, 2.46), (3.38, 2.46))
for i, lab in enumerate(["$m \\sim$ stratified uniform", "$m \\sim$ Bernoulli(0.5)",
                         "no masking (ablation)"]):
    ax.text(.15, 1.78 - .27 * i, "$\\bullet$ " + lab, fontsize=7.2, color="#6f6f6b")
ax.text(.15, .72, "masking is necessary;\nits distribution is not critical", fontsize=7.2,
        color=BLUE, style="italic", linespacing=1.35)
# stage 2
box(ax, 5.1, 2.15, 1.5, .62, "frozen $\\phi$\n($R$ seeds)", LIGHT["blue"], BLUE, fs=8)
box(ax, 6.9, 2.15, 1.35, .62, "$Z_{y,S}$\nfeatures", LIGHT["gray"], GRAY, fs=8)
box(ax, 8.55, 2.15, 1.35, .62, "closed-form\nridge $\\beta_{y,S}$", LIGHT["green"], GREEN,
    fs=8, weight="bold")
arrow(ax, (6.62, 2.46), (6.88, 2.46)); arrow(ax, (8.27, 2.46), (8.53, 2.46))
ax.text(5.15, 1.75, "$Z_{y,S}=[\\,x\\odot m_S \\mid (x\\odot m_S)^2 \\mid \\phi_1 \\mid \\cdots \\mid \\phi_R\\,]$",
        fontsize=7.6, color=INK)
ax.text(5.15, 1.40, "solved per $(S,y)$ on train only; test used for $R^2$ alone",
        fontsize=7.2, color=GREEN, style="italic")
# contrast
ax.add_patch(FancyBboxPatch((5.1, .22), 4.8, .92, boxstyle="round,pad=0.012,rounding_size=0.02",
                            facecolor="#fdf1ee", edgecolor=RED, linewidth=1.0))
ax.text(5.3, .97, "contrast: shared output head", fontsize=7.6, color=RED, weight="bold",
        va="center")
ax.text(5.3, .62, "one set of output weights for all $2^{|G|}$ masks", fontsize=7.1,
        color=INK, va="center")
ax.text(5.3, .38, "$\\Rightarrow$ $3.2\\times$ / $4.6\\times$ larger $V$ error", fontsize=7.1,
        color=INK, va="center")
fig.savefig(F / "fig2_model.pdf"); fig.savefig(F / "fig2_model.png", dpi=170)
plt.close(fig); print("fig2 done")
