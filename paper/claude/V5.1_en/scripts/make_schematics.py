# -*- coding: utf-8 -*-
"""V4_en 示意图：图 1 审计框架，图 3 摊销攻击者（通用推断模型）与扫描—认证。"""
import os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.pyplot as plt
import style as S

S.setup()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
FILL = {"in": "#f3f3f3", "v": "#e8f1fb", "m": "#fdf0e8", "d": "#e9f6f0"}
EDGE = {"in": "#9a9a9a", "v": S.BLUE, "m": S.ORANGE, "d": S.AQUA}


def box(ax, x, y, w, h, title, body=None, kind="in", tsize=7.6, bsize=6.7, lw=0.8, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=FILL[kind], ec=EDGE[kind], lw=lw))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, color=S.INK, fontweight="bold")
        return
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=tsize, color=S.INK, fontweight="bold")
    tx = x + w / 2 if align == "center" else x + 1.2
    ax.text(tx, y + h - 5.2, body, ha=align, va="top", fontsize=bsize, color=S.INK2, linespacing=1.35)


def arrow(ax, p, q, text=None, color=S.INK2, off=(0, 1.2), rad=0.0, fs=6.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=fs,
                color=color, style="italic")


def header(ax, x, w, y, text, color):
    ax.text(x + w / 2, y, text, ha="center", va="bottom", fontsize=7.8, color=color, fontweight="bold")
    ax.plot([x, x + w], [y - 0.6, y - 0.6], color=color, lw=1.0)


def fig_framework():
    fig, ax = plt.subplots(figsize=(S.TEXTW, 3.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    T, B = 7.0, 6.0
    header(ax, 0.5, 19, 41.2, "Inputs", S.MUTED); header(ax, 24, 27, 41.2, "Set level: inferability $V_y(S)$", S.BLUE)
    header(ax, 55, 23, 41.2, "Field level: risk $M^{(K)}_{i\\to y}$", S.ORANGE); header(ax, 82, 17.5, 41.2, "Decisions", S.AQUA)
    box(ax, 0.5, 27.8, 19, 10.9, "Operational records", "full history of every\nfield the auditor holds", "in", tsize=T, bsize=B)
    box(ax, 0.5, 14.4, 19, 10.9, "Disclosure rules", "category and release\ntiming of each field", "in", tsize=T, bsize=B)
    box(ax, 0.5, 1.0, 19, 10.9, "Field roles", "candidates $G$, public\nbaseline $B$, targets $y$", "in", tsize=T, bsize=B)
    arrow(ax, (10, 14.2), (10, 12.1))
    ax.plot([21.3, 21.3], [6.4, 33.2], color=S.INK2, lw=0.8)
    for yy in (33.2, 6.4):
        ax.plot([19.8, 21.3], [yy, yy], color=S.INK2, lw=0.8)
    arrow(ax, (21.3, 30.8), (23.8, 30.8)); arrow(ax, (21.3, 8.9), (23.8, 8.9))
    box(ax, 24, 23.0, 27, 15.7, "Amortized attacker (scan)",
        "masked pre-training once per dataset,\nclosed-form readout per query\n$\\to\\ \\hat V_y(S)$ for all $|S|\\leq K{+}1$, ms per set", "v", tsize=T, bsize=B)
    box(ax, 24, 1.0, 27, 15.7, "Dedicated retraining (certify)",
        "single-/multi-target DNN, gradient\nboosting; best one on validation;\ntest $R^2$ + monotone closure $\\to\\ V_y(S)$", "v", tsize=T, bsize=B)
    arrow(ax, (31, 22.8), (31, 16.9), color=S.BLUE); arrow(ax, (44, 16.9), (44, 22.8), color=S.BLUE)
    ax.text(30.2, 19.85, "top-$k$\nbackgrounds", ha="right", va="center", fontsize=5.8, color=S.BLUE, style="italic")
    ax.text(44.8, 19.85, "certified\nlower bound", ha="left", va="center", fontsize=5.8, color=S.BLUE, style="italic")
    box(ax, 55, 1.0, 23, 37.7, "Budgeted marginal risk",
        "$M^{(K)}_{i\\to y}=\\max_{|T|\\leq K}\\,\\Delta_i(B\\cup T)$\n$\\Delta_i(A)=V_y(A\\cup i)-V_y(A)$\n\n"
        "$\\bullet$ single-field risk $M^{(0)}$\n$\\bullet$ escalation $M^{(K)}-M^{(0)}$\n$\\bullet$ witness background $T^{*}$\n"
        "$\\bullet$ gain matrix $V(ij)-V(j)$\n$\\bullet$ $\\tau$-critical $\\Leftrightarrow$ in a minimal\n    unsafe set of size $\\leq K{+}1$",
        "m", align="left", tsize=T, bsize=B)
    arrow(ax, (51.3, 30.8), (54.8, 30.8)); arrow(ax, (51.3, 8.9), (54.8, 8.9))
    box(ax, 82, 27.8, 17.5, 10.9, "Grading", "grade by $M^{(K)}$, flag\nescalated fields", "d", tsize=T, bsize=B)
    box(ax, 82, 14.4, 17.5, 10.9, "Disposal", "withhold a minimum\nhitting set; keep apart\nfrom witness $T^{*}$", "d", tsize=T, bsize=B)
    box(ax, 82, 1.0, 17.5, 10.9, "Re-grading", "recompute when the\npublic baseline grows", "d", tsize=T, bsize=B)
    for yy in (33.2, 19.8, 6.4):
        arrow(ax, (78.3, yy), (81.8, yy))
    S.save(fig, os.path.join(OUT, "fig1_framework"))


def fig_model():
    import numpy as np
    fig, ax = plt.subplots(figsize=(S.TEXTW, 2.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    T, B = 7.0, 6.0
    header(ax, 0.5, 31, 37.2, "(a) Pre-training, once per dataset", S.BLUE)
    header(ax, 35, 37, 37.2, "(b) Query $(S,y)$: subset-specific readout", S.BLUE)
    header(ax, 76, 23.5, 37.2, "(c) Scan–certify", S.ORANGE)
    rng = np.random.RandomState(3); rows, cols = 6, 8; m = rng.rand(rows, cols) < 0.55
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((1.5 + c * 1.6, 17 + r * 1.95), 1.4, 1.7, fc=S.BLUE if m[r, c] else "#e6e6e6", ec="white", lw=0.4))
    ax.text(7.8, 29.3, "rows $\\times$ all columns held,\nrandom visibility mask $m$", ha="center", va="bottom", fontsize=B, color=S.INK2)
    ax.add_patch(plt.Rectangle((2.0, 14.2), 1.2, 1.3, fc=S.BLUE, ec="none")); ax.text(3.7, 14.85, "visible", va="center", fontsize=5.8, color=S.INK2)
    ax.add_patch(plt.Rectangle((8.6, 14.2), 1.2, 1.3, fc="#e6e6e6", ec="none")); ax.text(10.3, 14.85, "hidden", va="center", fontsize=5.8, color=S.INK2)
    box(ax, 18.5, 17, 13, 11.6, "MLP", "input $[x\\odot m,\\ m]$\n3 layers × 256", "v", tsize=T, bsize=B)
    arrow(ax, (15.0, 22.8), (18.3, 22.8))
    box(ax, 0.5, 0.3, 31, 12.0, "Objective", "predict targets + reconstruct hidden\ncolumns; freeze last hidden layer $\\phi_r$;\nuntrained twin $\\tilde\\phi_r$ (seeds $r=1,2,3$)", "in", tsize=T, bsize=B)
    arrow(ax, (31.8, 22.8), (34.8, 22.8), color=S.BLUE)
    ax.text(33.3, 23.6, "freeze", ha="center", va="bottom", fontsize=5.8, color=S.BLUE, style="italic")
    box(ax, 35, 17, 17.5, 11.6, "Features of $S$", "$x_S=x\\odot m_S$\n$Z_r=[\\,x_S,\\ x_S^2,\\ \\phi_r,\\ \\tilde\\phi_r\\,](x_S, m_S)$", "v", tsize=T, bsize=B)
    box(ax, 54.5, 17, 17.5, 11.6, "Closed-form ridge", "$\\beta_r=(Z_r^{\\top}Z_r+\\lambda I)^{-1}Z_r^{\\top}y$\n$\\lambda$: 5-fold CV on train", "v", tsize=T, bsize=B)
    arrow(ax, (52.7, 22.8), (54.3, 22.8))
    box(ax, 35, 0.3, 37, 12.0, "Estimate", "$\\hat y=\\frac{1}{3}\\sum_r Z_r\\beta_r$ ;  $\\hat V_y(S)=R^2_{\\mathrm{test}}(\\hat y)$ + closure\n"
        "safeguards: s.d. floor; clip features, $\\hat y$ to train range", "v", tsize=T, bsize=B)
    arrow(ax, (63.2, 16.8), (63.2, 12.6))
    box(ax, 76, 17, 23.5, 11.6, "Scan", "$\\hat V_y$ for all $|S|\\leq K{+}1$ $\\to$\n$\\hat M^{(K)}$ + ranked backgrounds", "m", tsize=T, bsize=B)
    box(ax, 76, 0.3, 23.5, 12.0, "Certify", "retrain attackers only on the\ntop-$k$ backgrounds $\\to$ $L^{(K)}\\leq M^{(K)}$", "m", tsize=T, bsize=B)
    arrow(ax, (72.3, 6.6), (75.8, 22.8), rad=-0.25, color=S.ORANGE); arrow(ax, (87.7, 16.8), (87.7, 12.6), color=S.ORANGE)
    S.save(fig, os.path.join(OUT, "fig3_model"))


if __name__ == "__main__":
    fig_framework(); fig_model(); print("ok")
