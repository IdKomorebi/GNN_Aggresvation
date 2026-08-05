#!/usr/bin/env python3
"""绘制 89 号实验的三块核心证据。"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-d89")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    stability = pd.read_csv(OUT / "split_stability.csv")
    beam = pd.read_csv(OUT / "beam_path_alignment.csv")
    dnn = pd.read_csv(OUT / "dnn_hierarchy_summary.csv")

    # 稳定性指标与 Spearman 不依赖 tau，同一 order 只取一行；
    # 强集合的复现率则明确使用主阈值 tau=0.10。
    stable_order = stability.drop_duplicates("order").sort_values("order")
    stable_tau = stability[np.isclose(stability.tau, 0.10)].sort_values("order")
    beam_tau = beam[np.isclose(beam.tau, 0.10)].sort_values("order")
    dnn = dnn.set_index("selection_group").loc[
        [x for x in ("aligned", "cross_conf", "blind") if x in set(dnn.selection_group)]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25))

    x = np.arange(len(stable_order))
    width = 0.24
    axes[0].bar(x - width, stable_order.spearman, width, label="Spearman")
    axes[0].bar(x, stable_order.conf_match, width, label="argmax-conf match")
    axes[0].bar(
        x + width,
        stable_tau.audit_recall_from_search_threshold,
        width,
        label="strong-set recall",
    )
    axes[0].set_xticks(x, [f"order {o}" for o in stable_order.order])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("(a) Search-to-audit stability")
    axes[0].set_ylabel("fraction / correlation")
    axes[0].legend(fontsize=8)

    x = np.arange(len(beam_tau))
    width = 0.32
    axes[1].bar(x - width / 2, beam_tau.recall, width, label="found by beam")
    axes[1].bar(
        x + width / 2,
        beam_tau.aligned_path_recall,
        width,
        label="same-conf path",
    )
    axes[1].set_xticks(x, [f"order {o}" for o in beam_tau.order])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) Actual B=1000 beam, audit truth")
    axes[1].set_ylabel("recall at syn > 0.10")
    axes[1].legend(fontsize=8)

    x = np.arange(len(dnn))
    width = 0.25
    axes[2].bar(x - width, dnn.dnn_syn4, width, label="4th-order syn")
    axes[2].bar(x, dnn.dnn_parent_any, width, label="best 3rd-order parent")
    axes[2].bar(
        x + width,
        dnn.dnn_parent_aligned,
        width,
        label="same-conf parent",
    )
    axes[2].set_xticks(x, dnn.index)
    axes[2].set_title("(c) Dedicated-DNN audit sample")
    axes[2].set_ylabel("mean R2 increment")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG / "hierarchy_validation.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / "hierarchy_validation.pdf", bbox_inches="tight")
    print(FIG / "hierarchy_validation.png")

    error = pd.read_csv(OUT / "proxy_error_decomposition.csv")
    fig2, axes2 = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, order in zip(axes2, (4, 5)):
        g = error[error.order == order].set_index("group").loc[
            ["random", "audit_strong"]
        ]
        x = np.arange(2)
        width = 0.25
        ax.bar(x - width, g.child_v_mae, width, label="child v MAE")
        ax.bar(x, g.parent_v_mae, width, label="best-parent v MAE")
        ax.bar(x + width, g.syn_mae, width, label="difference MAE")
        ax.set_xticks(x, ["random", "audit-strong"])
        ax.set_title(f"order {order}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    axes2[0].set_ylabel("search-to-audit absolute error")
    axes2[1].legend(fontsize=8)
    fig2.suptitle("Tail selection amplifies parent/child difference error")
    fig2.tight_layout()
    fig2.savefig(FIG / "proxy_error_decomposition.png", dpi=180, bbox_inches="tight")
    fig2.savefig(FIG / "proxy_error_decomposition.pdf", bbox_inches="tight")
    print(FIG / "proxy_error_decomposition.png")


if __name__ == "__main__":
    main()
