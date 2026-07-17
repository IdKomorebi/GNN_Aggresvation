#!/usr/bin/env python3
"""DNN53 汇总：经典(加性)GAT vs DNN52 transformer(q·k) GAT，同编码同模式直接比。

参考线：DNN probe 上界、DNN52 gcn_dynamic（各档最佳非-GAT 基线）。
产出：
  outputs/_summary/classic_vs_transformer.png   折线(实线=classic, 虚线=transformer)
  outputs/_summary/per_encoder_bars.png          每档分组柱状
  outputs/_summary/summary.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
D52 = ROOT.parent / "DNN_Aggresvation52" / "outputs"   # transformer GAT 参照

ENCS = ["enc1_linear_noact", "enc2_linear", "enc3_mlp"]
ELAB = {"enc1_linear_noact": "linear_noact\n(weak)", "enc2_linear": "linear\n(mid)",
        "enc3_mlp": "mlp\n(strong)"}
MODES = ["gat_noprior", "gat_static", "gat_dynamic"]
MCOLOR = {"gat_noprior": "#f28e2b", "gat_static": "#e15759", "gat_dynamic": "#4c72b0"}


def rd(base, enc, mode):
    f = base / enc / mode / "summary.json"
    return json.loads(f.read_text())["mean_r2_12conf"] if f.exists() else np.nan


def probe_mean():
    df = pd.read_csv(OUT / "_probe_reference/target_probe_results.csv")
    conf = json.load(open(OUT / "enc3_mlp/gat_dynamic/summary.json"))["per_target_r2"].keys()
    return float(df[df["confidential_field"].isin(conf)]["probe_r2"].mean())


def main():
    pm = probe_mean()
    rows = []
    for e in ENCS:
        for mo in MODES:
            c = rd(OUT, e, mo); t = rd(D52, e, mo)
            rows.append({"encoder": e, "mode": mo, "classic_r2": round(c, 4),
                         "transformer_r2": round(t, 4), "delta_classic_minus_tf": round(c - t, 4)})
    sumdir = OUT / "_summary"; sumdir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df["probe"] = round(pm, 4)
    df["gcn_dynamic_ref"] = [round(rd(D52, r["encoder"], "gcn_dynamic"), 4) for r in rows]
    df.to_csv(sumdir / "summary.csv", index=False)

    x = np.arange(len(ENCS))
    gdyn = [rd(D52, e, "gcn_dynamic") for e in ENCS]

    # 折线：classic(实线) vs transformer(虚线)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for mo in MODES:
        ax.plot(x, [rd(OUT, e, mo) for e in ENCS], "-o", color=MCOLOR[mo], lw=2.2,
                label=f"{mo} (classic)")
        ax.plot(x, [rd(D52, e, mo) for e in ENCS], "--s", color=MCOLOR[mo], lw=1.6,
                alpha=0.7, label=f"{mo} (transformer)")
    ax.plot(x, gdyn, ":^", color="#59a14f", lw=2, label="gcn_dynamic (DNN52 best)")
    ax.axhline(pm, ls=":", color="#333", lw=1.3, label=f"DNN probe = {pm:.4f}")
    ax.set_xticks(x); ax.set_xticklabels([ELAB[e] for e in ENCS], fontsize=8)
    ax.set_ylabel("mean test R2 over 12 confidential")
    ax.set_title("DNN53: classic(additive, K=V=Wh) vs transformer(q·k) GAT")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(sumdir / "classic_vs_transformer.png", dpi=140); plt.close(fig)

    # 每档分组柱状
    fig, axes = plt.subplots(1, len(ENCS), figsize=(13, 5), sharey=True)
    for ax, e in zip(axes, ENCS):
        cv = [rd(OUT, e, mo) for mo in MODES]
        tv = [rd(D52, e, mo) for mo in MODES]
        xx = np.arange(len(MODES)); w = 0.38
        ax.bar(xx - w / 2, cv, w, label="classic", color="#4c72b0")
        ax.bar(xx + w / 2, tv, w, label="transformer", color="#bbbbbb")
        ax.axhline(pm, ls=":", color="#333", lw=1)
        ax.axhline(rd(D52, e, "gcn_dynamic"), ls="--", color="#59a14f", lw=1.2)
        for i, (a, b) in enumerate(zip(cv, tv)):
            ax.text(i - w / 2, a + 0.001, f"{a:.3f}", ha="center", fontsize=6.5)
            ax.text(i + w / 2, b + 0.001, f"{b:.3f}", ha="center", fontsize=6.5)
        ax.set_xticks(xx); ax.set_xticklabels([m.replace("gat_", "") for m in MODES], fontsize=8)
        ax.set_title(ELAB[e].replace("\n", " ")); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(min(min(cv), min(tv)) - 0.02, max(pm, max(cv)) + 0.01)
    axes[0].set_ylabel("mean R2"); axes[0].legend(fontsize=8)
    fig.suptitle("DNN53 classic vs DNN52 transformer GAT  (green dashed = gcn_dynamic, dotted = probe)")
    fig.tight_layout(); fig.savefig(sumdir / "per_encoder_bars.png", dpi=140); plt.close(fig)

    print("=== DNN53 经典 GAT vs DNN52 transformer GAT (12-conf 平均 R²) ===")
    print(f"DNN probe = {pm:.4f}\n")
    print(f"{'encoder':18s}{'mode':14s}{'classic':>9s}{'transf':>9s}{'Δ(c-t)':>9s}{'gcn_dyn':>9s}")
    for r, gd in zip(rows, df["gcn_dynamic_ref"]):
        print(f"{r['encoder']:18s}{r['mode']:14s}{r['classic_r2']:>9.4f}"
              f"{r['transformer_r2']:>9.4f}{r['delta_classic_minus_tf']:>+9.4f}{gd:>9.4f}")
    print(f"\n图: {sumdir}/classic_vs_transformer.png, per_encoder_bars.png")


if __name__ == "__main__":
    main()
