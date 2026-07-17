#!/usr/bin/env python3
"""DNN52 汇总：每编码档的 6 方法对比图 + 跨编码汇总（核心：弱编码是否拉开差距）。

产出：
  outputs/<enc>/comparison_6methods.png + comparison.csv   每档 6 方法平均 R²
  outputs/_summary/encoder_effect.png                       每方法 R² 随编码强度变化(折线)
  outputs/_summary/heatmap.png                              mode × encoder 平均 R² 热图
  outputs/_summary/spread.png                               每档 6 方法极差(max-min) 随编码
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

ENC_ORDER = ["enc1_linear_noact", "enc2_linear", "enc3_mlp", "enc4_mlp_deep"]
ENC_LABEL = {"enc1_linear_noact": "linear_noact\n(weak)", "enc2_linear": "linear\n(mid)",
             "enc3_mlp": "mlp\n(strong)", "enc4_mlp_deep": "mlp_deep\n(strongest)"}
MODES = ["gcn_noprior", "gcn_static", "gcn_dynamic",
         "gat_noprior", "gat_static", "gat_dynamic"]
MCOLOR = {"gcn_noprior": "#9e9e9e", "gcn_static": "#76b7b2", "gcn_dynamic": "#59a14f",
          "gat_noprior": "#f28e2b", "gat_static": "#e15759", "gat_dynamic": "#4c72b0"}


def probe_mean(conf):
    df = pd.read_csv(OUT / "_probe_reference/target_probe_results.csv")
    return float(df[df["confidential_field"].isin(conf)]["probe_r2"].mean())


def main():
    encs = [e for e in ENC_ORDER if (OUT / e).is_dir()]
    grid = {}                                # grid[enc][mode] = mean_r2
    conf = None
    for e in encs:
        grid[e] = {}
        for mo in MODES:
            sj = OUT / e / mo / "summary.json"
            if sj.exists():
                s = json.loads(sj.read_text())
                grid[e][mo] = s["mean_r2_12conf"]
                if conf is None:
                    conf = list(s["per_target_r2"].keys())
    pm = probe_mean(conf) if conf else None

    # 每档：6 方法对比图
    for e in encs:
        vals = [grid[e].get(mo, np.nan) for mo in MODES]
        pd.DataFrame({"mode": MODES, "mean_r2_12conf": vals}).to_csv(
            OUT / e / "comparison.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(MODES, vals, color=[MCOLOR[mo] for mo in MODES])
        if pm is not None:
            ax.axhline(pm, ls=":", color="#333", lw=1.3, label=f"DNN probe = {pm:.4f}")
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
        lo = np.nanmin(vals); hi = np.nanmax(vals)
        ax.set_ylim(lo - 0.02, max(hi, pm or hi) + 0.01)
        ax.set_ylabel("mean test R2 over 12 confidential")
        ax.set_title(f"{e}: 6 aggregation modes (spread {hi-lo:.4f})")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / e / "comparison_6methods.png", dpi=130); plt.close(fig)

    # 跨编码汇总
    sumdir = OUT / "_summary"; sumdir.mkdir(exist_ok=True)
    rows = []
    for e in encs:
        for mo in MODES:
            rows.append({"encoder": e, "mode": mo, "mean_r2": grid[e].get(mo, np.nan)})
    sdf = pd.DataFrame(rows)
    sdf["probe_mean"] = pm
    sdf.to_csv(sumdir / "summary.csv", index=False)

    xlab = [ENC_LABEL.get(e, e) for e in encs]
    x = np.arange(len(encs))

    # 折线：每方法 R² 随编码强度
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mo in MODES:
        ax.plot(x, [grid[e].get(mo, np.nan) for e in encs], "-o",
                color=MCOLOR[mo], label=mo, lw=2)
    if pm is not None:
        ax.axhline(pm, ls=":", color="#333", lw=1.3, label=f"DNN probe = {pm:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=8)
    ax.set_ylabel("mean test R2 over 12 confidential")
    ax.set_title("DNN52: aggregation R2 vs input-encoder strength")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(sumdir / "encoder_effect.png", dpi=140); plt.close(fig)

    # 极差折线：6 方法分化随编码
    fig, ax = plt.subplots(figsize=(8, 5))
    spreads = [np.nanmax([grid[e].get(mo, np.nan) for mo in MODES]) -
               np.nanmin([grid[e].get(mo, np.nan) for mo in MODES]) for e in encs]
    ax.plot(x, spreads, "-o", color="#c0392b", lw=2)
    for i, v in enumerate(spreads):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=8)
    ax.set_ylabel("6-mode spread (max - min)")
    ax.set_title("DNN52: do methods diverge as encoder weakens?")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(sumdir / "spread.png", dpi=140); plt.close(fig)

    # 热图 mode × encoder
    M = np.array([[grid[e].get(mo, np.nan) for e in encs] for mo in MODES])
    fig, ax = plt.subplots(figsize=(1.6 * len(encs) + 2, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(encs))); ax.set_xticklabels([e.replace("enc", "") for e in encs], fontsize=8)
    ax.set_yticks(range(len(MODES))); ax.set_yticklabels(MODES, fontsize=9)
    for i in range(len(MODES)):
        for j in range(len(encs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean R2")
    ax.set_title("DNN52: mean R2  (mode × encoder)")
    fig.tight_layout(); fig.savefig(sumdir / "heatmap.png", dpi=140); plt.close(fig)

    print("=== DNN52 跨编码汇总 (12-conf 平均 R²) ===")
    if pm: print(f"DNN probe 上界 = {pm:.4f}")
    head = "mode".ljust(14) + "".join(e.replace("enc", "").replace("_", "")[:10].rjust(12) for e in encs)
    print(head)
    for mo in MODES:
        print(mo.ljust(14) + "".join(f"{grid[e].get(mo, float('nan')):.4f}".rjust(12) for e in encs))
    print("spread".ljust(14) + "".join(f"{s:.4f}".rjust(12) for s in spreads))
    print(f"\n图已保存: {sumdir}/encoder_effect.png, heatmap.png, spread.png")


if __name__ == "__main__":
    main()
