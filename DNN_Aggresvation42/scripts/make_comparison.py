#!/usr/bin/env python3
"""DNN42 调参汇总：GNN single 调参前(DNN40 baseline) vs 调参后(combo) vs probe 上界。

combo 配置 = input_encoder=mlp + attention_temperature=0.25 + alpha_lr_multiplier=2
            + lr=1e-3 + epochs=500 + patience=150（其余沿用 DNN40，window=1、bipartite）。

生成：
  - outputs/_comparison/tuning_comparison.csv
  - outputs/_comparison/tuning_comparison.png
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

D42 = Path(__file__).resolve().parents[1]
D40 = D42.parent / "DNN_Aggresvation40_InterimSummary"


def combo_r2(tgt: str) -> float | None:
    sj = glob.glob(str(D42 / f"outputs/tuning/full_{tgt}/*/results/summary.json"))
    if not sj:
        return None
    d = json.load(open(sj[0])).get("per_target_r2", {})
    return list(d.values())[0] if d else None


def main() -> None:
    probe: dict[str, float] = {}
    with open(D42 / "outputs/_probe_reference/target_probe_results.csv") as f:
        for r in csv.DictReader(f):
            probe[r["confidential_field"]] = float(r["probe_r2"])

    base: dict[str, float] = {}
    for sj in glob.glob(str(D40 / "outputs/graph_single_window1_fixed_info/*/*/results/summary.json")):
        for k, v in json.load(open(sj)).get("per_target_r2", {}).items():
            base[k] = v

    fields = sorted(probe, key=lambda f: probe[f] - base.get(f, 0))
    rows = []
    for f in fields:
        b, c, p = base.get(f), combo_r2(f), probe.get(f)
        if None in (b, c, p):
            continue
        rows.append({"field": f, "baseline": b, "combo": c, "delta": c - b,
                     "probe": p, "remaining_gap": c - p})

    out_csv = D42 / "outputs/_comparison/tuning_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    B = np.mean([r["baseline"] for r in rows])
    C = np.mean([r["combo"] for r in rows])
    P = np.mean([r["probe"] for r in rows])
    print(f"{'字段':38s}{'base':>8s}{'combo':>8s}{'Δ':>8s}{'probe':>8s}")
    print("-" * 70)
    for r in rows:
        print(f"{r['field']:38s}{r['baseline']:8.4f}{r['combo']:8.4f}{r['delta']:+8.4f}{r['probe']:8.4f}")
    print("-" * 70)
    print(f"{'MEAN':38s}{B:8.4f}{C:8.4f}{C-B:+8.4f}{P:8.4f}")
    print(f"\nbaseline {B:.4f} → combo {C:.4f}（+{C-B:.4f}）；probe 上界 {P:.4f}")
    print(f"GNN–probe 缺口 {P-B:.4f} → {P-C:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        short = [r["field"].replace("_mw", "").replace("da_as_total_", "as_")[:18] for r in rows]
        x = np.arange(len(rows)); w = 0.26
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(x - w, [r["baseline"] for r in rows], w, label="GNN baseline (DNN40)", color="#4c72b0")
        ax.bar(x, [r["combo"] for r in rows], w, label="GNN tuned (DNN42 combo)", color="#dd8452")
        ax.bar(x + w, [r["probe"] for r in rows], w, label="DNN probe (upper bound)", color="#9e9e9e")
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(f"DNN42: GNN single tuning  (mean {B:.3f} → {C:.3f}, probe {P:.3f})")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(D42 / "outputs/_comparison/tuning_comparison.png", dpi=130)
        print(f"图已保存: {D42/'outputs/_comparison/tuning_comparison.png'}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")
    print(f"对比表已保存: {out_csv}")


if __name__ == "__main__":
    main()
