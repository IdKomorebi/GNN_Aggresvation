#!/usr/bin/env python3
"""DNN39 跨项目对比：general 聚合「不阻断」是否带来提升。

对比 DNN39（general_aggregation=include_confidential，General 也从 Confidential
聚合）与 DNN32_test（general_only，原始阻断），同口径 1 multi + 12 single、
window=1、bipartite、shuffle、nostaged、allloss、seed 42。

生成：
  - outputs/include_vs_block_comparison.csv
  - outputs/include_vs_block.png
并打印关键统计。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

D39 = Path(__file__).resolve().parents[1] / "outputs"
D32 = Path(__file__).resolve().parents[2] / "DNN_Aggresvation32_test" / "outputs"


def multi_r2(base: Path) -> dict[str, float]:
    csv = glob.glob(f"{base}/multi/*/tables/r2_probe_vs_model.csv")[0]
    df = pd.read_csv(csv)
    return dict(zip(df["field"], df["model_r2"]))


def single_r2(base: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for csv in glob.glob(f"{base}/single/*/*/tables/r2_probe_vs_model.csv"):
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            out[r["field"]] = float(r["model_r2"])
    return out


def best_loss(base: Path) -> float:
    sj = glob.glob(f"{base}/multi/*/results/summary.json")[0]
    return float(json.load(open(sj))["best_test_loss"])


def main() -> None:
    m39, m32 = multi_r2(D39), multi_r2(D32)
    s39, s32 = single_r2(D39), single_r2(D32)
    fields = list(m32.keys())

    rows = []
    for f in fields:
        rows.append({
            "field": f,
            "multi_block_32test": m32[f],
            "multi_include_39": m39[f],
            "d_multi": m39[f] - m32[f],
            "single_block_32test": s32.get(f, np.nan),
            "single_include_39": s39.get(f, np.nan),
            "d_single": s39.get(f, np.nan) - s32.get(f, np.nan),
        })
    table = pd.DataFrame(rows).set_index("field")
    out_csv = D39 / "include_vs_block_comparison.csv"
    table.to_csv(out_csv)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 12)
    print(table.round(4).to_string())
    M32 = table["multi_block_32test"].values; M39 = table["multi_include_39"].values
    S32 = table["single_block_32test"].values; S39 = table["single_include_39"].values
    print(f"\nMEAN  multi: {M32.mean():.4f}(block) → {M39.mean():.4f}(include)  Δ={M39.mean()-M32.mean():+.4f}")
    print(f"MEAN single: {S32.mean():.4f}(block) → {S39.mean():.4f}(include)  Δ={S39.mean()-S32.mean():+.4f}")
    print(f"\n【multi】 include vs block: mean Δ={np.mean(M39-M32):+.4f}  median Δ={np.median(M39-M32):+.4f}  胜={int((M39>M32).sum())}/12")
    print(f"multi best_test_loss: block={best_loss(D32):.4f} → include={best_loss(D39):.4f}")
    print(f"single-multi gap: block={S32.mean()-M32.mean():.4f} → include={S39.mean()-M39.mean():.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        short = [f.replace("_mw", "").replace("da_as_total_", "as_")[:18] for f in fields]
        x = np.arange(len(fields)); w = 0.26
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(x - w, table["single_block_32test"], w, label="single (upper bound)", color="#9e9e9e")
        ax.bar(x, table["multi_block_32test"], w, label="multi general_only (DNN32_test)", color="#4c72b0")
        ax.bar(x + w, table["multi_include_39"], w, label="multi include_confidential (DNN39)", color="#dd8452")
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("model R²")
        ax.set_title("DNN39: general aggregation include_confidential vs block (general_only)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(D39 / "include_vs_block.png", dpi=130)
        print(f"\n图已保存: {D39/'include_vs_block.png'}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")
    print(f"对比表已保存: {out_csv}")


if __name__ == "__main__":
    main()
