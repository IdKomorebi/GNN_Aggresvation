#!/usr/bin/env python3
"""DNN35 跨 run 汇总：对比 target-specific value（FiLM）的增益。

三臂（都在 DNN34 最优的 dynamic head attention + 8 heads 上）：
  - dynamic_8head      ：无 FiLM（= DNN34 最优复现，对照锚点）
  - film_static_8head  ：静态 FiLM（per-target 查找表）
  - film_dynamic_8head ：动态 FiLM（随当前状态生成 Γ/B）

读取 outputs/<variant>/<latest_timestamp>/ 下的：
  - tables/r2_probe_vs_model.csv  （每个 confidential 字段的 model_r2）
  - results/summary.json          （best_test_loss）
生成：
  - outputs/film_comparison.csv
  - outputs/film_comparison.png
并打印关键统计，供 CHANGELOG 引用。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
# single 模式（每个字段单独训一个模型）的性能上限，来自 DNN32。
SINGLE_CSV = (
    Path(__file__).resolve().parents[2]
    / "DNN_Aggresvation32" / "outputs" / "multi_vs_single.csv"
)
VARIANTS = [
    "dynamic_8head",       # 无 FiLM，对照锚点
    "film_static_8head",   # 静态 FiLM
    "film_dynamic_8head",  # 动态 FiLM
]


def latest(variant: str) -> Path | None:
    dirs = sorted(p for p in (OUT / variant).glob("*/") if p.is_dir())
    return dirs[-1] if dirs else None


def main() -> None:
    rows: dict[str, dict[str, float]] = {}
    probe: dict[str, float] = {}
    best_loss: dict[str, float] = {}
    field_order: list[str] = []

    for v in VARIANTS:
        d = latest(v)
        if d is None:
            print(f"[warn] 缺少 variant: {v}")
            continue
        df = pd.read_csv(d / "tables" / "r2_probe_vs_model.csv")
        for _, r in df.iterrows():
            f = r["field"]
            if f not in rows:
                rows[f] = {}
                field_order.append(f)
            rows[f][v] = float(r["model_r2"])
            probe[f] = float(r["probe_r2"])
        sj = json.loads((d / "results" / "summary.json").read_text())
        best_loss[v] = float(sj.get("best_test_loss", float("nan")))

    table = pd.DataFrame(
        {v: [rows[f].get(v, np.nan) for f in field_order] for v in VARIANTS},
        index=field_order,
    )
    # single 上限（DNN32）
    single_r2: dict[str, float] = {}
    if SINGLE_CSV.exists():
        sdf = pd.read_csv(SINGLE_CSV)
        single_r2 = dict(zip(sdf["field"], sdf["single_r2"]))
    else:
        print(f"[warn] 缺少 single 上限文件: {SINGLE_CSV}")

    table.insert(0, "single_r2", [single_r2.get(f, np.nan) for f in field_order])
    table.insert(0, "probe_r2", [probe[f] for f in field_order])
    table["film_static_gain"] = table["film_static_8head"] - table["dynamic_8head"]
    table["film_dynamic_gain"] = table["film_dynamic_8head"] - table["dynamic_8head"]
    table["best_arm_minus_single"] = (
        table[["dynamic_8head", "film_static_8head", "film_dynamic_8head"]].max(axis=1)
        - table["single_r2"]
    )

    out_csv = OUT / "film_comparison.csv"
    table.to_csv(out_csv)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n===== per-field model R² =====")
    print(table.round(4).to_string())

    print("\n===== best_test_loss =====")
    for v in VARIANTS:
        print(f"  {v:20s}: {best_loss.get(v, float('nan')):.4f}")

    def stat(col: str) -> str:
        s = table[col].dropna()
        n_pos = int((s > 0).sum())
        return (f"mean={s.mean():+.4f}  median={s.median():+.4f}  "
                f"win={n_pos}/{len(s)}")

    print("\n===== FiLM 增益（相对无 FiLM 的 dynamic_8head）=====")
    print(f"  static  FiLM gain : {stat('film_static_gain')}")
    print(f"  dynamic FiLM gain : {stat('film_dynamic_gain')}")
    print("\n===== 距 single 上限（取三臂中每字段最好的）=====")
    print(f"  best_arm - single : {stat('best_arm_minus_single')}")

    # ---- 柱状图：每字段 single(上限) / nofilm / film_static / film_dynamic ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        short = [f.replace("_mw", "").replace("da_as_total_", "as_")[:18]
                 for f in field_order]
        x = np.arange(len(field_order))
        w = 0.2  # 细柱
        fig, ax = plt.subplots(figsize=(16, 6.5))
        bars = [
            ("single (DNN32 upper bound)", "single_r2", "#9e9e9e"),
            ("dynamic_8head (no FiLM)", "dynamic_8head", "#4c72b0"),
            ("film_static_8head", "film_static_8head", "#dd8452"),
            ("film_dynamic_8head", "film_dynamic_8head", "#55a868"),
        ]
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
        for off, (label, col, color) in zip(offsets, bars):
            ax.bar(x + off, table[col], w, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("model R²")
        ax.set_title(
            "DNN35: target-specific value (FiLM) on top of dynamic_8head, "
            "with single upper bound"
        )
        ax.legend(ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "film_comparison.png", dpi=130)
        print(f"\n图已保存: {OUT / 'film_comparison.png'}")
    except Exception as e:  # 绘图失败不影响 csv 产出
        print(f"[warn] 绘图失败: {e}")

    print(f"\n对比表已保存: {out_csv}")


if __name__ == "__main__":
    main()
