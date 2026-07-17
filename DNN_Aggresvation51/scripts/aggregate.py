#!/usr/bin/env python3
"""DNN51 聚合消融汇总：读各变体 summary.json，对比 12-conf 平均 R²。

每个变体标注完整配置维度：
  - uses_gat   : 是否采用 GAT（动态 q·k 注意力）
  - gat_scope  : GAT 作用范围（无 / 仅confidential / 全图）
  - corr_weight: 相关系数权重类型（无 / 全局静态 / 部分静态 / 全局动态）
  - mixing     : 先验与动态注意力的混合方式
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# 配置维度（顺序即展示顺序）
VARIANTS = [
    {"variant": "split_baseline", "uses_gat": "是", "gat_scope": "仅confidential(G→C)",
     "corr_weight": "部分静态", "mixing": "G-G:GCN静态邻接 / G→C:逐对动态gate(q·k与逐边先验)",
     "desc": "当前设计：分裂聚合"},
    {"variant": "gcn_noprior", "uses_gat": "否", "gat_scope": "无",
     "corr_weight": "无", "mixing": "GCN，均匀邻接(行归一)",
     "desc": "全图GCN，仅用相关性筛出的拓扑、边不加权"},
    {"variant": "gcn_prior", "uses_gat": "否", "gat_scope": "无",
     "corr_weight": "全局静态", "mixing": "GCN，先验加权静态邻接(行归一)",
     "desc": "全图GCN，全局alpha先验加权"},
    {"variant": "gat_noprior", "uses_gat": "是", "gat_scope": "全图",
     "corr_weight": "无", "mixing": "纯动态 q·k 注意力",
     "desc": "全图GAT，无先验"},
    {"variant": "gat_prior_static", "uses_gat": "是", "gat_scope": "全图",
     "corr_weight": "全局静态", "mixing": "动态q·k + 单一全局scale×log先验(no_gate)",
     "desc": "全图GAT + 全局静态注意力(no_gate)"},
    {"variant": "gat_prior", "uses_gat": "是", "gat_scope": "全图",
     "corr_weight": "全局动态", "mixing": "逐对动态gate混合(q·k与全局先验)",
     "desc": "全图GAT + 全局动态gate"},
]

PROBE_CSV = OUT / "_probe_reference" / "target_probe_results.csv"


def probe_mean(conf_fields) -> float | None:
    if not PROBE_CSV.exists():
        return None
    df = pd.read_csv(PROBE_CSV)
    sub = df[df["confidential_field"].isin(conf_fields)]
    return float(sub["probe_r2"].mean()) if len(sub) else None


def main() -> None:
    rows = []
    base = None
    conf_fields = None
    for cfg in VARIANTS:
        sj = OUT / cfg["variant"] / "summary.json"
        if not sj.exists():
            print(f"  [缺失] {sj}")
            continue
        s = json.loads(sj.read_text())
        if conf_fields is None:
            conf_fields = list(s["per_target_r2"].keys())
        r2 = s["mean_r2_12conf"]
        if cfg["variant"] == "split_baseline":
            base = r2
        row = dict(cfg); row["mean_r2_12conf"] = round(r2, 4)
        rows.append(row)

    pm = probe_mean(conf_fields) if conf_fields else None
    for r in rows:
        r["delta_vs_baseline"] = None if base is None else round(r["mean_r2_12conf"] - base, 4)
        r["gap_to_probe"] = None if pm is None else round(r["mean_r2_12conf"] - pm, 4)

    df = pd.DataFrame(rows, columns=["variant", "uses_gat", "gat_scope", "corr_weight",
                                     "mixing", "mean_r2_12conf", "delta_vs_baseline",
                                     "gap_to_probe", "desc"])
    df.to_csv(OUT / "ablation_summary.csv", index=False)

    # 柱状图：平均 R²，标注基准虚线 + probe 上界虚线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["variant"] for r in rows]
        vals = [r["mean_r2_12conf"] for r in rows]
        colors = ["#444444" if r["variant"] == "split_baseline" else
                  ("#c0392b" if r["mean_r2_12conf"] < (base or 0) - 0.012 else "#4c72b0")
                  for r in rows]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        bars = ax.bar(names, vals, color=colors)
        if base is not None:
            ax.axhline(base, ls="--", color="#444444", lw=1.2, label=f"split baseline = {base:.4f}")
        if pm is not None:
            ax.axhline(pm, ls=":", color="#59a14f", lw=1.4, label=f"DNN probe upper bound = {pm:.4f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0.84, 0.90)
        ax.set_ylabel("mean test R2 over 12 confidential")
        ax.set_title("DNN51: aggregation ablation (vs split design & DNN probe)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "ablation_r2.png", dpi=140)
        print(f"柱状图已保存: {OUT/'ablation_r2.png'}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")

    print("\n=== DNN51 聚合方式消融：12-conf 平均 R² ===")
    if pm is not None:
        print(f"(DNN probe 上界 = {pm:.4f})")
    print(f"{'变体':18s}{'平均R²':>9s}{'Δvs基准':>10s}{'距上界':>9s}  {'GAT范围':14s}{'相关系数权重'}")
    for r in rows:
        d = "" if r["delta_vs_baseline"] is None else f"{r['delta_vs_baseline']:+.4f}"
        g = "" if r["gap_to_probe"] is None else f"{r['gap_to_probe']:+.4f}"
        print(f"{r['variant']:18s}{r['mean_r2_12conf']:>9.4f}{d:>10s}{g:>9s}  "
              f"{r['gat_scope']:14s}{r['corr_weight']}")
    print(f"\n汇总表已保存: {OUT/'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
