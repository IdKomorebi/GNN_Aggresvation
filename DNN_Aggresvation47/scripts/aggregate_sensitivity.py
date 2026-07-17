#!/usr/bin/env python3
"""DNN47 敏感度汇总：四种方法各自独立标准化(max=1)+排名，再做一致性分析。

读取 outputs/sensitivity/{mask,single,attn,ig}_raw.csv，输出：
  - outputs/sensitivity/sensitivity_ranking.csv  每个 general × 四方法 (norm 分数 + 排名)
  - outputs/sensitivity/consistency_spearman.csv  四方法排名的 Spearman 相关矩阵
  - outputs/sensitivity/sensitivity_overview.png   一致性热图 + top-15 字段四方法对比
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs/sensitivity"
# 自动检测所有已产出的方法（*_raw.csv），按固定顺序排列已知方法
_ORDER = ["mask", "single", "attn", "ig", "gate", "shapley"]
_found = {p.stem.replace("_raw", "") for p in OUT.glob("*_raw.csv")}
METHODS = [m for m in _ORDER if m in _found] + sorted(_found - set(_ORDER))


def main() -> None:
    dfs = {}
    for mth in METHODS:
        p = OUT / f"{mth}_raw.csv"
        if not p.exists():
            print(f"[warn] 缺少 {p}")
            continue
        dfs[mth] = pd.read_csv(p).set_index("general_field")["raw_score"]
    if not dfs:
        print("没有可汇总的结果"); return

    table = pd.DataFrame(index=next(iter(dfs.values())).index)
    for mth, s in dfs.items():
        raw = s.reindex(table.index)
        mx = raw.max()
        norm = raw / mx if mx > 0 else raw * 0.0      # 独立标准化：最高=1
        table[f"{mth}_norm"] = norm
        table[f"{mth}_rank"] = norm.rank(ascending=False, method="min").astype(int)

    # 按四方法平均排名给一个总览顺序（仅用于展示，不是最终分数）
    rank_cols = [f"{m}_rank" for m in dfs]
    table["avg_rank"] = table[rank_cols].mean(axis=1)
    table = table.sort_values("avg_rank")
    table.to_csv(OUT / "sensitivity_ranking.csv")

    # 一致性：四方法排名的 Spearman 相关
    norm_cols = [f"{m}_norm" for m in dfs]
    spearman = table[norm_cols].corr(method="spearman")
    spearman.to_csv(OUT / "consistency_spearman.csv")

    print("\n===== 四方法排名一致性 (Spearman) =====")
    print(spearman.round(3).to_string())
    print("\n===== 各方法 top-5 general 字段 =====")
    for m in dfs:
        top = table.sort_values(f"{m}_rank").head(5).index.tolist()
        print(f"  {m:7s}: {top}")
    print("\n===== 综合(平均排名) top-10 =====")
    print(table.head(10)[[f"{m}_rank" for m in dfs] + ["avg_rank"]].to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        M = list(dfs.keys())
        # 图1：一致性热图（单独）
        fig1, ax = plt.subplots(figsize=(1.3 * len(M) + 2, 1.3 * len(M) + 1))
        im = ax.imshow(spearman.values, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(len(M))); ax.set_xticklabels(M, rotation=30, ha="right")
        ax.set_yticks(range(len(M))); ax.set_yticklabels(M)
        for i in range(len(M)):
            for j in range(len(M)):
                v = spearman.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.85 else "black", fontsize=10)
        ax.set_title("Method ranking consistency (Spearman)")
        fig1.colorbar(im, ax=ax, fraction=0.046)
        fig1.tight_layout(); fig1.savefig(OUT / "consistency_spearman.png", dpi=140)
        print(f"\n一致性图: {OUT/'consistency_spearman.png'}")

        # 图2：全字段 × 方法 大表格（按平均排名降序，每格写 0-1 分数 + 颜色）
        norm_cols = [f"{m}_norm" for m in M]
        data = table[norm_cols].values                  # (n_fields, n_methods)
        nf = data.shape[0]
        fig2, ax2 = plt.subplots(figsize=(1.5 * len(M) + 4, 0.32 * nf + 1.5))
        im2 = ax2.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
        ax2.set_xticks(range(len(M))); ax2.set_xticklabels(M, fontweight="bold")
        ax2.set_yticks(range(nf))
        ax2.set_yticklabels([f"{r+1}. {f}" for r, f in enumerate(table.index)], fontsize=7)
        for i in range(nf):
            for j in range(len(M)):
                v = data[i, j]
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.6 else "black", fontsize=6.5)
        ax2.set_title(f"Per-field sensitivity ({len(M)} methods, sorted by avg rank; "
                      "each cell = normalized score 0-1)")
        fig2.colorbar(im2, ax=ax2, fraction=0.02)
        fig2.tight_layout(); fig2.savefig(OUT / "sensitivity_table.png", dpi=140)
        print(f"大表格图: {OUT/'sensitivity_table.png'}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")
    print(f"排名表已保存: {OUT/'sensitivity_ranking.csv'}")


if __name__ == "__main__":
    main()
