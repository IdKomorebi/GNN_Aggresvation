#!/usr/bin/env python3
"""DNN48 top-k 测试：评估 7 种敏感度方法的排名质量。

对每种方法,按其敏感度分数从高到低取 general 字段,从 top-1 递增到 top-44,
每次只用这 top-k 个 general 字段(其余置 0)在同一载体模型(multi graph-combo)上
推断 12 个 confidential,记录平均 R²(over 12 字段)。排名越准 → 曲线越高/越快逼近
全字段上界。画 7 条曲线 + 全字段上界(图结构最优 mean R²)的水平虚线。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from compute_sensitivity import load_model_and_test, per_conf_r2  # noqa

METHODS = ["mask", "single", "attn", "ig", "shapley", "gate", "lrp"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ROOT / "configs/multi_graphcombo.yaml"
    model_pt = Path(sorted(glob.glob(
        str(ROOT / "outputs/tuning/multi_graphcombo/*/training/model.pt")))[-1])
    model, X, y, data_info = load_model_and_test(cfg, model_pt, device)
    gen_idx = data_info["general_indices"]
    gen_names = data_info["general"]
    name2pos = {gen_names[p]: p for p in range(len(gen_idx))}

    def mean_r2(positions: list[int]) -> float:
        Xk = torch.zeros_like(X)
        on = [gen_idx[p] for p in positions]
        if on:
            Xk[:, on, :] = X[:, on, :]
        with torch.no_grad():
            pred = model(Xk)
        return float(np.mean(per_conf_r2(pred, y)))   # 12 confidential 平均 R²

    n_gen = len(gen_idx)
    upper = mean_r2(list(range(n_gen)))               # 全字段上界(图结构最优)
    print(f"全 {n_gen} 字段上界 mean R² = {upper:.4f}")

    rank_df = pd.read_csv(ROOT / "outputs/sensitivity/sensitivity_ranking.csv",
                          index_col="general_field")
    methods = [m for m in METHODS if f"{m}_norm" in rank_df.columns]

    curves = {}
    for m in methods:
        order = rank_df.sort_values(f"{m}_norm", ascending=False).index.tolist()
        pos = [name2pos[nm] for nm in order if nm in name2pos]
        curves[m] = [mean_r2(pos[:k]) for k in range(1, n_gen + 1)]
        print(f"  {m:8s}: top1={curves[m][0]:.3f}  top5={curves[m][4]:.3f}  "
              f"top10={curves[m][9]:.3f}  top20={curves[m][19]:.3f}")

    # 保存曲线数据
    out = pd.DataFrame(curves, index=range(1, n_gen + 1))
    out.index.name = "k"
    out.to_csv(ROOT / "outputs/sensitivity/topk_curves.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(range(1, n_gen + 1))
        colors = {"mask": "#4c72b0", "single": "#dd8452", "attn": "#55a868",
                  "ig": "#c44e52", "shapley": "#8172b3", "gate": "#937860",
                  "lrp": "#da8bc3"}
        fig, ax = plt.subplots(figsize=(11, 7))
        for m in methods:
            ax.plot(ks, curves[m], "-o", ms=2.5, lw=1.6, label=m,
                    color=colors.get(m))
        ax.axhline(upper, ls="--", color="gray", lw=1.5,
                   label=f"full-graph upper bound ({upper:.4f})")
        ax.set_xlabel("k = number of top general fields used")
        ax.set_ylabel("mean R² over 12 confidential")
        ax.set_title("DNN48 top-k test: ranking quality of 7 sensitivity methods")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / "outputs/sensitivity/topk_curves.png", dpi=140)
        print(f"\n曲线图: {ROOT/'outputs/sensitivity/topk_curves.png'}")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")
    print(f"曲线数据: {ROOT/'outputs/sensitivity/topk_curves.csv'}")


if __name__ == "__main__":
    main()
