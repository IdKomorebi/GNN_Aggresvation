"""DNN68 Part B：二阶精确协同图（用 67 号 990 重训真值，逐 conf）。

syn_c(i,j) = v_c(ij) − max(v_c(i), v_c(j))   （>0：两字段合体比单看更泄露 = 协同）
red_c(i,j) = [v_c(i)+v_c(j)] − v_c(ij) 相关的冗余度参考。
阈值：60 号逐 conf 噪声底 p90≈0.007 的倍数（默认 3×=0.021）判显著协同。
对比 56 号置零口径，量化置零对协同排名的扭曲。
输出：synergy2_top_pairs.csv / synergy2_perconf.csv / synergy2_graph_{conf}.png / synergy2_heatmap.png
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TRUTH67 = ROOT.parent / "DNN_Aggresvation67/outputs/retrain"
SYN_THRESH = 0.021   # 3× 噪声底
TOPN = 40


def main():
    # 载入 67 真值：单字段 s{i}、对 p{i}_{j}
    singles = {}; pairs = {}; conf_names = None
    for f in TRUTH67.glob("*_dnn_seed0.json"):
        j = json.loads(f.read_text()); sid = j["subset_id"]
        if conf_names is None:
            conf_names = sorted(j["per_conf_r2"].keys())
        if sid.startswith("s"):
            singles[int(sid[1:])] = j["per_conf_r2"]
        elif sid.startswith("p"):
            a, b = sid[1:].split("_"); pairs[(int(a), int(b))] = j["per_conf_r2"]
    nG = len(singles)
    # 字段名
    import yaml
    sys.path.insert(0, str(ROOT))
    from src.data_processing import prepare_data
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    gen = prepare_data(cfg)["general"]

    # 逐 (pair, conf) 协同
    rows = []
    for (i, j), pv in pairs.items():
        for c in conf_names:
            vi, vj, vij = singles[i][c], singles[j][c], pv[c]
            rows.append({"i": i, "j": j, "fi": gen[i], "fj": gen[j], "conf": c,
                         "vi": vi, "vj": vj, "vij": vij,
                         "max_single": max(vi, vj), "synergy": vij - max(vi, vj)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/synergy2_perconf.csv", index=False)

    top = df.sort_values("synergy", ascending=False).head(TOPN)
    top[["fi", "fj", "conf", "vi", "vj", "vij", "synergy"]].to_csv(
        ROOT / "outputs/synergy2_top_pairs.csv", index=False)
    print(f"=== 逐 conf 二阶协同 top{TOPN}（重训真值）===")
    print(top[["fi", "fj", "conf", "vi", "vj", "vij", "synergy"]].head(20).to_string(index=False))
    n_sig = (df["synergy"] > SYN_THRESH).sum()
    print(f"\n显著协同 (synergy>{SYN_THRESH}) 的 (pair,conf) 条目: {n_sig} / {len(df)}")
    print(f"涉及的不同字段对: {df[df.synergy>SYN_THRESH][['i','j']].drop_duplicates().shape[0]}")

    # ---- 对比 56 号置零口径 ----
    p56 = ROOT.parent / "DNN_Aggresvation56/outputs/synergy_perconf.csv"
    if p56.exists():
        s56 = pd.read_csv(p56)
        # 56 是每对的 argmax-conf 行；对齐我们 per-(pair,conf) 的同 (g1,g2,conf)
        key = lambda r: (frozenset([r["fi"], r["fj"]]), r["conf"])
        our = {(frozenset([r.fi, r.fj]), r.conf): r.synergy for r in df.itertuples()}
        m = []
        for r in s56.itertuples():
            k = (frozenset([r.g1, r.g2]), r.confidential)
            if k in our:
                m.append((r.synergy, our[k]))
        if m:
            a = np.array(m)
            from scipy.stats import spearmanr
            rho = spearmanr(a[:, 0], a[:, 1]).statistic
            print(f"\n56 号置零 vs 重训真值 协同 Spearman={rho:.3f}（{len(m)} 条对齐）")
            print(f"  置零均值 {a[:,0].mean():.3f} vs 重训均值 {a[:,1].mean():.3f}"
                  f"（置零系统性低估协同强度）")

    # ---- 可视化：全局协同热力（对 conf 取 max）----
    heat = np.zeros((nG, nG))
    for (i, j), _ in pairs.items():
        s = df[(df.i == i) & (df.j == j)]["synergy"].max()
        heat[i, j] = heat[j, i] = max(s, 0)
    plt.figure(figsize=(9, 7.5))
    plt.imshow(heat, cmap="hot_r"); plt.colorbar(label="max-over-conf synergy")
    plt.title("Pairwise synergy (retrain truth, max over 12 confidential)")
    plt.xlabel("field j"); plt.ylabel("field i"); plt.tight_layout()
    plt.savefig(ROOT / "outputs/synergy2_heatmap.png", dpi=150)
    print("\nsaved synergy2_heatmap.png / top_pairs.csv / perconf.csv")


if __name__ == "__main__":
    main()
