#!/usr/bin/env python3
"""DNN60 汇总：噪声底 σ 与无偏随机子集真值表。

输出：
  outputs/noise_floor.csv    每个 nf 子集 × 结构：5 种子的 mean/std/range（12-conf 平均 R² 口径）
  outputs/noise_floor_conf.csv  逐 confidential 的种子 std（防护/协同结论的逐目标噪声底）
  outputs/random_eval.csv    50 个随机子集 × 2 结构的 v 真值 + best-of-struct
  outputs/phase0_summary.json
  outputs/phase0_plots.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RET = ROOT / "outputs/retrain"

rows = []
for f in sorted(RET.glob("*.json")):
    rows.append(json.loads(f.read_text()))
df = pd.DataFrame(rows)
conf_fields = sorted(rows[0]["per_conf_r2"].keys())
for c in conf_fields:
    df[f"r2::{c}"] = df["per_conf_r2"].apply(lambda d: d[c])

# ---------- 1. 噪声底 ----------
nf = df[df["group"] == "noise_floor"]
nf_rows = []
for (sid, struct), g in nf.groupby(["subset_id", "struct"]):
    vals = g["mean_r2"].to_numpy()
    nf_rows.append({
        "subset_id": sid, "size": int(g["size"].iloc[0]), "struct": struct,
        "n_seeds": len(vals), "v_mean": vals.mean(), "v_std": vals.std(ddof=1),
        "v_range": vals.max() - vals.min(),
    })
nf_df = pd.DataFrame(nf_rows).sort_values(["size", "struct"]).reset_index(drop=True)
nf_df.to_csv(ROOT / "outputs/noise_floor.csv", index=False)

# 逐 confidential 的种子 std
nfc_rows = []
for (sid, struct), g in nf.groupby(["subset_id", "struct"]):
    for c in conf_fields:
        vals = g[f"r2::{c}"].to_numpy()
        nfc_rows.append({"subset_id": sid, "size": int(g["size"].iloc[0]), "struct": struct,
                         "conf": c, "r2_mean": vals.mean(), "r2_std": vals.std(ddof=1)})
nfc_df = pd.DataFrame(nfc_rows)
nfc_df.to_csv(ROOT / "outputs/noise_floor_conf.csv", index=False)

# ---------- 2. 随机评测真值 ----------
rs = df[df["group"] == "random_eval"]
piv = rs.pivot_table(index=["subset_id", "size"], columns="struct", values="mean_r2").reset_index()
piv["v_best"] = piv[["dnn", "gcn"]].max(axis=1)
piv["best_struct"] = np.where(piv["gcn"] >= piv["dnn"], "gcn", "dnn")
piv = piv.sort_values("size").reset_index(drop=True)
piv.to_csv(ROOT / "outputs/random_eval.csv", index=False)

# ---------- 3. 摘要 ----------
summary = {
    "noise_floor": {
        "n_subsets": int(nf_df["subset_id"].nunique()),
        "sigma_mean_r2": {
            struct: {
                "median": float(g["v_std"].median()),
                "max": float(g["v_std"].max()),
                "by_size": {int(r["size"]): round(float(r["v_std"]), 5) for _, r in g.iterrows()},
            } for struct, g in nf_df.groupby("struct")
        },
        "sigma_per_conf": {
            struct: {"median": float(g["r2_std"].median()), "p90": float(g["r2_std"].quantile(0.9)),
                     "max": float(g["r2_std"].max())}
            for struct, g in nfc_df.groupby("struct")
        },
    },
    "random_eval": {
        "n_subsets": int(len(piv)),
        "gcn_wins": int((piv["best_struct"] == "gcn").sum()),
        "dnn_wins": int((piv["best_struct"] == "dnn").sum()),
        "gcn_minus_dnn_by_sizeband": {},
        "v_best_range": [float(piv["v_best"].min()), float(piv["v_best"].max())],
    },
}
bands = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]
for lo, hi in bands:
    sub = piv[(piv["size"] >= lo) & (piv["size"] <= hi)]
    if len(sub):
        summary["random_eval"]["gcn_minus_dnn_by_sizeband"][f"{lo}-{hi}"] = {
            "n": int(len(sub)), "mean_diff": round(float((sub["gcn"] - sub["dnn"]).mean()), 4)}
(ROOT / "outputs/phase0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

# ---------- 4. 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for struct, marker in [("dnn", "o"), ("gcn", "s")]:
    g = nf_df[nf_df["struct"] == struct]
    axes[0].plot(g["size"], g["v_std"], marker, ls="-", label=struct)
axes[0].set_xlabel("subset size"); axes[0].set_ylabel("seed std of mean R2")
axes[0].set_title("Noise floor (5 seeds)"); axes[0].legend(); axes[0].grid(alpha=0.3)

for struct, marker in [("dnn", "o"), ("gcn", "s")]:
    axes[1].scatter(piv["size"], piv[struct], marker=marker, alpha=0.7, label=struct)
axes[1].set_xlabel("subset size"); axes[1].set_ylabel("v(S) mean R2")
axes[1].set_title("Random subsets: v(S) vs size"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].scatter(piv["size"], piv["gcn"] - piv["dnn"], c="tab:red", alpha=0.7)
axes[2].axhline(0, color="gray", lw=1)
med_sigma = nf_df["v_std"].median()
axes[2].axhline(med_sigma, color="gray", lw=1, ls="--", label=f"median noise floor ±{med_sigma:.3f}")
axes[2].axhline(-med_sigma, color="gray", lw=1, ls="--")
axes[2].set_xlabel("subset size"); axes[2].set_ylabel("v_gcn - v_dnn")
axes[2].set_title("GCN advantage vs subset size"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / "outputs/phase0_plots.png", dpi=150)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\n随机评测（按尺寸排序，前 15 行）:")
print(piv.head(15).to_string(index=False))
