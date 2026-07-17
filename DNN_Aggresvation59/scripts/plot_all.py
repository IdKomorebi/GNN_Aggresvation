#!/usr/bin/env python3
"""DNN59 出图：
  fig1 approx_vs_retrain.png   —— 遮蔽排名下：重训(DNN/GCN) vs 遮蔽近似 vs 置零近似
  fig2 topk_dnn_5methods.png   —— DNN 结构下，5 个方法(排名)各自的 top-k R²
  fig3 topk_gcndyn_5methods.png—— GCN+dynamic 结构下，5 个方法的 top-k R²
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 44]
RANKINGS = ["dnn", "noprior", "static", "zeromask", "masked"]
RLAB = {"dnn": "DNN Shapley", "noprior": "gcn_noprior Shapley", "static": "gcn_static Shapley",
        "zeromask": "gcn_dynamic(zero) Shapley", "masked": "gcn_dynamic(masked) Shapley"}
COL = {"dnn": "#c0392b", "noprior": "#9e9e9e", "static": "#76b7b2",
       "zeromask": "#4c72b0", "masked": "#59a14f"}
R = ROOT / "outputs/retrain"


def per_conf_r2(p, t):
    ss = ((t - p) ** 2).sum(0); st = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def get(ranking, struct, k):
    f = R / f"{ranking}_{struct}_k{k}.json"
    return json.load(open(f))["mean_r2"] if f.exists() else np.nan


def main():
    # 汇总表
    rows = []
    for rk in RANKINGS:
        for st in ["dnn", "gcn"]:
            for k in KS:
                rows.append({"ranking": rk, "struct": st, "k": k, "mean_r2": get(rk, st, k)})
    pd.DataFrame(rows).to_csv(R / "all_retrain.csv", index=False)

    # ---- fig2: DNN 结构下 5 方法 ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for rk in RANKINGS:
        ax.plot(KS, [get(rk, "dnn", k) for k in KS], "-o", color=COL[rk], lw=2.2, label=RLAB[rk])
    ax.set_xlabel("k = top-k sensitive fields (DNN retrained each k)")
    ax.set_ylabel("12-conf mean R2 (inference accuracy)")
    ax.set_title("DNN59: top-k accuracy on DNN structure — 5 ranking methods")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/topk_dnn_5methods.png", dpi=140); plt.close(fig)

    # ---- fig3: GCN+dynamic 结构下 5 方法 ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for rk in RANKINGS:
        ax.plot(KS, [get(rk, "gcn", k) for k in KS], "-o", color=COL[rk], lw=2.2, label=RLAB[rk])
    ax.set_xlabel("k = top-k sensitive fields (GCN+dynamic retrained each k)")
    ax.set_ylabel("12-conf mean R2 (inference accuracy)")
    ax.set_title("DNN59: top-k accuracy on GCN+dynamic structure — 5 ranking methods")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/topk_gcndyn_5methods.png", dpi=140); plt.close(fig)

    # ---- fig1: 遮蔽排名下 重训 vs 两个 approx ----
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]; nG = di["n_general"]; gi = np.array(di["general_indices"])
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                                n_general=nG, bipartite=True)

    def load(path):
        mdl = InferenceDrivenGNN(
            metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"], n_general=nG,
            confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
            attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
            attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
            alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
            gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
            input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(DEV)
        mdl.load_state_dict(torch.load(path, map_location=DEV)); mdl.eval(); return mdl

    masked = load(ROOT / "approx_models/gcn_dynamic_masked.pt")
    zero = load(ROOT / "approx_models/gcn_dynamic_zeromask.pt")
    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    Xg = torch.as_tensor(feat, dtype=torch.float32, device=DEV)
    order_fields = json.load(open(R / "ranking_masked.json"))["order_fields"]
    name2local = {n: i for i, n in enumerate(di["general"])}
    order = [name2local[f] for f in order_fields]

    def vtopk(mdl, k):
        Xs = torch.zeros_like(Xg); on = gi[order[:k]]; Xs[:, on, :] = Xg[:, on, :]
        with torch.no_grad(): p = mdl(Xs).cpu().numpy()
        return float(per_conf_r2(p, tgt).mean())

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(KS, [get("masked", "dnn", k) for k in KS], "-o", color="#c0392b", lw=2.5, label="RETRAIN — DNN structure")
    ax.plot(KS, [get("masked", "gcn", k) for k in KS], "-o", color="#4c72b0", lw=2.5, label="RETRAIN — GCN+dynamic structure")
    ax.plot(KS, [vtopk(masked, k) for k in KS], "--s", color="#59a14f", lw=1.8, alpha=0.85, label="approx — masked-trained v(S)")
    ax.plot(KS, [vtopk(zero, k) for k in KS], ":^", color="#9e9e9e", lw=1.8, alpha=0.85, label="approx — zero-mask v(S)")
    ax.set_xlabel("k = top-k sensitive fields (masked ranking)")
    ax.set_ylabel("12-conf mean R2 (inference accuracy)")
    ax.set_title("DNN59: retrain ground-truth vs two v(S) approximations (masked ranking)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/approx_vs_retrain.png", dpi=140); plt.close(fig)

    # 打印两张主图的表
    for st, tag in [("dnn", "DNN 结构"), ("gcn", "GCN+dynamic 结构")]:
        print(f"\n=== {tag} 下 5 方法 top-k 平均 R² ===")
        print("  k   " + "".join(f"{rk[:9]:>11s}" for rk in RANKINGS))
        for k in KS:
            print(f"  {k:<3d} " + "".join(f"{get(rk, st, k):>11.4f}" for rk in RANKINGS))
    print(f"\n图: outputs/topk_dnn_5methods.png, topk_gcndyn_5methods.png, approx_vs_retrain.png")


if __name__ == "__main__":
    main()
