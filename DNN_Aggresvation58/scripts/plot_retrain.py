#!/usr/bin/env python3
"""DNN58：画 retrain-based v(top-k) 两条曲线(DNN 结构 vs GCN+dynamic 结构)，
叠加"遮蔽近似"曲线做对照，看省算力的近似离重训 ground-truth 差多少。"""
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


def per_conf_r2(p, t):
    ss = ((t - p) ** 2).sum(0); st = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    R = ROOT / "outputs/retrain"
    dnn = [json.load(open(R / f"dnn_k{k}.json"))["mean_r2"] if (R / f"dnn_k{k}.json").exists() else np.nan for k in KS]
    gcn = [json.load(open(R / f"gcn_k{k}.json"))["mean_r2"] if (R / f"gcn_k{k}.json").exists() else np.nan for k in KS]

    # ---- 遮蔽近似对照：用遮蔽版 & 置零版 gcn_dynamic 的 v(top-k) ----
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]; nG = di["n_general"]; gi = np.array(di["general_indices"])
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                                n_general=nG, bipartite=True)
    order = json.load(open(R / "ranking.json"))["order"]

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

    masked = load(ROOT / "trained/gcn_dynamic_masked.pt")
    zeromask = load(ROOT / "old_model/gcn_dynamic.pt")
    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    Xg = torch.as_tensor(feat, dtype=torch.float32, device=DEV)

    def vtopk(mdl, k):
        Xs = torch.zeros_like(Xg); on = gi[order[:k]]
        Xs[:, on, :] = Xg[:, on, :]
        with torch.no_grad(): p = mdl(Xs).cpu().numpy()
        return float(per_conf_r2(p, tgt).mean())
    v_masked = [vtopk(masked, k) for k in KS]
    v_zero = [vtopk(zeromask, k) for k in KS]

    pd.DataFrame({"k": KS, "retrain_dnn": dnn, "retrain_gcn": gcn,
                  "approx_masked": v_masked, "approx_zeromask": v_zero}).to_csv(
        ROOT / "outputs/retrain/retrain_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(KS, dnn, "-o", color="#c0392b", lw=2.5, label="RETRAIN — DNN structure")
    ax.plot(KS, gcn, "-o", color="#4c72b0", lw=2.5, label="RETRAIN — GCN+dynamic structure")
    ax.plot(KS, v_masked, "--s", color="#59a14f", lw=1.8, alpha=0.8, label="approx — masked-trained v(S)")
    ax.plot(KS, v_zero, ":^", color="#9e9e9e", lw=1.8, alpha=0.8, label="approx — zero-mask v(S)")
    ax.set_xlabel("k = number of top-sensitive general fields (retrained each time)")
    ax.set_ylabel("12-conf mean R2 (inference accuracy)")
    ax.set_title("DNN58: TRUE retrain-based v(top-k) — DNN vs GCN structure (+ mask approximations)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/retrain/retrain_curves.png", dpi=140); plt.close(fig)

    print("=== retrain-based v(top-k) ===")
    print(f"  {'k':>4s}{'DNN(retrain)':>14s}{'GCN(retrain)':>14s}{'masked-approx':>15s}{'zero-approx':>13s}")
    for i, k in enumerate(KS):
        print(f"  {k:>4d}{dnn[i]:>14.4f}{gcn[i]:>14.4f}{v_masked[i]:>15.4f}{v_zero[i]:>13.4f}")
    print(f"\n图: outputs/retrain/retrain_curves.png")


if __name__ == "__main__":
    main()
