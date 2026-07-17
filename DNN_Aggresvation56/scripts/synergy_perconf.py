#!/usr/bin/env python3
"""DNN56 协同泄露（逐 confidential）：找"两个字段单看都低、合起来把某个 confidential 推出来"。

对每个 general 字段对 (g1,g2) 和每个 confidential c：
  synergy_c = R2_c(only {g1,g2}) − max( R2_c(only g1), R2_c(only g2) )
取每对的最强 confidential。单字段法给不了这个（它按字段单独打分）。
oracle = 最佳推断模型 gcn_dynamic。
"""
from __future__ import annotations

import itertools, json, sys
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


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]
    em = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                         n_general=di["n_general"], bipartite=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=em, n_nodes=di["n_nodes"], n_general=di["n_general"],
        confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
        attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
        attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
        alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
        gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
        input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(dev)
    model.load_state_dict(torch.load(ROOT / "inference_model/model.pt", map_location=dev))
    model.eval()
    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    X = torch.as_tensor(feat, dtype=torch.float32, device=dev)
    y = torch.as_tensor(tgt, dtype=torch.float32, device=dev)
    gi = np.array(di["general_indices"]); gn = di["general"]; cn = di["confidential"]; nG = di["n_general"]

    def r2vec(active):
        Xs = torch.zeros_like(X); Xs[:, gi[list(active)], :] = X[:, gi[list(active)], :]
        with torch.no_grad():
            p = model(Xs).cpu().numpy()
        t = y.cpu().numpy(); ss = ((t - p) ** 2).sum(0); st = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
        return np.clip(1 - ss / st, 0, None)

    sing = np.array([r2vec({i}) for i in range(nG)])              # (nG,C)
    rows = []
    for i, j in itertools.combinations(range(nG), 2):
        pr = r2vec({i, j})
        syn = pr - np.maximum(sing[i], sing[j])
        c = int(syn.argmax())
        rows.append({"g1": gn[i], "g2": gn[j], "confidential": cn[c],
                     "single1": round(float(sing[i][c]), 3), "single2": round(float(sing[j][c]), 3),
                     "pair": round(float(pr[c]), 3), "synergy": round(float(syn[c]), 3)})
    df = pd.DataFrame(rows).sort_values("synergy", ascending=False).reset_index(drop=True)
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    df.to_csv(out / "synergy_perconf.csv", index=False)

    vals = df["synergy"].to_numpy()
    summary = {"n_pairs": len(df), "synergy_median": float(np.median(vals)),
               "synergy_max": float(vals.max()), "n>0.2": int((vals > 0.2).sum()),
               "n>0.3": int((vals > 0.3).sum()),
               "n_both_single<0.1_pair>0.2": int(((df["single1"] < 0.1) & (df["single2"] < 0.1)
                                                  & (df["pair"] > 0.2)).sum())}
    (out / "summary_perconf.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    top = df.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    yy = np.arange(len(top)); h = 0.26
    ax.barh(yy - h, top["single1"], h, color="#bbbbbb", label="field1 alone")
    ax.barh(yy, top["single2"], h, color="#888888", label="field2 alone")
    ax.barh(yy + h, top["pair"], h, color="#c0392b", label="pair together")
    labels = [f"{r.confidential[:20]}  ←  {r.g1[:16]} + {r.g2[:16]}" for r in top.itertuples()]
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("inference R2 of the confidential field"); ax.set_xlim(0, 1)
    ax.set_title("DNN56: synergistic leakage — pair reconstructs confidential that neither field leaks alone")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "synergy_perconf.png", dpi=140); plt.close(fig)

    print("=== 逐 confidential 协同泄露 top12（单看都低、合起来高）===")
    print(f"{'confidential':22s}{'field1':20s}{'field2':18s}{'s1':>6s}{'s2':>6s}{'pair':>6s}{'syn':>6s}")
    for r in df.head(12).itertuples():
        print(f"{r.confidential[:21]:22s}{r.g1[:19]:20s}{r.g2[:17]:18s}"
              f"{r.single1:>6.2f}{r.single2:>6.2f}{r.pair:>6.2f}{r.synergy:>6.2f}")
    print(f"\n协同 中位数={summary['synergy_median']:.3f} 最大={summary['synergy_max']:.3f} "
          f">0.2:{summary['n>0.2']} 对  >0.3:{summary['n>0.3']} 对  "
          f"(两 single<0.1 且 pair>0.2):{summary['n_both_single<0.1_pair>0.2']} 对")
    print(f"图: {out}/synergy_perconf.png ; 表: synergy_perconf.csv")


if __name__ == "__main__":
    main()
