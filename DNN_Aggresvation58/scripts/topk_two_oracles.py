#!/usr/bin/env python3
"""DNN58：5 个模型的 Shapley 排名，分别放到两种推断模型(oracle)上测 top-k 准确率(R²)。

oracle A = DNN(44→12)；oracle B = gcn_dynamic(置零，全字段准确率最高)。
每个模型给出自己的 Shapley 排名（用自己的 v(S)），top-k 的 12-conf 平均 R² 一律用
同一个 oracle 测 → 纵轴可比。出两张图，看"模型间排名有无差异"是否与 oracle 无关。
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
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
N_PERM = 50


def per_conf_r2(p, t):
    ss = ((t - p) ** 2).sum(0); st = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


class DNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, n_out))

    def forward(self, x): return self.net(x)


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]; nG = di["n_general"]
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                                n_general=nG, bipartite=True)

    def load_gcn(path, mode):
        mdl = InferenceDrivenGNN(
            metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"], n_general=nG,
            confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
            attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
            attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
            alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
            gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
            input_encoder="mlp", bipartite=True, unified_aggregation=mode).to(DEV)
        mdl.load_state_dict(torch.load(path, map_location=DEV)); mdl.eval(); return mdl

    tr, te = di["train_data"], di["test_data"]
    XtrD = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    YtrD = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    XteD = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV); YteD = te[:, ci]
    torch.manual_seed(42); dnn = DNN(nG, len(ci)).to(DEV)
    opt = torch.optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=5e-4)
    best = 1e9; bs = None; pat = 0
    for ep in range(500):
        dnn.train(); pm = torch.randperm(len(XtrD), device=DEV)
        for k in range(0, len(XtrD), 128):
            ix = pm[k:k+128]; opt.zero_grad()
            (((dnn(XtrD[ix]) - YtrD[ix]) ** 2).mean()).backward(); opt.step()
        dnn.eval()
        with torch.no_grad():
            v = float(((dnn(XteD).cpu().numpy() - YteD) ** 2).mean())
        if v < best: best = v; bs = deepcopy(dnn.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= 150: break
    dnn.load_state_dict(bs); dnn.eval()

    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    Xg = torch.as_tensor(feat, dtype=torch.float32, device=DEV); Yg = tgt

    def dnn_vec(S):
        mask = torch.zeros(nG, device=DEV)
        if S: mask[list(S)] = 1.0
        with torch.no_grad(): p = dnn(XteD * mask).cpu().numpy()
        return per_conf_r2(p, YteD)

    def gcn_vec(mdl, S):
        Xs = torch.zeros_like(Xg); on = gi[list(S)]
        if len(on): Xs[:, on, :] = Xg[:, on, :]
        with torch.no_grad(): p = mdl(Xs).cpu().numpy()
        return per_conf_r2(p, Yg)

    gcn_noprior = load_gcn(str(ROOT / "old_model/gcn_noprior.pt"), "gcn_noprior")
    gcn_static = load_gcn(str(ROOT / "old_model/gcn_static.pt"), "gcn_static")
    gcn_dyn = load_gcn(str(ROOT / "old_model/gcn_dynamic.pt"), "gcn_dynamic")
    gcn_masked = load_gcn(str(ROOT / "trained/gcn_dynamic_masked.pt"), "gcn_dynamic")

    own = {
        "DNN(44->12)": lambda S: float(dnn_vec(S).mean()),
        "gcn_noprior": lambda S: float(gcn_vec(gcn_noprior, S).mean()),
        "gcn_static": lambda S: float(gcn_vec(gcn_static, S).mean()),
        "gcn_dynamic(zero-mask)": lambda S: float(gcn_vec(gcn_dyn, S).mean()),
        "gcn_dynamic(masked)": lambda S: float(gcn_vec(gcn_masked, S).mean()),
    }

    def shapley(leak):
        rng = np.random.default_rng(42); phi = np.zeros(nG); v0 = leak(set())
        for _ in range(N_PERM):
            perm = rng.permutation(nG); act = set(); prev = v0
            for idx in perm:
                act.add(int(idx)); v = leak(act); phi[idx] += v - prev; prev = v
        return phi / N_PERM

    print("计算 5 个模型的 Shapley 排名 ...")
    orders = {name: np.argsort(-shapley(leak)) for name, leak in own.items()}

    oracles = {
        "oracle=DNN(44->12)": (lambda S: float(dnn_vec(S).mean()), "topk_oracle_dnn"),
        "oracle=gcn_dynamic(best acc)": (lambda S: float(gcn_vec(gcn_dyn, S).mean()), "topk_oracle_gcndyn"),
    }
    COL = {"DNN(44->12)": "#c0392b", "gcn_noprior": "#9e9e9e", "gcn_static": "#76b7b2",
           "gcn_dynamic(zero-mask)": "#4c72b0", "gcn_dynamic(masked)": "#59a14f"}
    ks = list(range(1, nG + 1))

    for oname, (oracle, fn) in oracles.items():
        curves = {name: [oracle(set(orders[name][:k].tolist())) for k in ks] for name in own}
        pd.DataFrame({"k": ks, **curves}).to_csv(ROOT / "outputs" / f"{fn}.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for name in own:
            ax.plot(ks, curves[name], "-", color=COL[name], lw=2.2, label=name)
        ax.set_xlabel("top-k general fields revealed (ranked by each model's Shapley)")
        ax.set_ylabel("12-conf mean R2 (inference accuracy)")
        ax.set_title(f"DNN58 top-k accuracy — all 5 rankings measured on  {oname}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / "outputs" / f"{fn}.png", dpi=140); plt.close(fig)
        print(f"\n=== {oname} — top-k 平均 R² ===")
        print("  k   " + "".join(f"{n[:11]:>13s}" for n in own))
        for kk in [1, 2, 4, 8, 16, 24]:
            print(f"  {kk:<3d} " + "".join(f"{curves[n][kk-1]:>13.3f}" for n in own))
        print(f"  图: outputs/{fn}.png")


if __name__ == "__main__":
    main()
