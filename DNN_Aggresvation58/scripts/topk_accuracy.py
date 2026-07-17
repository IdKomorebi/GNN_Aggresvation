#!/usr/bin/env python3
"""DNN58 公平 top-k：5 个模型各自的 Shapley 排名，全部在同一把尺子上评测。

关键修正：topk_5models 里每条曲线用"各自的 v(S)"，而置零版 v(S) 在小子集会塌陷，
遮蔽版 v(S) 忠实——纵轴不可比，绿线的"前期优势"是测量伪影。
本脚本：每个模型给出自己的 Shapley 排名（模型属性），但 top-k 泄露一律用
**遮蔽版的忠实 v(S)** 来量（共同 oracle），纵轴统一，隔离出"排名质量"这一个变量。
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

    def dnn_leak(active):
        mask = torch.zeros(nG, device=DEV)
        if active: mask[list(active)] = 1.0
        with torch.no_grad(): p = dnn(XteD * mask).cpu().numpy()
        return per_conf_r2(p, YteD)

    def gcn_leak_vec(mdl, active):
        Xs = torch.zeros_like(Xg); on = gi[list(active)]
        if len(on): Xs[:, on, :] = Xg[:, on, :]
        with torch.no_grad(): p = mdl(Xs).cpu().numpy()
        return per_conf_r2(p, Yg)

    gcn_noprior = load_gcn(str(ROOT / "old_model/gcn_noprior.pt"), "gcn_noprior")
    gcn_static = load_gcn(str(ROOT / "old_model/gcn_static.pt"), "gcn_static")
    gcn_dyn = load_gcn(str(ROOT / "old_model/gcn_dynamic.pt"), "gcn_dynamic")
    gcn_masked = load_gcn(str(ROOT / "trained/gcn_dynamic_masked.pt"), "gcn_dynamic")

    # 每个模型"自己的 leak"（算它自己的 Shapley 排名用），a_c 用自己的全字段
    def make_leak(kind, mdl=None):
        if kind == "dnn":
            ac = dnn_leak(set(range(nG))); aw = ac / ac.sum()
            return lambda S: float(dnn_leak(S).mean())
        ac = gcn_leak_vec(mdl, set(range(nG))); aw = ac / ac.sum()
        return lambda S: float(gcn_leak_vec(mdl, S).mean())

    own = {"DNN(44->12)": make_leak("dnn"),
           "gcn_noprior": make_leak("g", gcn_noprior),
           "gcn_static": make_leak("g", gcn_static),
           "gcn_dynamic(zero-mask)": make_leak("g", gcn_dyn),
           "gcn_dynamic(masked)": make_leak("g", gcn_masked)}

    # 共同 oracle = 遮蔽版忠实 v(S)
    ac_m = gcn_leak_vec(gcn_masked, set(range(nG))); aw_m = ac_m / ac_m.sum()
    oracle = lambda S: float((gcn_leak_vec(gcn_masked, S)).mean())

    def shapley(leak):
        rng = np.random.default_rng(42); phi = np.zeros(nG); v0 = leak(set())
        for _ in range(N_PERM):
            perm = rng.permutation(nG); act = set(); prev = v0
            for idx in perm:
                act.add(int(idx)); v = leak(act); phi[idx] += v - prev; prev = v
        return phi / N_PERM

    ks = list(range(1, nG + 1)); curves = {}
    for name, leak in own.items():
        print(f"  {name} ...")
        order = np.argsort(-shapley(leak))                 # 该模型自己的排名
        curves[name] = [oracle(set(order[:k].tolist())) for k in ks]   # 一律用共同 oracle 评

    out = ROOT / "outputs"
    pd.DataFrame({"k": ks, **curves}).to_csv(out / "topk_accuracy.csv", index=False)
    COL = {"DNN(44->12)": "#c0392b", "gcn_noprior": "#9e9e9e", "gcn_static": "#76b7b2",
           "gcn_dynamic(zero-mask)": "#4c72b0", "gcn_dynamic(masked)": "#59a14f"}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name in own:
        ax.plot(ks, curves[name], "-", color=COL[name], lw=2.2, label=name)
    ax.set_xlabel("top-k general fields revealed (ranked by each model's Shapley)")
    ax.set_ylabel("12-conf mean R2 (inference accuracy), all on faithful masked model")
    ax.set_title("DNN58 top-k inference accuracy (R2): 5 models, common faithful oracle")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "topk_accuracy.png", dpi=140); plt.close(fig)
    print("\n=== FAIR top-k (共同 oracle=遮蔽版) 前几个 k ===")
    print("  k   " + "  ".join(f"{n[:10]:>11s}" for n in own))
    for kk in [1, 2, 4, 8, 16]:
        print(f"  {kk:<3d} " + "  ".join(f"{curves[n][kk-1]:>11.3f}" for n in own))
    print(f"\n图: {out}/topk_accuracy.png")


if __name__ == "__main__":
    main()
