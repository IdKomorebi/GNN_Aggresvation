#!/usr/bin/env python3
"""DNN58 五模型 top-k 对比：DNN(44→12) / gcn_noprior / gcn_static / gcn_dynamic(置零) /
gcn_dynamic(遮蔽)。每个模型用【自己的 Shapley 排名】在【自己的 v(S)】上做 top-k。
"""
from __future__ import annotations

import json, sys
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

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
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

    # ---- 数据 ----
    tr, te = di["train_data"], di["test_data"]
    XtrD = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    YtrD = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    XteD = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV)
    YteD_np = te[:, ci]
    fte, tte = _build_windowed_samples(te, di["confidential_indices"], 1)
    Xg = torch.as_tensor(fte, dtype=torch.float32, device=DEV); Yg = tte

    # ---- 训练 DNN(44→12) ----
    torch.manual_seed(42); dnn = DNN(nG, len(ci)).to(DEV)
    opt = torch.optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
    best = 1e9; bstate = None; pat = 0; n = len(XtrD)
    for ep in range(500):
        dnn.train(); perm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            idx = perm[k:k + 128]; opt.zero_grad()
            ((dnn(XtrD[idx]) - YtrD[idx]) ** 2).mean().backward(); opt.step()
        sched.step(); dnn.eval()
        with torch.no_grad(): tl = ((dnn(XteD) - torch.as_tensor(YteD_np, dtype=torch.float32, device=DEV)) ** 2).mean().item()
        if tl < best: best = tl; bstate = deepcopy(dnn.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= 150: break
    dnn.load_state_dict(bstate); dnn.eval()

    def load_gcn(path):
        mdl = InferenceDrivenGNN(
            metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"], n_general=nG,
            confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
            attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
            attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
            alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
            gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
            input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic" if "dynamic" in path
            else ("gcn_static" if "static" in path else "gcn_noprior")).to(DEV)
        mdl.load_state_dict(torch.load(path, map_location=DEV)); mdl.eval(); return mdl

    # ---- 各模型的 leak(S) ----
    def dnn_leak_factory():
        with torch.no_grad(): ac = per_conf_r2(dnn(XteD).cpu().numpy(), YteD_np)
        aw = ac / ac.sum()
        def leak(active):
            mask = torch.zeros(nG, device=DEV)
            if active: mask[list(active)] = 1.0
            with torch.no_grad(): p = dnn(XteD * mask).cpu().numpy()
            return float((aw * per_conf_r2(p, YteD_np)).sum())
        return leak
    def gcn_leak_factory(mdl):
        with torch.no_grad(): ac = per_conf_r2(mdl(Xg).cpu().numpy(), Yg)
        aw = ac / ac.sum()
        def leak(active):
            Xs = torch.zeros_like(Xg); on = gi[list(active)]
            if len(on): Xs[:, on, :] = Xg[:, on, :]
            with torch.no_grad(): p = mdl(Xs).cpu().numpy()
            return float((aw * per_conf_r2(p, Yg)).sum())
        return leak

    models = {
        "DNN(44->12)": dnn_leak_factory(),
        "gcn_noprior": gcn_leak_factory(load_gcn(str(ROOT / "old_model/gcn_noprior.pt"))),
        "gcn_static": gcn_leak_factory(load_gcn(str(ROOT / "old_model/gcn_static.pt"))),
        "gcn_dynamic(zero-mask)": gcn_leak_factory(load_gcn(str(ROOT / "old_model/gcn_dynamic.pt"))),
        "gcn_dynamic(masked)": gcn_leak_factory(load_gcn(str(ROOT / "trained/gcn_dynamic_masked.pt"))),
    }

    def shapley(leak):
        rng = np.random.default_rng(42); phi = np.zeros(nG); v0 = leak(set())
        for _ in range(N_PERM):
            perm = rng.permutation(nG); act = set(); prev = v0
            for idx in perm:
                act.add(int(idx)); v = leak(act); phi[idx] += v - prev; prev = v
        return phi / N_PERM

    ks = list(range(1, nG + 1)); curves = {}; ub = {}
    for name, leak in models.items():
        print(f"  {name} ...")
        sh = shapley(leak); order = np.argsort(-sh)
        curves[name] = [leak(set(order[:k].tolist())) for k in ks]
        ub[name] = leak(set(range(nG)))

    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    pd.DataFrame({"k": ks, **{n: curves[n] for n in curves}}).to_csv(out / "topk_5models.csv", index=False)
    smallk = {n: float(np.mean(curves[n][:10])) for n in curves}
    (out / "topk_5models_summary.json").write_text(json.dumps(
        {"upper_bounds": ub, "smallk_first10": smallk}, indent=2, ensure_ascii=False))

    COL = {"DNN(44->12)": "#c0392b", "gcn_noprior": "#9e9e9e", "gcn_static": "#76b7b2",
           "gcn_dynamic(zero-mask)": "#4c72b0", "gcn_dynamic(masked)": "#59a14f"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in models:
        lw = 2.6 if "masked" in name else 1.9
        ax.plot(ks, curves[name], "-", color=COL[name], lw=lw, label=f"{name} (UB={ub[name]:.3f})")
    ax.set_xlabel("top-k general fields revealed (ranked by each model's Shapley)")
    ax.set_ylabel("leakage v(top-k)  (a_c weighted R2, each on its own v(S))")
    ax.set_title("DNN58: top-k of 5 models (4 zero-mask + 1 masked-trained)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(1, nG)
    fig.tight_layout(); fig.savefig(out / "topk_5models.png", dpi=140); plt.close(fig)

    print("\n=== 五模型 top-k 前10平均 + 满字段上界 ===")
    print(f"{'model':26s}{'top10-mean':>12s}{'upper':>9s}")
    for n in models:
        print(f"{n:26s}{smallk[n]:>12.4f}{ub[n]:>9.4f}")
    print(f"\n图: {out}/topk_5models.png ; 表: topk_5models.csv")


if __name__ == "__main__":
    main()
