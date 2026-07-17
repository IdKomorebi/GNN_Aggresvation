#!/usr/bin/env python3
"""DNN57：DNN(44→12) vs GCN(noprior/static/dynamic)，比①准确率、②Shapley 敏感度一致性。

- 推断模型 4 个：
    dnn         : 朴素 MLP 44→12（现训）
    gcn_noprior / gcn_static / gcn_dynamic : 复用 DNN52 已训模型（mlp 编码）
- 敏感度：对每个模型做蒙特卡洛 Shapley（value(S)=只给 general 子集 S 时 a_c 加权 R²）。
- 对比：4 个模型的 12-conf 平均 R²（对照 DNN single probe 0.8937）；
        4 个模型 Shapley 敏感度排名的 Spearman 一致性 + top-k 重合。
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
from scipy.stats import spearmanr

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
PROBE = 0.8937


def per_conf_r2_np(pred, tgt):
    ss = ((tgt - pred) ** 2).sum(0); st = ((tgt - tgt.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss / st


class DNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def train_dnn(Xtr, Ytr, Xte, Yte):
    torch.manual_seed(42)
    model = DNN(Xtr.shape[1], Ytr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32, device=DEV)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=DEV)
    Yte_t = torch.as_tensor(Yte, dtype=torch.float32, device=DEV)
    best = float("inf"); best_state = None; patience = 0
    n = len(Xtr_t)
    for ep in range(500):
        model.train(); perm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            idx = perm[k:k + 128]
            opt.zero_grad()
            loss = ((model(Xtr_t[idx]) - Ytr_t[idx]) ** 2).mean()
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            te = ((model(Xte_t) - Yte_t) ** 2).mean().item()
        if te < best:
            best = te; best_state = deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= 150:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    gnames = di["general"]; nG = di["n_general"]
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                                n_general=di["n_general"], bipartite=True)

    # ---- DNN 数据（general → confidential，window=1）----
    tr, te = di["train_data"], di["test_data"]
    Xtr, Ytr = tr[:, gi], tr[:, ci]; Xte, Yte = te[:, gi], te[:, ci]
    print(f"训练 DNN(44→12) ...")
    dnn = train_dnn(Xtr, Ytr, Xte, Yte)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=DEV)

    def dnn_r2vec(active):
        mask = torch.zeros(nG, device=DEV); mask[list(active)] = 1.0
        with torch.no_grad():
            p = dnn(Xte_t * mask).cpu().numpy()
        return np.clip(per_conf_r2_np(p, Yte), 0, None)

    # ---- GCN：加载 3 个，构造各自的 r2vec ----
    def load_gcn(mode):
        model = InferenceDrivenGNN(
            metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"], n_general=di["n_general"],
            confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
            attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
            attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
            alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
            gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
            input_encoder="mlp", bipartite=True, unified_aggregation=mode).to(DEV)
        model.load_state_dict(torch.load(ROOT / f"gcn_models/{mode}.pt", map_location=DEV))
        model.eval()
        return model

    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    Xg = torch.as_tensor(feat, dtype=torch.float32, device=DEV)
    gcns = {mode: load_gcn(mode) for mode in ["gcn_noprior", "gcn_static", "gcn_dynamic"]}

    def gcn_r2vec_factory(model):
        def f(active):
            Xs = torch.zeros_like(Xg); on = gi[list(active)]
            if len(on): Xs[:, on, :] = Xg[:, on, :]
            with torch.no_grad():
                p = model(Xs).cpu().numpy()
            return np.clip(per_conf_r2_np(p, Yte), 0, None)
        return f

    r2vecs = {"dnn": dnn_r2vec}
    for mode, mdl in gcns.items():
        r2vecs[mode] = gcn_r2vec_factory(mdl)

    # ---- 准确率（全 general）----
    order = ["dnn", "gcn_noprior", "gcn_static", "gcn_dynamic"]
    full = {name: r2vecs[name](set(range(nG))) for name in order}
    acc = {name: float(full[name].mean()) for name in order}

    # ---- 蒙特卡洛 Shapley（每个模型用自己的 a_c 加权）----
    def shapley(r2vec, a_c):
        aw = a_c / a_c.sum()
        rng = np.random.default_rng(42)
        phi = np.zeros(nG)
        v_empty = float((aw * r2vec(set())).sum())
        for _ in range(N_PERM):
            perm = rng.permutation(nG); active = set(); prev = v_empty
            for idx in perm:
                active.add(int(idx))
                v = float((aw * r2vec(active)).sum())
                phi[idx] += v - prev; prev = v
        return phi / N_PERM

    print("计算 Shapley ...")
    shap = {name: shapley(r2vecs[name], full[name]) for name in order}
    sdf = pd.DataFrame({"general_field": gnames, **{name: shap[name] for name in order}})
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    sdf.to_csv(out / "shapley_by_model.csv", index=False)

    # ---- 一致性：Spearman + top-10 重合 ----
    sp = np.ones((4, 4))
    for a in range(4):
        for b in range(4):
            sp[a, b] = spearmanr(shap[order[a]], shap[order[b]]).correlation
    top10 = {name: set(np.argsort(-shap[name])[:10]) for name in order}
    ov = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            ov[a, b] = len(top10[order[a]] & top10[order[b]]) / 10.0

    summary = {"accuracy": acc, "probe": PROBE,
               "spearman": {order[a]: {order[b]: round(float(sp[a, b]), 3) for b in range(4)} for a in range(4)},
               "top10_overlap": {order[a]: {order[b]: round(float(ov[a, b]), 2) for b in range(4)} for a in range(4)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 图1：准确率 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [acc[n] for n in order]
    ax.bar(order, vals, color=["#c0392b", "#9e9e9e", "#76b7b2", "#59a14f"])
    ax.axhline(PROBE, ls=":", color="#333", lw=1.3, label=f"DNN single probe = {PROBE}")
    for i, v in enumerate(vals): ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.84, 0.905); ax.set_ylabel("mean test R2 over 12 confidential")
    ax.set_title("DNN57: accuracy — DNN(44→12) vs GCN variants vs probe")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "accuracy.png", dpi=140); plt.close(fig)

    # ---- 图2：Shapley 一致性热图 ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sp, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(order, fontsize=8)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{sp[a,b]:.2f}", ha="center", va="center",
                    color="white" if sp[a, b] < 0.85 else "black", fontsize=9)
    fig.colorbar(im, label="Spearman of Shapley sensitivity")
    ax.set_title("DNN57: sensitivity ranking consistency across models")
    fig.tight_layout(); fig.savefig(out / "consistency_heatmap.png", dpi=140); plt.close(fig)

    # ---- 打印 ----
    print("\n=== ① 准确率 (12-conf 平均 R², DNN single probe=0.8937) ===")
    for n in order:
        print(f"  {n:14s} {acc[n]:.4f}   距probe {acc[n]-PROBE:+.4f}")
    print("\n=== ② Shapley 敏感度一致性 (Spearman) ===")
    print(" " * 14 + "".join(f"{n[:11]:>13s}" for n in order))
    for a in range(4):
        print(f"{order[a]:14s}" + "".join(f"{sp[a,b]:>13.3f}" for b in range(4)))
    print("\n   top-10 敏感字段重合率:")
    print(" " * 14 + "".join(f"{n[:11]:>13s}" for n in order))
    for a in range(4):
        print(f"{order[a]:14s}" + "".join(f"{ov[a,b]:>13.2f}" for b in range(4)))
    print(f"\n图: {out}/accuracy.png, consistency_heatmap.png ; 表: shapley_by_model.csv")


if __name__ == "__main__":
    main()
