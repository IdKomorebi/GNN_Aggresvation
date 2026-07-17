#!/usr/bin/env python3
"""DNN58 top-k：在遮蔽训练的忠实 oracle 上，比多种敏感度方法的 top-k 推断曲线。

方法(对 gcn_dynamic 适用的)：mask(必要性)、single(充分性)、shapley、ig(积分梯度)、
lrp(输入×梯度)、gate(动态门控)、random(基线)。
排名后取 top-k general 字段(其余置 0)，在遮蔽版上测 12-conf a_c 加权 R²(忠实 v(topk))。
上界 = 满字段 leakage。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def per_conf_r2(pred, tgt):
    ss = ((tgt - pred) ** 2).sum(0); st = ((tgt - tgt.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]; nG = di["n_general"]
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"]); gnames = di["general"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                                n_general=nG, bipartite=True)
    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"], n_general=nG,
        confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
        attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
        attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
        alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
        gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
        input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(DEV)
    model.load_state_dict(torch.load(ROOT / "trained/gcn_dynamic_masked.pt", map_location=DEV)); model.eval()

    fte, tte = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    X = torch.as_tensor(fte, dtype=torch.float32, device=DEV); Y = tte

    with torch.no_grad():
        a_c = per_conf_r2(model(X).cpu().numpy(), Y)
    aw = a_c / a_c.sum(); a_t = torch.as_tensor(a_c, dtype=torch.float32, device=DEV)

    def leak(active):
        Xs = torch.zeros_like(X); on = gi[list(active)]
        if len(on): Xs[:, on, :] = X[:, on, :]
        with torch.no_grad():
            return float((aw * per_conf_r2(model(Xs).cpu().numpy(), Y)).sum())

    base = leak(set(range(nG)))
    print(f"遮蔽版 满字段 leakage(上界) = {base:.4f}")

    # ---------- 各方法敏感度分数 ----------
    scores = {}
    # single（充分性）
    scores["single"] = np.array([leak({j}) for j in range(nG)])
    # mask（必要性）
    allset = set(range(nG))
    scores["mask"] = np.array([base - leak(allset - {j}) for j in range(nG)])
    # shapley
    rng = np.random.default_rng(42); phi = np.zeros(nG); v0 = leak(set())
    for _ in range(N_PERM):
        perm = rng.permutation(nG); act = set(); prev = v0
        for idx in perm:
            act.add(int(idx)); v = leak(act); phi[idx] += v - prev; prev = v
    scores["shapley"] = phi / N_PERM
    # ig（积分梯度）
    steps = 32; baseline = torch.zeros_like(X); total = torch.zeros_like(X)
    for s in range(1, steps + 1):
        Xi = (baseline + (s / steps) * (X - baseline)).clone().requires_grad_(True)
        out = (model(Xi) * a_t).sum(); grad, = torch.autograd.grad(out, Xi); total += grad.detach()
    ig = ((X - baseline) * (total / steps)).abs().mean(0).squeeze(-1).cpu().numpy()
    scores["ig"] = ig[gi]
    # lrp（输入×梯度）
    Xi = X.clone().requires_grad_(True); out = (model(Xi) * a_t).sum()
    grad, = torch.autograd.grad(out, Xi); rel = (Xi * grad).abs().mean(0).squeeze(-1).detach().cpu().numpy()
    scores["lrp"] = rel[gi]
    # gate（动态门控：conf 目标对 general 源的平均门控，a_c 加权）
    with torch.no_grad():
        _ = model(X)
    gate = model.get_unified_gate()
    if gate is not None:
        gate = gate.cpu().numpy()
        scores["gate"] = np.array([(a_c * gate[ci, gi[j]]).sum() for j in range(nG)])
    # random
    scores["random"] = np.random.default_rng(0).random(nG)

    # ---------- top-k 曲线（在忠实 oracle 上）----------
    ks = list(range(1, nG + 1))
    curves = {}
    for name, sc in scores.items():
        order = np.argsort(-sc)
        curves[name] = [leak(set(order[:k].tolist())) for k in ks]

    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    pd.DataFrame({"k": ks, **{n: curves[n] for n in curves}}).to_csv(out / "topk_masked.csv", index=False)
    smallk = {n: float(np.mean(curves[n][:10])) for n in curves}
    (out / "topk_summary.json").write_text(json.dumps(
        {"upper_bound": base, "smallk_mean_first10": smallk}, indent=2, ensure_ascii=False))

    COL = {"mask": "#e15759", "single": "#4c72b0", "shapley": "#59a14f", "ig": "#f28e2b",
           "lrp": "#9467bd", "gate": "#17becf", "random": "#bbbbbb"}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for name in ["shapley", "mask", "single", "ig", "lrp", "gate", "random"]:
        if name in curves:
            ax.plot(ks, curves[name], "-", color=COL[name], lw=2 if name != "random" else 1.3,
                    ls="--" if name == "random" else "-", label=name)
    ax.axhline(base, ls=":", color="#333", lw=1.3, label=f"full-field upper bound = {base:.3f}")
    ax.set_xlabel("top-k general fields revealed"); ax.set_ylabel("faithful leakage v(top-k)  (a_c weighted R2)")
    ax.set_title("DNN58: multi-method top-k on the FAITHFUL (masked-trained) oracle")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3); ax.set_xlim(1, nG)
    fig.tight_layout(); fig.savefig(out / "topk_masked.png", dpi=140); plt.close(fig)

    print("\n=== 各方法 top-k 前10平均(越高=越早挑中真泄露字段) ===")
    for n, v in sorted(smallk.items(), key=lambda x: -x[1]):
        print(f"  {n:9s} {v:.4f}")
    print(f"\n图: {out}/topk_masked.png ; 表: topk_masked.csv")


if __name__ == "__main__":
    main()
