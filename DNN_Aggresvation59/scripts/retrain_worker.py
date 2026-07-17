#!/usr/bin/env python3
"""DNN58 retrain worker：对 top-k 敏感字段，用 DNN 或 GCN+dynamic 结构从头重训，测 12-conf R²。

这是 ground-truth v(top-k)：攻击者只有这 k 个字段时、重新训练能达到的最好推断质量。
用法：retrain_worker.py --k K --struct {dnn,gcn}
输出：outputs/retrain/{struct}_k{K}.json
"""
from __future__ import annotations

import argparse, json, sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 120


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


def train_generic(model, Xtr, Ytr, Xte, Yte, is_gcn):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best = 1e9; bs = None; pat = 0
    n = len(Xtr)
    for ep in range(EPOCHS):
        model.train(); pm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            ix = pm[k:k + 128]; opt.zero_grad()
            loss = ((model(Xtr[ix]) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Xte).cpu().numpy() - Yte) ** 2).mean())
        if v < best: best = v; bs = deepcopy(model.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    model.load_state_dict(bs); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--struct", required=True, choices=["dnn", "gcn"])
    ap.add_argument("--ranking", required=True,
                    choices=["dnn", "noprior", "static", "zeromask", "masked"])
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    m = cfg["model"]; g = cfg["graph"]
    order_fields = json.load(open(ROOT / f"outputs/retrain/ranking_{args.ranking}.json"))["order_fields"]
    name2local = {n: i for i, n in enumerate(di["general"])}
    sel_local = [name2local[order_fields[i]] for i in range(args.k)]   # top-k general 局部索引
    sel_gen_global = gi[sel_local]                        # 全局列索引
    tr, te = di["train_data"], di["test_data"]
    Yte_np = te[:, ci]

    if args.struct == "dnn":
        Xtr = torch.as_tensor(tr[:, sel_gen_global], dtype=torch.float32, device=DEV)
        Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
        Xte = torch.as_tensor(te[:, sel_gen_global], dtype=torch.float32, device=DEV)
        model = DNN(args.k, len(ci)).to(DEV)
        model = train_generic(model, Xtr, Ytr, Xte, Yte_np, False)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()
    else:
        # 子图：节点 = [top-k general] + [12 confidential]，general 在前
        sel_nodes = np.concatenate([sel_gen_global, ci])
        mt_sub = mt[np.ix_(sel_nodes, sel_nodes)]
        nGk = args.k; nN = nGk + len(ci); conf_new = list(range(nGk, nN))
        em = build_edge_mask(mt_sub, top_k=min(int(g["top_k"]), max(1, nGk)),
                             threshold=float(g["threshold"]), symmetrize=True,
                             n_general=nGk, bipartite=True)
        tr_sub = tr[:, sel_nodes]; te_sub = te[:, sel_nodes]
        ftr, ttr = _build_windowed_samples(tr_sub, conf_new, 1)
        fte, tte = _build_windowed_samples(te_sub, conf_new, 1)
        Xtr = torch.as_tensor(ftr, dtype=torch.float32, device=DEV)
        Ytr = torch.as_tensor(ttr, dtype=torch.float32, device=DEV)
        Xte = torch.as_tensor(fte, dtype=torch.float32, device=DEV)
        model = InferenceDrivenGNN(
            metric_tensor=mt_sub, edge_mask=em, n_nodes=nN, n_general=nGk,
            confidential_indices=conf_new, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"],
            dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
            attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
            attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
            alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
            gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
            input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(DEV)
        model = train_generic(model, Xtr, Ytr, Xte, tte, True)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()

    r2 = per_conf_r2(pred, Yte_np)
    res = {"ranking": args.ranking, "k": args.k, "struct": args.struct, "mean_r2": float(r2.mean()),
           "per_conf_r2": {di["confidential"][j]: float(r2[j]) for j in range(len(ci))}}
    (ROOT / "outputs/retrain" / f"{args.ranking}_{args.struct}_k{args.k}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[{args.ranking}/{args.struct} k={args.k}] 12-conf 平均 R² = {r2.mean():.4f}")


if __name__ == "__main__":
    main()
