#!/usr/bin/env python3
"""DNN60 retrain worker：对任意 general 字段子集，用 DNN 或 GCN+dynamic 结构从头重训，测 12-conf R²。

与 DNN59 的差别：
1. 字段集合不再来自排名 top-k，而是 outputs/subsets.json 里的任意子集（--subset-id）；
2. 增加 --seed 控制训练随机性（模型初始化 / dropout / batch 顺序），
   数据切分种子固定为 base.yaml 的 runtime.seed=42，保证所有任务同一份 train/test。
输出：outputs/retrain/{subset_id}_{struct}_seed{seed}.json
"""
from __future__ import annotations

import argparse, json, sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

# DNN77：逐字复制自 67 号，只改路径头——src/、相关性张量、CSV 取自 69 号（保持真值口径
# 与 67/68 号既有三阶真值完全一致），base.yaml / subsets.json / outputs 落在 77 号目录。
ROOT = Path(__file__).resolve().parent.parent          # = DNN_Aggresvation77
R69 = ROOT.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
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


def train_generic(model, Xtr, Ytr, Xte, Yte):
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
    ap.add_argument("--subset-id", required=True)
    ap.add_argument("--struct", required=True, choices=["dnn", "gcn"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    # 数据切分种子固定（runtime.seed=42），与 59 号的真值口径一致
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    m = cfg["model"]; g = cfg["graph"]

    subset = json.load(open(ROOT / "outputs/subsets.json"))[args.subset_id]
    fields = subset["fields"]
    name2local = {n: i for i, n in enumerate(di["general"])}
    sel_local = [name2local[f] for f in fields]
    sel_gen_global = gi[sel_local]
    nGk = len(fields)
    tr, te = di["train_data"], di["test_data"]
    Yte_np = te[:, ci]

    # 训练随机性由 --seed 控制（初始化 / dropout / batch 顺序）
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.struct == "dnn":
        Xtr = torch.as_tensor(tr[:, sel_gen_global], dtype=torch.float32, device=DEV)
        Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
        Xte = torch.as_tensor(te[:, sel_gen_global], dtype=torch.float32, device=DEV)
        model = DNN(nGk, len(ci)).to(DEV)
        model = train_generic(model, Xtr, Ytr, Xte, Yte_np)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()
    else:
        # 子图：节点 = [子集 general] + [12 confidential]，general 在前
        sel_nodes = np.concatenate([sel_gen_global, ci])
        mt_sub = mt[np.ix_(sel_nodes, sel_nodes)]
        nN = nGk + len(ci); conf_new = list(range(nGk, nN))
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
        model = train_generic(model, Xtr, Ytr, Xte, tte)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()

    r2 = per_conf_r2(pred, Yte_np)
    res = {"subset_id": args.subset_id, "group": subset["group"], "size": nGk,
           "fields": fields, "struct": args.struct, "seed": args.seed,
           "mean_r2": float(r2.mean()),
           "per_conf_r2": {di["confidential"][j]: float(r2[j]) for j in range(len(ci))}}
    (ROOT / "outputs/retrain" / f"{args.subset_id}_{args.struct}_seed{args.seed}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[{args.subset_id}/{args.struct}/seed{args.seed} size={nGk}] 12-conf 平均 R² = {r2.mean():.4f}")


if __name__ == "__main__":
    main()
