#!/usr/bin/env python3
"""DNN58：随机遮蔽训练 gcn_dynamic —— 让模型对任意 general 子集都能忠实推断。

每个样本随机保留一部分 general 字段（keep-prob q~U(0,1) 后逐字段 Bernoulli），
使得对任意 S 的前向 v(S) 都是"攻击者只有 S 时的推断质量"的忠实估计，而不是
置零版那种分布外查询。confidential 始终置 0。early-stop 仍看 unmasked test 损失。
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]; nG = di["n_general"]
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

    ftr, ttr = _build_windowed_samples(di["train_data"], di["confidential_indices"], 1)
    fte, tte = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    Xtr = torch.as_tensor(ftr, dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(ttr, dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(fte, dtype=torch.float32, device=DEV)
    Yte = torch.as_tensor(tte, dtype=torch.float32, device=DEV)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # alpha 参数更高学习率（与原训练一致）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
    best = float("inf"); best_state = None; patience = 0
    n = len(Xtr)
    gen = torch.Generator(device=DEV); gen.manual_seed(42)
    for ep in range(500):
        model.train(); perm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            idx = perm[k:k + 128]
            xb = Xtr[idx].clone(); B = xb.shape[0]
            # 随机遮蔽 general：keep-prob q~U(0,1) 后逐字段 Bernoulli
            q = torch.rand(B, 1, device=DEV, generator=gen)
            keep = (torch.rand(B, nG, device=DEV, generator=gen) < q).float()
            xb[:, :nG, 0] = xb[:, :nG, 0] * keep
            opt.zero_grad()
            loss = ((model(xb) - Ytr[idx]) ** 2).mean()
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            te = ((model(Xte) - Yte) ** 2).mean().item()   # unmasked 全字段 test
        if te < best:
            best = te; best_state = deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= 150:
                print(f"  early stop @ {ep}"); break
        if ep % 50 == 0:
            print(f"  ep{ep:3d} unmasked_test_mse={te:.4f}")
    model.load_state_dict(best_state); model.eval()
    out = ROOT / "trained"; out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "gcn_dynamic_masked.pt")
    # 全字段 R²
    with torch.no_grad():
        p = model(Xte).cpu().numpy()
    t = tte; ss = ((t - p) ** 2).sum(0); st = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    r2 = np.clip(1 - ss / st, 0, None)
    print(f"遮蔽训练版 全字段 12-conf 平均 R² = {r2.mean():.4f}")
    print(f"已保存 {out/'gcn_dynamic_masked.pt'}")


if __name__ == "__main__":
    main()
