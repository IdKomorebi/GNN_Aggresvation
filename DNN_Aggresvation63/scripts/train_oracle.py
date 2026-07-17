"""DNN63 训练子集条件 oracle（MLP 或 GNN），随机子集失活。

用法：train_oracle.py --arch {mlp,gnn} --seed S [--data-frac F]
输出：outputs/oracle_{arch}_seed{S}[_f{F}].pt
数据切分固定 shuffle seed 42（与 59/60 真值口径一致）；--data-frac 只截训练集前 F 比例，
测试集不变（低数据实验）。
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import MLPOracle, GNNOracle, build_priors, sample_mask

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BATCH = 400, 60, 256
N_VAL_MASK = 8       # 验证时每个验证样本用多少个随机掩码平均（稳定 early stop 信号）


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mlp", "gnn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--gnn-hidden", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]

    tr, te = di["train_data"], di["test_data"]
    if args.data_frac < 1.0:
        n_keep = int(len(tr) * args.data_frac)
        tr = tr[:n_keep]
        print(f"  低数据实验：训练集截取前 {args.data_frac:.0%} = {n_keep} 行")

    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV)
    Yte = te[:, ci]

    # 训练/验证切分（在训练集内部再留 15% 做 early stop）
    n = len(Xtr); n_val = max(int(n * 0.15), 1)
    perm = np.random.RandomState(args.seed).permutation(n)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = torch.as_tensor(perm[n_val:], device=DEV)

    if args.arch == "mlp":
        model = MLPOracle(nG, nC).to(DEV)
    else:
        mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
        em = build_edge_mask(mt, top_k=cfg["graph"]["top_k"], threshold=cfg["graph"]["threshold"],
                             symmetrize=True, n_general=nG, bipartite=True)
        A_gg, prior_cg = build_priors(mt, em, nG)
        model = GNNOracle(A_gg, prior_cg, nC,
                          hidden=args.gnn_hidden, n_layers=args.gnn_layers).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    # 固定验证掩码集（early stop 信号稳定）
    Xva, Yva = Xtr[va_idx], Ytr[va_idx]
    val_masks = [torch.as_tensor(sample_mask(len(va_idx), nG, val_rng), device=DEV)
                 for _ in range(N_VAL_MASK)]

    best = 1e9; best_state = None; pat = 0
    ntr = len(tr_idx)
    for ep in range(EPOCHS):
        model.train()
        order = tr_idx[torch.randperm(ntr, device=DEV)]
        for b in range(0, ntr, BATCH):
            ix = order[b:b + BATCH]
            m = torch.as_tensor(sample_mask(len(ix), nG, mask_rng), device=DEV)
            opt.zero_grad()
            pred = model(Xtr[ix], m)
            loss = ((pred - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = np.mean([((model(Xva, mm) - Yva) ** 2).mean().item() for mm in val_masks])
        if vloss < best - 1e-6:
            best = vloss; best_state = deepcopy(model.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()

    # 快速自检：满输入 R²（应接近满输入攻击者），单字段平均 R²
    with torch.no_grad():
        m_full = torch.ones(len(Xte), nG, device=DEV)
        r2_full = per_conf_r2(model(Xte, m_full).cpu().numpy(), Yte).mean()
    tag = f"_f{args.data_frac}" if args.data_frac < 1.0 else ""
    out = ROOT / "outputs" / f"oracle_{args.arch}_seed{args.seed}{tag}.pt"
    torch.save({"state": model.state_dict(), "arch": args.arch, "seed": args.seed,
                "data_frac": args.data_frac, "nG": nG, "nC": nC,
                "gnn_hidden": args.gnn_hidden, "gnn_layers": args.gnn_layers,
                "val_loss": float(best), "r2_full": float(r2_full)}, out)
    print(f"[{args.arch} seed{args.seed} f{args.data_frac}] val_loss={best:.4f} "
          f"满输入12-conf R²={r2_full:.4f} -> {out.name}")


if __name__ == "__main__":
    main()
