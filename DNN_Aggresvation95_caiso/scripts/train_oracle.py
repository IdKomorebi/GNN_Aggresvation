# -*- coding: utf-8 -*-
"""95_caiso：训练 uniform（两段式）掩码分布的 MLP-oracle。

协议逐项对齐 75 号 train_variant.py（EPOCHS 400 / PATIENCE 60 / BATCH 256 /
Adam lr 1e-3 wd 5e-4 / 训练集内 15% 验证 / 数据切分 seed 42 / 8 组固定验证掩码），
掩码分布固定为 69 号 sample_mask（uniform 两段式，主线方案）。

用法：python scripts/train_oracle.py --seed 0
"""
import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data   # noqa: E402  (69 号)
from src.oracle import MLPOracle, sample_mask  # noqa: E402  (69 号)
from runlog import log                         # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BATCH, N_VAL_MASK = 400, 60, 256, 8
LR, WD = 1e-3, 5e-4


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/caiso_2025_hourly_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"])
    ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]

    Xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
    Yte = di["test_data"][:, ci]

    n = len(Xtr)
    n_val = max(int(n * 0.15), 1)
    perm = np.random.RandomState(args.seed).permutation(n)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = torch.as_tensor(perm[n_val:], device=DEV)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOracle(nG, nC).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    Xva, Yva = Xtr[va_idx], Ytr[va_idx]
    val_masks = [torch.as_tensor(sample_mask(len(va_idx), nG, val_rng), device=DEV)
                 for _ in range(N_VAL_MASK)]

    best, best_state, pat = 1e9, None, 0
    ntr = len(tr_idx)
    t0 = time.time()
    log("ORACLE", "START", note=f"uniform seed{args.seed} nG={nG} nC={nC} n_train={n}")
    for ep in range(EPOCHS):
        model.train()
        order = tr_idx[torch.randperm(ntr, device=DEV)]
        for b in range(0, ntr, BATCH):
            ix = order[b:b + BATCH]
            m = torch.as_tensor(sample_mask(len(ix), nG, mask_rng), device=DEV)
            opt.zero_grad()
            loss = ((model(Xtr[ix], m) - Ytr[ix]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([((model(Xva, mm) - Yva) ** 2).mean().item()
                                   for mm in val_masks]))
        if vloss < best - 1e-6:
            best, best_state, pat = vloss, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        m_full = torch.ones(len(Xte), nG, device=DEV)
        r2_full = float(per_conf_r2(model(Xte, m_full).cpu().numpy(), Yte).mean())

    out = ROOT / "outputs" / f"oracle_uniform_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "scheme": "uniform", "seed": args.seed,
                "nG": nG, "nC": nC, "val_loss": best, "r2_full": r2_full,
                "epochs_run": ep + 1,
                "n_params": sum(p.numel() for p in model.parameters())}, out)
    msg = (f"[uniform seed{args.seed}] ep={ep+1}/{EPOCHS} val={best:.4f} "
           f"满输入12-conf R²={r2_full:.4f} [{time.time()-t0:.0f}s] -> {out.name}")
    print(msg, flush=True)
    log("ORACLE", "DONE", note=msg)


if __name__ == "__main__":
    main()
