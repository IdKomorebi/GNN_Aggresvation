"""DNN68 训练难度分带专家 oracle。除掩码采样范围（--size-lo/--size-hi）与隐藏维
（--hidden）外，与 63 号 train_oracle.py 协议逐项一致（EPOCHS/PATIENCE/BATCH、内部验证、
种子口径）。等容量版 hidden=256（同 63）；容量再分配版由调度器按带指定不同 hidden。
用法：train_banded.py --band E2 --size-lo 6 --size-hi 9 --hidden 256 --seed 0
输出：outputs/expert_{band}_h{hidden}_seed{seed}.pt
"""
from __future__ import annotations

import argparse, sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle, sample_mask_banded

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BATCH = 400, 60, 256
N_VAL_MASK = 8


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True)
    ap.add_argument("--size-lo", type=int, required=True)
    ap.add_argument("--size-hi", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    tr, te = di["train_data"], di["test_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV); Yte = te[:, ci]

    n = len(Xtr); n_val = max(int(n * 0.15), 1)
    perm = np.random.RandomState(args.seed).permutation(n)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = torch.as_tensor(perm[n_val:], device=DEV)

    model = MLPOracle(nG, nC, hidden=args.hidden).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    Xva, Yva = Xtr[va_idx], Ytr[va_idx]
    # 验证掩码也限制在本专家的尺寸带（early stop 对齐部署分布）
    val_masks = [torch.as_tensor(
        sample_mask_banded(len(va_idx), nG, val_rng, args.size_lo, args.size_hi), device=DEV)
        for _ in range(N_VAL_MASK)]

    best = 1e9; best_state = None; pat = 0
    ntr = len(tr_idx)
    for ep in range(EPOCHS):
        model.train()
        order = tr_idx[torch.randperm(ntr, device=DEV)]
        for b in range(0, ntr, BATCH):
            ix = order[b:b + BATCH]
            m = torch.as_tensor(
                sample_mask_banded(len(ix), nG, mask_rng, args.size_lo, args.size_hi), device=DEV)
            opt.zero_grad()
            loss = ((model(Xtr[ix], m) - Ytr[ix]) ** 2).mean()
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

    n_params = sum(p.numel() for p in model.parameters())
    out = ROOT / "outputs" / f"expert_{args.band}_h{args.hidden}_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "band": args.band,
                "size_lo": args.size_lo, "size_hi": args.size_hi,
                "hidden": args.hidden, "seed": args.seed, "nG": nG, "nC": nC,
                "n_params": n_params, "val_loss": float(best)}, out)
    print(f"[{args.band} size{args.size_lo}-{args.size_hi} h{args.hidden}] "
          f"val_loss={best:.4f} params={n_params} -> {out.name}")


if __name__ == "__main__":
    main()
