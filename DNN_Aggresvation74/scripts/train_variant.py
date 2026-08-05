# -*- coding: utf-8 -*-
"""DNN74：训练一个图 oracle 变体。

训练协议逐项对齐 DNN73/scripts/train_variants.py 的 `ours` 方案
（= DNN63/scripts/train_oracle.py）：
  EPOCHS 400 / PATIENCE 60 / BATCH 256 / Adam lr 1e-3 wd 5e-4 /
  训练集内 15% 验证 / 数据切分 seed 42 / 8 组固定验证掩码 /
  两段式失活采样（直接复用 69 号的 sample_mask，不重写）。

**唯一允许改训练预算的是 B4_train / B6_all / D_bypass 变体**（EPOCHS 500 / PATIENCE 150，
依据 42 号"充分训练 +0.035"），且在结果表里单独标注。

用法：
    python scripts/train_variant.py --variant A1_gcn_dynamic --seed 0
    python scripts/train_variant.py --variant B6_all --seed 0 --base A1_gcn_dynamic
"""
import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

R74 = Path(__file__).resolve().parents[1]
R69 = R74.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R74 / "src"))

from src.data_processing import prepare_data                      # noqa: E402  (69 号)
from src.model import build_edge_mask                             # noqa: E402  (69 号)
from src.oracle import MLPOracle, sample_mask                     # noqa: E402  (69 号)
from graph_oracle import (MaskedGraphOracle, VARIANTS,        # noqa: E402  (74 号)
                              build_round_b, n_params)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, N_VAL_MASK, LR, WD = 256, 8, 1e-3, 5e-4
EPOCHS_DEFAULT, PATIENCE_DEFAULT = 400, 60          # = 63/73 号
EPOCHS_LONG, PATIENCE_LONG = 500, 150               # = 42 号"充分训练"，仅 B4/B6/D 用
LONG_BUDGET = {"B4_train", "B6_all", "D_bypass"}


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def resolve_variant(name: str, base: str | None) -> dict:
    """取变体配置：Round A 直接查表；Round B 需要给出 Round A 的胜出者 --base。
    特例 "MLP" 走 69 号原版 MLPOracle（用于在本 harness 下补多种子 MLP 基线）。"""
    if name == "MLP":
        return {}
    if name in VARIANTS:
        return dict(VARIANTS[name])
    if base is None:
        raise SystemExit(f"变体 {name} 属于 Round B，必须同时给 --base <RoundA 胜出者>")
    rb = build_round_b(base)
    if name not in rb:
        raise SystemExit(f"未知变体 {name}；Round B 可选：{sorted(rb)}")
    return rb[name]


def load_data(cfg):
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"])
    ci = np.array(di["confidential_indices"])
    return di, gi, ci


def build_graph_inputs(cfg, nG):
    """相关性张量 + 拓扑。与 63/69/73 完全一致（top_k=8, threshold=0.15, bipartite）。"""
    mt = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
    em = build_edge_mask(mt, top_k=cfg["graph"]["top_k"],
                         threshold=cfg["graph"]["threshold"],
                         symmetrize=True, n_general=nG, bipartite=True)
    return mt, em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--base", default=None, help="Round B 变体所基于的 Round A 胜出者")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="输出名后缀，用于同名变体建在不同 base 上时区分")
    args = ap.parse_args()

    vcfg = resolve_variant(args.variant, args.base)
    epochs = EPOCHS_LONG if args.variant in LONG_BUDGET else EPOCHS_DEFAULT
    patience = PATIENCE_LONG if args.variant in LONG_BUDGET else PATIENCE_DEFAULT

    cfg = yaml.safe_load((R74 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di, gi, ci = load_data(cfg)
    nG, nC = di["n_general"], di["n_confidential"]

    Xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
    Yte = di["test_data"][:, ci]

    # 训练集内再留 15% 做 early stop（与 63/73 逐项一致）
    n = len(Xtr)
    n_val = max(int(n * 0.15), 1)
    perm = np.random.RandomState(args.seed).permutation(n)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = torch.as_tensor(perm[n_val:], device=DEV)

    mt, em = build_graph_inputs(cfg, nG)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = (MLPOracle(nG, nC) if args.variant == "MLP"
             else MaskedGraphOracle(mt, em, nG, nC, **vcfg)).to(DEV)
    npar = n_params(model)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    Xva, Yva = Xtr[va_idx], Ytr[va_idx]
    val_masks = [torch.as_tensor(sample_mask(len(va_idx), nG, val_rng), device=DEV)
                 for _ in range(N_VAL_MASK)]

    best, best_state, pat = 1e9, None, 0
    ntr = len(tr_idx)
    t0 = time.time()
    for ep in range(epochs):
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
            if pat >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        m_full = torch.ones(len(Xte), nG, device=DEV)
        r2_full = float(per_conf_r2(model(Xte, m_full).cpu().numpy(), Yte).mean())

    out = R74 / "outputs" / f"oracle_{args.variant}{args.tag}_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "variant": args.variant, "base": args.base,
                "vcfg": vcfg, "seed": args.seed, "nG": nG, "nC": nC,
                "val_loss": best, "r2_full": r2_full, "epochs_run": ep + 1,
                "epochs_budget": epochs, "n_params": npar,
                "train_seconds": time.time() - t0}, out)
    print(f"[{args.variant}{args.tag} seed{args.seed}] ep={ep+1}/{epochs} params={npar/1000:.0f}k "
          f"val={best:.4f} 满输入12-conf R²={r2_full:.4f} "
          f"[{time.time()-t0:.0f}s] -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
