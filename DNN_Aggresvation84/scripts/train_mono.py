# -*- coding: utf-8 -*-
"""84 号 杠杆A：单调性 / 增量一致性目标 oracle。

动机（80 号钉死）：最强三阶协同在 K=0 时 v̂(ijk) 被压到比它的二元子集还低，
syn3=v̂(ijk)−max_pair 变负 → S1 结构性漏最强档。而 v(S) 单调（加字段不该降低可推断性）
是重训真值意义下的恒真性质：攻击者可忽略多余字段，故 v(S∪{k}) ≥ v(S)。

本脚本在标准 uniform 掩码 MSE 训练上，叠加一个**纯自洽、不碰重训真值表**的单调性软惩罚：
对当前 batch 施加同一掩码算逐-conf MSE，则 v(S)单调 ⟺ MSE(S∪k) ≤ MSE(S)，
惩罚 = mean_conf relu( MSE_conf(超集) − MSE_conf(子集) )。
聚焦低阶（子集 size∈{1,2,3}、超集加一元），因为病灶集中在 1→2→3→4 相变处。

红线核查：惩罚只用训练集 (x,y_train) 上 oracle 自身预测的 MSE（和主损失同源），
never touches 68 号重训真值表 v_ijk_true。字段选取一律随机，不用任何已知协同位置。

变体：
  --penalty none        复现 75 uniform（sanity，应≈基线）
  --penalty mono        随机嵌套对 (S, S∪{k})，S size∈{1,2,3}
  --penalty incr        显式三阶：v(ijk) ≥ 每个二元子集
  --penalty mono_incr   两者叠加
  --lam                 惩罚权重
"""
from __future__ import annotations

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
R75 = REPO / "DNN_Aggresvation75"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from src.oracle import MLPOracle  # noqa: E402
from runlog import log  # noqa: E402

BATCH = 256
LR = 1e-3
WD = 5e-4
N_VAL_MASK = 8
N_MONO = 8   # 每 batch 采样的嵌套约束组数
N_INCR = 4   # 每 batch 采样的三阶约束组数


def per_conf_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None)


def load_data() -> dict:
    cfg = yaml.safe_load((R75 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    torch.manual_seed(42)
    np.random.seed(42)
    return prepare_data(cfg)


def uniform_masks(batch: int, n_general: int, rng: np.random.RandomState) -> np.ndarray:
    m = np.zeros((batch, n_general), dtype=np.float32)
    ks = rng.randint(1, n_general + 1, size=batch)
    for i, k in enumerate(ks):
        idx = rng.choice(n_general, size=int(k), replace=False)
        m[i, idx] = 1.0
    return m


def mono_penalty(model, x, y, n_general, rng, n_pairs, device):
    """随机嵌套对 (S, S∪{k})，S size∈{1,2,3}。惩罚 mean_conf relu(MSE_sub − MSE_sup)。"""
    pen = x.new_zeros(())
    for _ in range(n_pairs):
        base_size = int(rng.randint(1, 4))          # 1,2,3
        base = rng.choice(n_general, size=base_size, replace=False)
        remaining = np.setdiff1d(np.arange(n_general), base)
        add = int(rng.choice(remaining))
        sub = np.zeros(n_general, dtype=np.float32); sub[base] = 1.0
        sup = sub.copy(); sup[add] = 1.0
        m_sub = torch.as_tensor(sub, device=device).unsqueeze(0).expand(len(x), -1)
        m_sup = torch.as_tensor(sup, device=device).unsqueeze(0).expand(len(x), -1)
        mse_sub = ((model(x, m_sub) - y) ** 2).mean(dim=0)   # (nC,)
        mse_sup = ((model(x, m_sup) - y) ** 2).mean(dim=0)
        pen = pen + torch.relu(mse_sup - mse_sub).mean()
    return pen / max(n_pairs, 1)


def incr_penalty(model, x, y, n_general, rng, n_tri, device):
    """随机三元组 {i,j,k}，强制 v(ijk) ≥ 每个二元子集 ⟺ MSE(ijk) ≤ MSE(pair)。"""
    pen = x.new_zeros(())
    for _ in range(n_tri):
        tri = rng.choice(n_general, size=3, replace=False)
        m_tri = np.zeros(n_general, dtype=np.float32); m_tri[tri] = 1.0
        mt = torch.as_tensor(m_tri, device=device).unsqueeze(0).expand(len(x), -1)
        mse_tri = ((model(x, mt) - y) ** 2).mean(dim=0)      # (nC,)
        for pair in ((0, 1), (0, 2), (1, 2)):
            m_pair = np.zeros(n_general, dtype=np.float32); m_pair[tri[list(pair)]] = 1.0
            mp = torch.as_tensor(m_pair, device=device).unsqueeze(0).expand(len(x), -1)
            mse_pair = ((model(x, mp) - y) ** 2).mean(dim=0)
            pen = pen + torch.relu(mse_tri - mse_pair).mean()
    return pen / max(n_tri * 3, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty", choices=("none", "mono", "incr", "mono_incr"), required=True)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()
    started = time.time()
    scheme = f"{args.penalty}_lam{args.lam:g}" + (f"_{args.tag}" if args.tag else "")
    epochs, patience = args.epochs, args.patience
    log("TRAIN", "START", note=f"{scheme}/seed{args.seed}", scheme=scheme, seed=args.seed)

    data = load_data()
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.as_tensor(data["train_data"][:, general_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(data["train_data"][:, conf_idx], dtype=torch.float32, device=device)
    x_test = torch.as_tensor(data["test_data"][:, general_idx], dtype=torch.float32, device=device)
    y_test = data["test_data"][:, conf_idx]

    n_samples = len(x_train)
    n_val = max(int(n_samples * 0.15), 1)
    permutation = np.random.RandomState(args.seed).permutation(n_samples)
    val_idx = torch.as_tensor(permutation[:n_val], device=device)
    train_idx = torch.as_tensor(permutation[n_val:], device=device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOracle(n_general, n_conf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    mask_rng = np.random.RandomState(1234 + args.seed)
    pen_rng = np.random.RandomState(4321 + args.seed)
    val_rng = np.random.RandomState(999)
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    val_masks = [
        torch.as_tensor(uniform_masks(len(val_idx), n_general, val_rng), dtype=torch.float32, device=device)
        for _ in range(N_VAL_MASK)
    ]

    do_mono = args.penalty in ("mono", "mono_incr")
    do_incr = args.penalty in ("incr", "mono_incr")
    best, best_state, stale = float("inf"), None, 0
    n_train = len(train_idx)
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(n_train, device=device)]
        for begin in range(0, n_train, BATCH):
            batch_idx = order[begin : begin + BATCH]
            xb, yb = x_train[batch_idx], y_train[batch_idx]
            masks = torch.as_tensor(uniform_masks(len(batch_idx), n_general, mask_rng),
                                    dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = ((model(xb, masks) - yb) ** 2).mean()
            if do_mono:
                loss = loss + args.lam * mono_penalty(model, xb, yb, n_general, pen_rng, N_MONO, device)
            if do_incr:
                loss = loss + args.lam * incr_penalty(model, xb, yb, n_general, pen_rng, N_INCR, device)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([((model(x_val, mask) - y_val) ** 2).mean().item()
                                      for mask in val_masks]))
        if val_loss < best - 1e-6:
            best, best_state, stale = val_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full_mask = torch.ones(len(x_test), n_general, device=device)
        full_r2 = float(per_conf_r2(model(x_test, full_mask).cpu().numpy(), y_test).mean())
    output = ROOT / "outputs" / f"oracle_{scheme}_seed{args.seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "scheme": scheme, "penalty": args.penalty,
                "lam": args.lam, "seed": args.seed, "val_loss": best, "r2_full": full_r2,
                "epochs_run": epoch + 1,
                "n_params": sum(p.numel() for p in model.parameters())}, output)
    elapsed = time.time() - started
    log("TRAIN", "DONE", note=f"{scheme}/seed{args.seed}", elapsed_s=elapsed,
        scheme=scheme, seed=args.seed, epochs=epoch + 1, val_loss=round(best, 6),
        r2_full=round(full_r2, 4))
    print(f"[{scheme}/seed{args.seed}] epoch={epoch+1} val={best:.5f} fullR2={full_r2:.4f} [{elapsed:.1f}s]",
          flush=True)


if __name__ == "__main__":
    main()
