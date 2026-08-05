# -*- coding: utf-8 -*-
"""DNN73：随机子集失活【训练方案】消融——训练三个变体。

核心问题（用户提出，方法本身的关键论证）：随机失活训练出的通用 oracle，
是否比【没有随机失活】的常规模型更接近逐集合重训真值？

三个变体，除【训练掩码分布】外一切相同（架构/数据/切分/轮数/耐心/优化器/种子）：
  none   : 始终满输入训练（常规模型）。查询任意 S 时把不可见字段置零 —— 即 IGNN 式置零口径。
  bern50 : 每字段独立 Bernoulli(0.5) 失活。等价于 2^nG 均匀采样，尺寸集中在 nG/2 附近。
           用于检验我们两段式采样的设计选择（此前只有先验论证、从未实测）。
  ours   : 两段式尺寸均匀采样（先 k~U{1..nG} 再均匀取 k 个可见）= 69 号 oracle 的训练法。

公平性设计：**每个变体用自己的掩码分布做验证/early-stop**（none 用满输入验证损失），
即把每个基线都训到它自己的最优，不以"用别人的目标早停"来削弱基线。

训练协议逐项对齐 DNN63/scripts/train_oracle.py（EPOCHS400/PATIENCE60/BATCH256/
Adam lr1e-3 wd5e-4/训练集内 15% 验证/数据切分 seed42）。
"""
import argparse, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, torch, yaml

R69 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69")
OUT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation73/outputs")
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import MLPOracle, GNNOracle, build_priors, sample_mask

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BATCH, N_VAL_MASK = 400, 60, 256, 8


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def make_sampler(scheme):
    """返回 (batch, nG, rng) -> mask 的采样器。"""
    if scheme == "ours":
        return sample_mask
    if scheme == "none":
        def f(batch, nG, rng):
            return np.ones((batch, nG), dtype=np.float32)
        return f
    if scheme == "bern50":
        def f(batch, nG, rng):
            m = (rng.random_sample((batch, nG)) < 0.5).astype(np.float32)
            # 避免全零掩码（退化：无任何可见字段）
            empty = m.sum(1) == 0
            if empty.any():
                idx = rng.randint(0, nG, size=int(empty.sum()))
                m[np.where(empty)[0], idx] = 1.0
            return m
        return f
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mlp", "gnn"])
    ap.add_argument("--scheme", required=True, choices=["none", "bern50", "ours"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gnn-hidden", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load((R69 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi, ci = np.array(di["general_indices"]), np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    tr, te = di["train_data"], di["test_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV)
    Yte = te[:, ci]

    n = len(Xtr); n_val = max(int(n * 0.15), 1)
    perm = np.random.RandomState(args.seed).permutation(n)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = torch.as_tensor(perm[n_val:], device=DEV)

    if args.arch == "mlp":
        model = MLPOracle(nG, nC).to(DEV)
    else:
        mt = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
        em = build_edge_mask(mt, top_k=cfg["graph"]["top_k"], threshold=cfg["graph"]["threshold"],
                             symmetrize=True, n_general=nG, bipartite=True)
        A_gg, prior_cg = build_priors(mt, em, nG)
        model = GNNOracle(A_gg, prior_cg, nC, hidden=args.gnn_hidden,
                          n_layers=args.gnn_layers).to(DEV)

    sampler = make_sampler(args.scheme)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    mask_rng = np.random.RandomState(1234 + args.seed)
    val_rng = np.random.RandomState(999)
    Xva, Yva = Xtr[va_idx], Ytr[va_idx]
    # 验证掩码用【本变体自己的】分布：把每个基线都训到它自己的最优
    val_masks = [torch.as_tensor(sampler(len(va_idx), nG, val_rng), device=DEV)
                 for _ in range(1 if args.scheme == "none" else N_VAL_MASK)]

    best, best_state, pat = 1e9, None, 0
    ntr = len(tr_idx)
    for ep in range(EPOCHS):
        model.train()
        order = tr_idx[torch.randperm(ntr, device=DEV)]
        for b in range(0, ntr, BATCH):
            ix = order[b:b + BATCH]
            m = torch.as_tensor(sampler(len(ix), nG, mask_rng), device=DEV)
            opt.zero_grad()
            loss = ((model(Xtr[ix], m) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([((model(Xva, mm) - Yva) ** 2).mean().item() for mm in val_masks]))
        if vloss < best - 1e-6:
            best, best_state, pat = vloss, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()

    with torch.no_grad():
        m_full = torch.ones(len(Xte), nG, device=DEV)
        r2_full = float(per_conf_r2(model(Xte, m_full).cpu().numpy(), Yte).mean())
    out = OUT / f"oracle_{args.arch}_{args.scheme}_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "arch": args.arch, "scheme": args.scheme,
                "seed": args.seed, "nG": nG, "nC": nC, "gnn_hidden": args.gnn_hidden,
                "gnn_layers": args.gnn_layers, "val_loss": best, "r2_full": r2_full,
                "epochs_run": ep + 1}, out)
    print(f"[{args.arch}/{args.scheme} seed{args.seed}] ep={ep+1} val={best:.4f} "
          f"满输入12-conf R²={r2_full:.4f} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
