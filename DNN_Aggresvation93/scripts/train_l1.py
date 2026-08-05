# -*- coding: utf-8 -*-
"""93 号 任务B：★L1——把闭式读出放进训练回路（meta-learned features）。

动机
----
91 号 E4 证明：**冻结**主干 + 闭式重解读出，就已经超过 50 步微调（ρ 0.904 vs 0.899）。
但那个主干是为**共享读出**训练的——它并不知道自己的读出层会被逐集合重解，
于是仍然把容量花在"记住每个掩码该输出什么"，而不是"造一组好重解的基"。

L1 直接把这件事告诉主干（R2D2 / MetaOptNet 的 differentiable closed-form base learner）：

    每步：采样掩码 m_S → 前向 support 得 Φ_sup → **闭式** β=(ΦᵀΦ+λI)⁻¹ΦᵀY_sup
          → 在 query 上算损失 → **反传穿过 solve**

λ 作为可学参数（log 参数化）。主干结构与参数量和 75 号 MLPOracle 完全一致，保证可比。

--aug N：★修 91 号 E1 暴露的"φ 非通用基"
    除 12 个真实 conf 外，每步再抽 N 个**合成目标**（从可见集合 S 内随机取 2–4 个字段
    的乘积，标准化），一并进入闭式解与损失。这逼迫 φ 张成的空间不只服务这 12 个 conf。
    代价是稀释真实 conf 的容量——是否值得由实验判定，两种结果都记录。

口径：掩码采样器、EPOCHS/PATIENCE/优化器与 75 号 `train_variant.py` 逐项对齐；
train 内部 15% 作 validation（与 75 号同一 seed 划分），测试集完全不参与训练。
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
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from src.oracle import MLPOracle, sample_mask  # noqa: E402
from runlog import log  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 60           # 与 75 号一致
LR, WD = 1e-3, 5e-4                  # 与 75 号一致
N_STEP = 8                           # 每 epoch 的 meta-step 数
B_MASK, N_SUP, N_QRY = 4, 512, 256   # 每步 4 个掩码，各 512 support + 256 query
D_PHI = 256                          # φ = 主干最后一层 ReLU 激活（对齐 91 号 last）
RELU_LAST = 7                        # MLPOracle.net 中最后一个 ReLU 的下标


def phi_of(model: MLPOracle, x, m):
    """主干前向到最后一个 ReLU，得 (n, 256) 条件特征。不经过 net[9] 共享读出。"""
    h = torch.cat([x * m, m], dim=1)
    for i, layer in enumerate(model.net):
        h = layer(h)
        if i == RELU_LAST:
            return h
    raise RuntimeError


def closed_form_loss(phi_sup, y_sup, phi_qry, y_qry, lam):
    """闭式岭读出 + query 损失，全程可微（梯度穿过 solve）。

    phi_*: (B,n,d)  y_*: (B,n,c)
    """
    mu = phi_sup.mean(1, keepdim=True)
    sd = phi_sup.std(1, keepdim=True).clamp_min(1e-6)
    P, Q = (phi_sup - mu) / sd, (phi_qry - mu) / sd
    ym = y_sup.mean(1, keepdim=True)
    d = P.shape[2]
    eye = torch.eye(d, device=P.device, dtype=P.dtype).expand(P.shape[0], d, d)
    G = P.transpose(1, 2) @ P + lam * eye
    H = P.transpose(1, 2) @ (y_sup - ym)
    beta = torch.linalg.solve(G, H)
    return ((y_qry - ym - Q @ beta) ** 2).mean()


def synth_targets(x, m, sets, rng, n_aug, device):
    """从每个掩码的可见字段里抽 2–4 个做乘积，作为合成目标（标准化）。

    x:(B,n,nG) 已按掩码取值；sets: list of 可见字段索引数组。
    """
    cols = []
    for b, sel in enumerate(sets):
        for _ in range(n_aug):
            r = min(int(rng.randint(2, 5)), len(sel))
            idx = rng.choice(sel, size=r, replace=False)
            p = x[b][:, idx].prod(dim=1)
            cols.append((p - p.mean()) / p.std().clamp_min(1e-6))
    t = torch.stack(cols).reshape(len(sets), n_aug, -1).transpose(1, 2)
    _ = m, device
    return t                                     # (B,n,n_aug)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", type=int, default=0, help=">0 则每步加 N 个合成目标")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="400 时 val 仍在下降（未收敛），可加长")
    ap.add_argument("--steps", type=int, default=N_STEP)
    args = ap.parse_args()
    n_epoch, n_step = args.epochs, args.steps

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    nG, nC = len(gi), len(ci)
    Xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)

    # train 内部 15% val，与 75 号同一划分方式
    n = len(Xtr)
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = max(int(n * 0.15), 1)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = perm[n_val:]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOracle(nG, nC).to(DEV)
    log_lam = torch.zeros(1, device=DEV, requires_grad=True)      # λ 可学，初始 1.0
    opt = torch.optim.Adam(list(model.parameters()) + [log_lam], lr=LR, weight_decay=WD)
    rng = np.random.RandomState(1234 + args.seed)

    # 固定的验证任务：8 组掩码 × (support/query)，整个训练中不变
    vr = np.random.RandomState(999)
    vm = sample_mask(8, nG, vr)
    vperm = vr.permutation(len(va_idx))
    v_sup = va_idx[torch.as_tensor(vperm[:len(vperm) // 2], device=DEV)]
    v_qry = va_idx[torch.as_tensor(vperm[len(vperm) // 2:], device=DEV)]
    vm_t = torch.as_tensor(vm, device=DEV)

    def val_loss():
        model.eval()
        with torch.no_grad():
            ps, pq, ys, yq = [], [], [], []
            for b in range(len(vm)):
                mb_s = vm_t[b:b + 1].expand(len(v_sup), -1)
                mb_q = vm_t[b:b + 1].expand(len(v_qry), -1)
                ps.append(phi_of(model, Xtr[v_sup], mb_s))
                pq.append(phi_of(model, Xtr[v_qry], mb_q))
                ys.append(Ytr[v_sup])
                yq.append(Ytr[v_qry])
            return float(closed_form_loss(torch.stack(ps), torch.stack(ys),
                                          torch.stack(pq), torch.stack(yq),
                                          log_lam.exp()))

    best, best_state, pat = 1e9, None, 0
    t0 = time.time()
    log("L1", "START", note=f"seed{args.seed} aug={args.aug} ep={args.epochs} "
                            f"B_MASK={B_MASK} sup/qry={N_SUP}/{N_QRY} step/ep={N_STEP}")
    for ep in range(n_epoch):
        model.train()
        for _ in range(n_step):
            masks = sample_mask(B_MASK, nG, rng)
            sets = [np.where(masks[b] > 0)[0] for b in range(B_MASK)]
            mt = torch.as_tensor(masks, device=DEV)
            ps, pq, ys, yq = [], [], [], []
            xs_raw, xq_raw = [], []
            for b in range(B_MASK):
                ix = rng.choice(tr_idx, size=N_SUP + N_QRY, replace=False)
                si = torch.as_tensor(ix[:N_SUP], device=DEV)
                qi = torch.as_tensor(ix[N_SUP:], device=DEV)
                mb = mt[b:b + 1]
                ps.append(phi_of(model, Xtr[si], mb.expand(N_SUP, -1)))
                pq.append(phi_of(model, Xtr[qi], mb.expand(N_QRY, -1)))
                ys.append(Ytr[si])
                yq.append(Ytr[qi])
                xs_raw.append(Xtr[si])
                xq_raw.append(Xtr[qi])
            Ys, Yq = torch.stack(ys), torch.stack(yq)
            if args.aug:
                Ys = torch.cat([Ys, synth_targets(torch.stack(xs_raw), mt, sets,
                                                  rng, args.aug, DEV)], dim=2)
                Yq = torch.cat([Yq, synth_targets(torch.stack(xq_raw), mt, sets,
                                                  rng, args.aug, DEV)], dim=2)
            opt.zero_grad()
            loss = closed_form_loss(torch.stack(ps), Ys, torch.stack(pq), Yq,
                                    log_lam.exp())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        v = val_loss()
        if v < best - 1e-6:
            best, best_state, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
        if (ep + 1) % 25 == 0:
            print(f"  ep{ep+1} val={v:.5f} best={best:.5f} "
                  f"lam={float(log_lam.exp()):.3f} [{time.time()-t0:.0f}s]", flush=True)

    model.load_state_dict(best_state)
    out = ROOT / "outputs" / f"oracle_l1{args.tag}_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "seed": args.seed, "nG": nG, "nC": nC,
                "val_loss": best, "epochs_run": ep + 1, "aug": args.aug,
                "lam": float(log_lam.exp())}, out)
    msg = (f"[L1 seed{args.seed} aug{args.aug}] ep={ep+1}/{n_epoch} val={best:.5f} "
           f"lam={float(log_lam.exp()):.3f} [{time.time()-t0:.0f}s] -> {out.name}")
    print(msg, flush=True)
    log("L1", "DONE", note=msg)


if __name__ == "__main__":
    main()
