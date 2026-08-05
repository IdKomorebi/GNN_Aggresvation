# -*- coding: utf-8 -*-
"""97 号：重做"合成目标增广"——修正 93 号 --aug 的三处缺陷。

为什么要重做
------------
96 号定标出管线的可认证边界 k*=5，并把 k=6/7 的失败**定位到表征**而非阶数：
同一批七阶注入信号，含生成单项式的手工 full 字典能稳定挑到 400 个随机集合的第一名
（CAISO 6/6），而 φ 的中位百分位只有 0.4-6.8。改进路线因此很明确——
**改造 φ 的元训练任务分布**，让读出基不再对未见过的交互型目标分布外。

93 号其实已经试过（`train_l1.py --aug N`），结论是"修不好，容量竞争"，
并被写进论文局限 (1)。但那次实验有三处缺陷，且都要到 96 号之后才看得清：

缺陷 1（致命）：合成目标用 `x[:, idx].prod(dim=1)`，即**原始乘积**。
    96 号已证明真实电网字段的原始乘积重尾极强（六阶峰度中位数 1127），
    被少数极端样本主导，而极端值由单个因子驱动 ⟹ "交互项"退化为单字段的函数。
    也就是说 93 号加进去的所谓"高阶合成目标"，信息量大半只是**单字段信息**，
    φ 本来就已能表达。增广没带来新方向，只稀释了容量——"修不好"是必然的，
    但那不构成"表征无法改造"的证据。
缺陷 2：阶数只取 2-4，而待修的问题在 6-7 阶。
缺陷 3：早停只看 12 个真实机密字段的 val 损失，等于**主动选择反对增广**。

本脚本三处都改，并做成独立开关以便拆开贡献：
  --aug_kind tanh|raw     tanh 为 96 号修正构造；raw 复刻 93 号（消融对照）
  --aug_rmin/--aug_rmax   合成目标的阶数范围
  --val_mode both|conf    both 把合成项计入早停；conf 复刻 93 号

★红线：合成目标只由**可见字段**构造、只进训练；不涉及重训真值，不碰测试集。
判决仍由 96 号的注入测试给出，其元组来自完全独立的 INJ_SEED 流，与训练无交集。
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
EPOCHS, PATIENCE = 2500, 60          # 与 93 号 long 口径一致
LR, WD = 1e-3, 5e-4
N_STEP = 8
B_MASK, N_SUP, N_QRY = 4, 512, 256
RELU_LAST = 7                        # MLPOracle.net 中最后一个 ReLU 的下标


def phi_of(model: MLPOracle, x, m):
    """主干前向到最后一个 ReLU，得 (n,256) 条件特征；不经过共享读出。"""
    h = torch.cat([x * m, m], dim=1)
    for i, layer in enumerate(model.net):
        h = layer(h)
        if i == RELU_LAST:
            return h
    raise RuntimeError


def closed_form_loss(phi_sup, y_sup, phi_qry, y_qry, lam):
    """闭式岭读出 + query 损失，梯度穿过 solve（R2D2 / MetaOptNet）。"""
    mu = phi_sup.mean(1, keepdim=True)
    sd = phi_sup.std(1, keepdim=True).clamp_min(1e-6)
    P, Q = (phi_sup - mu) / sd, (phi_qry - mu) / sd
    ym = y_sup.mean(1, keepdim=True)
    d = P.shape[2]
    eye = torch.eye(d, device=P.device, dtype=P.dtype).expand(P.shape[0], d, d)
    beta = torch.linalg.solve(P.transpose(1, 2) @ P + lam * eye,
                              P.transpose(1, 2) @ (y_sup - ym))
    return ((y_qry - ym - Q @ beta) ** 2).mean()


def draw_tuples(sets, rng, n_aug, rmin, rmax):
    """为每个掩码抽 n_aug 个可见字段元组，阶数在 [rmin, rmax] 内均匀。"""
    out = []
    for sel in sets:
        hi = min(rmax, len(sel))
        lo = min(rmin, hi)
        out.append([rng.choice(sel, size=int(rng.randint(lo, hi + 1)), replace=False)
                    for _ in range(n_aug)])
    return out


def synth_from_tuples(Zt, tups):
    """按元组造合成目标（逐列标准化）。

    ★与 93 号的唯一实质差别在调用方传进来的 Zt：
      tanh 模式 Zt = tanh((x-mu)/sd)，有界单调，乘积不再被极端值主导；
      raw  模式 Zt = x，逐字复刻 93 号行为，用作消融对照。
    """
    cols = []
    for b, tl in enumerate(tups):
        for idx in tl:
            p = Zt[b][:, idx].prod(dim=1)
            cols.append((p - p.mean()) / p.std().clamp_min(1e-6))
    return torch.stack(cols).reshape(len(tups), len(tups[0]), -1).transpose(1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", type=int, default=0, help="每步每掩码的合成目标数")
    ap.add_argument("--aug_kind", default="tanh", choices=["tanh", "raw"])
    ap.add_argument("--aug_rmin", type=int, default=3)
    ap.add_argument("--aug_rmax", type=int, default=7)
    ap.add_argument("--val_mode", default="both", choices=["conf", "both"])
    ap.add_argument("--aug_pair", default="shared", choices=["shared", "mismatch"],
                    help="★缺陷 4：93 号对 support 与 query 各抽一次元组，"
                         "同一列对应**不同**的字段元组，闭式解拟合 A 却在 B 上评估，"
                         "该合成损失项是不可约噪声。mismatch 复刻之，shared 为修正版")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps", type=int, default=N_STEP)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    nG, nC = len(gi), len(ci)
    Xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    Ttr = torch.tanh((Xtr - mu) / sd) if args.aug_kind == "tanh" else Xtr

    n = len(Xtr)
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = max(int(n * 0.15), 1)
    va_idx = torch.as_tensor(perm[:n_val], device=DEV)
    tr_idx = perm[n_val:]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOracle(nG, nC).to(DEV)
    log_lam = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam(list(model.parameters()) + [log_lam], lr=LR, weight_decay=WD)
    rng = np.random.RandomState(1234 + args.seed)

    # ---- 固定验证任务：8 组掩码；合成验证目标的元组也固定（seed 999，与训练流独立）
    vr = np.random.RandomState(999)
    vm = sample_mask(8, nG, vr)
    vsets = [np.where(vm[b] > 0)[0] for b in range(len(vm))]
    vperm = vr.permutation(len(va_idx))
    v_sup = va_idx[torch.as_tensor(vperm[:len(vperm) // 2], device=DEV)]
    v_qry = va_idx[torch.as_tensor(vperm[len(vperm) // 2:], device=DEV)]
    vm_t = torch.as_tensor(vm, device=DEV)
    v_tups = draw_tuples(vsets, vr, 8, args.aug_rmin, args.aug_rmax)

    def val_losses():
        """返回 (真实 conf 损失, 合成高阶损失)。两项恒计算，便于跨臂对比。"""
        model.eval()
        with torch.no_grad():
            ps, pq = [], []
            for b in range(len(vm)):
                ps.append(phi_of(model, Xtr[v_sup], vm_t[b:b + 1].expand(len(v_sup), -1)))
                pq.append(phi_of(model, Xtr[v_qry], vm_t[b:b + 1].expand(len(v_qry), -1)))
            Ps, Pq = torch.stack(ps), torch.stack(pq)
            Ys = Ytr[v_sup].unsqueeze(0).expand(len(vm), -1, -1)
            Yq = Ytr[v_qry].unsqueeze(0).expand(len(vm), -1, -1)
            lc = float(closed_form_loss(Ps, Ys, Pq, Yq, log_lam.exp()))
            Ss = synth_from_tuples(Ttr[v_sup].unsqueeze(0).expand(len(vm), -1, -1), v_tups)
            Sq = synth_from_tuples(Ttr[v_qry].unsqueeze(0).expand(len(vm), -1, -1), v_tups)
            ls = float(closed_form_loss(Ps, Ss, Pq, Sq, log_lam.exp()))
        return lc, ls

    best, best_state, pat, best_pair = 1e9, None, 0, (0.0, 0.0)
    t0 = time.time()
    log("L1HI", "START", note=f"seed{args.seed} aug={args.aug} kind={args.aug_kind} "
                              f"r={args.aug_rmin}-{args.aug_rmax} val={args.val_mode} pair={args.aug_pair} "
                              f"tag={args.tag}")
    ep = 0
    for ep in range(args.epochs):
        model.train()
        for _ in range(args.steps):
            masks = sample_mask(B_MASK, nG, rng)
            sets = [np.where(masks[b] > 0)[0] for b in range(B_MASK)]
            mt = torch.as_tensor(masks, device=DEV)
            ps, pq, ys, yq, ts, tq = [], [], [], [], [], []
            for b in range(B_MASK):
                ix = rng.choice(tr_idx, size=N_SUP + N_QRY, replace=False)
                si = torch.as_tensor(ix[:N_SUP], device=DEV)
                qi = torch.as_tensor(ix[N_SUP:], device=DEV)
                mb = mt[b:b + 1]
                ps.append(phi_of(model, Xtr[si], mb.expand(N_SUP, -1)))
                pq.append(phi_of(model, Xtr[qi], mb.expand(N_QRY, -1)))
                ys.append(Ytr[si])
                yq.append(Ytr[qi])
                ts.append(Ttr[si])
                tq.append(Ttr[qi])
            Ys, Yq = torch.stack(ys), torch.stack(yq)
            if args.aug:
                tups = draw_tuples(sets, rng, args.aug, args.aug_rmin, args.aug_rmax)
                tq_tups = (draw_tuples(sets, rng, args.aug, args.aug_rmin, args.aug_rmax)
                           if args.aug_pair == "mismatch" else tups)
                Ys = torch.cat([Ys, synth_from_tuples(torch.stack(ts), tups)], dim=2)
                Yq = torch.cat([Yq, synth_from_tuples(torch.stack(tq), tq_tups)], dim=2)
            opt.zero_grad()
            closed_form_loss(torch.stack(ps), Ys, torch.stack(pq), Yq,
                             log_lam.exp()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        lc, ls = val_losses()
        v = lc if args.val_mode == "conf" else lc + ls
        if v < best - 1e-6:
            best, best_state, pat, best_pair = v, deepcopy(model.state_dict()), 0, (lc, ls)
        else:
            pat += 1
            if pat >= PATIENCE:
                break
        if (ep + 1) % 50 == 0:
            print(f"  ep{ep+1} conf={lc:.5f} syn={ls:.5f} best={best:.5f} "
                  f"lam={float(log_lam.exp()):.3f} [{time.time()-t0:.0f}s]", flush=True)

    model.load_state_dict(best_state)
    out = ROOT / "outputs" / f"oracle_l1{args.tag}_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "seed": args.seed, "nG": nG, "nC": nC,
                "val_conf": best_pair[0], "val_syn": best_pair[1], "epochs_run": ep + 1,
                "aug": args.aug, "aug_kind": args.aug_kind, "val_mode": args.val_mode,
                "aug_r": [args.aug_rmin, args.aug_rmax], "aug_pair": args.aug_pair,
                "lam": float(log_lam.exp())}, out)
    msg = (f"[{args.tag or 'base'} seed{args.seed}] ep={ep+1} "
           f"val_conf={best_pair[0]:.5f} val_syn={best_pair[1]:.5f} "
           f"lam={float(log_lam.exp()):.3f} [{time.time()-t0:.0f}s] -> {out.name}")
    print(msg, flush=True)
    log("L1HI", "DONE", note=msg)


if __name__ == "__main__":
    main()
