"""DNN65 Reptile 元学习：把"特征子集=任务"，训练一个"为 K 步适应而生"的初始化。

目标 min_θ E_S[ loss(U_K(θ; S)) ]，U_K = 从 θ 在按 S 遮蔽的训练集上微调 K_inner 步。
Reptile 一阶近似：每个 meta-step 采样任务 S，内循环得 θ'，meta 更新 θ ← θ + ε(θ' − θ)。

meta-train 只用随机掩码（k~U{1..44} 再均匀抽 k 个），从不碰 60 号的 50 个固定 rs 子集，
所以对它们是干净的泛化测试。
用法：meta_train.py [--warm-start] [--k-inner 10] [--meta-iters 3000] [--seed 0]
输出：outputs/reptile_init[_ws]_seed{S}.pt
"""
from __future__ import annotations

import argparse, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle, sample_mask

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INNER_BATCH = 256
META_BATCH = 5           # 每个 meta-step 平均多少个任务的更新（稳定）
INNER_LR = 1e-3


def per_conf_r2_mean(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.clip(1 - ss / st, 0, None).mean())


def inner_adapt(model, Xtr, Ytr, mS, k_inner, rng):
    """从当前 model 参数出发内循环 k_inner 步，返回适应后的 state_dict（不改原 model）。"""
    fast = MLPOracle(model.nG_, model.nC_).to(DEV)
    fast.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(fast.parameters(), lr=INNER_LR, weight_decay=5e-4)
    n = len(Xtr)
    m_full = mS.expand(INNER_BATCH, -1)
    for _ in range(k_inner):
        ix = torch.as_tensor(rng.choice(n, size=INNER_BATCH, replace=True), device=DEV)
        fast.train(); opt.zero_grad()
        loss = ((fast(Xtr[ix], mS.expand(len(ix), -1)) - Ytr[ix]) ** 2).mean()
        loss.backward(); opt.step()
    return fast.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", action="store_true", help="从 63 oracle 初始化 meta-init")
    ap.add_argument("--k-inner", type=int, default=10)
    ap.add_argument("--meta-iters", type=int, default=3000)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    tr = di["train_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = MLPOracle(nG, nC).to(DEV)
    model.nG_, model.nC_ = nG, nC
    if args.warm_start:
        st = torch.load(ROOT / "outputs/oracle_mlp_seed0.pt", map_location=DEV, weights_only=False)["state"]
        model.load_state_dict(st)

    mask_rng = np.random.RandomState(1000 + args.seed)
    inner_rng = np.random.RandomState(7000 + args.seed)
    t0 = time.time()
    for it in range(args.meta_iters):
        theta0 = deepcopy(model.state_dict())
        # meta-batch：平均多个任务的 (θ' − θ0)
        accum = {k: torch.zeros_like(v) for k, v in theta0.items()}
        for _ in range(META_BATCH):
            model.load_state_dict(theta0)
            mvec = sample_mask(1, nG, mask_rng)[0]
            mS = torch.as_tensor(mvec, device=DEV).unsqueeze(0)
            adapted = inner_adapt(model, Xtr, Ytr, mS, args.k_inner, inner_rng)
            for k in accum:
                accum[k] += (adapted[k] - theta0[k])
        # Reptile meta 更新：θ ← θ0 + (meta_lr/META_BATCH) Σ(θ' − θ0)
        new_state = {k: theta0[k] + (args.meta_lr / META_BATCH) * accum[k] for k in theta0}
        model.load_state_dict(new_state)
        if (it + 1) % 500 == 0:
            # 监控：随机 held-out 子集上 K=10 内循环后的平均 R²
            vr = np.random.RandomState(55)
            r2s = []
            for _ in range(8):
                mvec = sample_mask(1, nG, vr)[0]
                mS = torch.as_tensor(mvec, device=DEV).unsqueeze(0)
                st_ad = inner_adapt(model, Xtr, Ytr, mS, args.k_inner, np.random.RandomState(1))
                fast = MLPOracle(nG, nC).to(DEV); fast.load_state_dict(st_ad); fast.eval()
                with torch.no_grad():
                    Xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
                    r2s.append(per_conf_r2_mean(fast(Xte, mS.expand(len(Xte), -1)).cpu().numpy(),
                                                di["test_data"][:, ci]))
            print(f"  it={it+1} held-out K10 平均 R²={np.mean(r2s):.4f} ({time.time()-t0:.0f}s)", flush=True)

    tag = "_ws" if args.warm_start else ""
    out = ROOT / "outputs" / f"reptile_init{tag}_seed{args.seed}.pt"
    torch.save({"state": model.state_dict(), "nG": nG, "nC": nC,
                "k_inner": args.k_inner, "meta_iters": args.meta_iters,
                "warm_start": args.warm_start}, out)
    print(f"[reptile{tag} seed{args.seed}] 完成 -> {out.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
