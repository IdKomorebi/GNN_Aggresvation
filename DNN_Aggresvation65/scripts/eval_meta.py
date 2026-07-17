"""DNN65 评测：元学习初始化 vs 冷启动 vs naive 暖启动 的偏差-K 曲线。

对 60 号 50 个固定 rs 子集（meta-train 从未见过的具体掩码），从各初始化微调 K 步、实测 R²。
冷启动 / naive 暖启动 直接复用 64 号已有的 outputs/ft 结果；本脚本只补算 meta 臂。
判据：meta 初始化能否 K≤10 步把 mean|v̂ − truth| 压到 <0.02 且稳定。
用法：eval_meta.py --init outputs/reptile_init_seed0.pt --tag meta
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = [0, 1, 2, 5, 10, 20, 50]
BATCH = 256
INNER_LR = 1e-3


def per_conf_r2_mean(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.clip(1 - ss / st, 0, None).mean())


def finetune_from(state, Xtr, Ytr, Xte, Yte, mS, nG, nC, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPOracle(nG, nC).to(DEV); model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=INNER_LR, weight_decay=5e-4)
    m_tr = mS.expand(len(Xtr), -1); m_te = mS.expand(len(Xte), -1)

    def r2():
        model.eval()
        with torch.no_grad():
            return per_conf_r2_mean(model(Xte, m_te).cpu().numpy(), Yte)
    out = {0: r2()}
    n = len(Xtr); step = 0; rng = np.random.RandomState(seed); kset = set(KGRID)
    while step < max(KGRID):
        order = rng.permutation(n)
        for b in range(0, n, BATCH):
            ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
            model.train(); opt.zero_grad()
            loss = ((model(Xtr[ix], m_tr[ix]) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step(); step += 1
            if step in kset:
                out[step] = r2()
            if step >= max(KGRID):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    general = di["general"]; name2idx = {n: i for i, n in enumerate(general)}
    tr, te = di["train_data"], di["test_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV); Yte = te[:, ci]

    state = torch.load(ROOT / args.init, map_location=DEV, weights_only=False)["state"]
    subs = json.load(open(ROOT.parent / "DNN_Aggresvation60/outputs/subsets.json"))
    rv = pd.read_csv(ROOT.parent / "DNN_Aggresvation60/outputs/random_eval.csv").set_index("subset_id")
    rs = [k for k, v in subs.items() if v["group"] == "random_eval"]

    recs = []
    for sid in rs:
        fields = subs[sid]["fields"]; idx = [name2idx[f] for f in fields]
        mS = torch.zeros(1, nG, device=DEV); mS[0, idx] = 1.0
        truth = float(rv.loc[sid, "v_best"])
        for seed in args.seeds:
            r2 = finetune_from(state, Xtr, Ytr, Xte, Yte, mS, nG, nC, seed)
            for K in KGRID:
                recs.append({"subset": sid, "size": subs[sid]["size"], "seed": seed,
                             "arm": args.tag, "K": K, "r2": r2[K], "truth": truth,
                             "err": r2[K] - truth})
    df = pd.DataFrame(recs)
    df.to_csv(ROOT / f"outputs/meta_eval_{args.tag}.csv", index=False)

    g = df.groupby(["arm", "K"])["err"].apply(lambda e: np.abs(e).mean())
    print(f"=== {args.tag}：mean|v̂(K) − truth| ===")
    for K in KGRID:
        print(f"  K={K:>3}: {g[(args.tag, K)]:.4f}")
    k10 = g[(args.tag, 10)]
    print(f"\n判据 K10<0.02: {'✅通过' if k10 < 0.02 else '❌未过'}（K10={k10:.4f}）")


if __name__ == "__main__":
    main()
