"""DNN67 估计器：对 990 个子集，从 63 号 MLP-oracle 暖启动微调，记录各 K 的
逐 conf 预测 v̂_c(S) 与累计耗时。之后 analyze 做尺寸校准 + 协同排序验证。

固定掩码 m_S 微调（同 64 号 warm_full），checkpoint K∈[0,10,50,100,200,500]。
每个子集输出 outputs/est/{sid}.json：{K: {mean_r2, per_conf_r2[12], time}}。
用法：estimate_ksweep.py --shard I --nshard N   （按子集分片，多 GPU 并行）
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = [0, 10, 50, 100, 200, 500]
BATCH = 256
LR = 1e-3


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
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
    oracle_state = torch.load(ROOT / "outputs/oracle_mlp_seed0.pt",
                              map_location=DEV, weights_only=False)["state"]

    subs = json.load(open(ROOT / "outputs/subsets.json"))
    sids = [s for idx, s in enumerate(sorted(subs)) if idx % args.nshard == args.shard]
    out_dir = ROOT / "outputs/est"; out_dir.mkdir(exist_ok=True)

    for sid in sids:
        if (out_dir / f"{sid}.json").exists():
            continue
        idx = [name2idx[f] for f in subs[sid]["fields"]]
        mS = torch.zeros(1, nG, device=DEV); mS[0, idx] = 1.0
        m_tr = mS.expand(len(Xtr), -1); m_te = mS.expand(len(Xte), -1)
        torch.manual_seed(0); np.random.seed(0)
        model = MLPOracle(nG, nC).to(DEV); model.load_state_dict(oracle_state)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

        rec = {}

        def snap():
            model.eval()
            with torch.no_grad():
                r2 = per_conf_r2(model(Xte, m_te).cpu().numpy(), Yte)
            return {"mean_r2": float(r2.mean()),
                    "per_conf_r2": {di["confidential"][j]: float(r2[j]) for j in range(nC)}}

        rec[0] = {**snap(), "time": 0.0}
        n = len(Xtr); step = 0; rng = np.random.RandomState(0); kset = set(KGRID); t0 = time.time()
        while step < max(KGRID):
            order = rng.permutation(n)
            for b in range(0, n, BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
                model.train(); opt.zero_grad()
                loss = ((model(Xtr[ix], m_tr[ix]) - Ytr[ix]) ** 2).mean()
                loss.backward(); opt.step(); step += 1
                if step in kset:
                    rec[step] = {**snap(), "time": time.time() - t0}
                if step >= max(KGRID):
                    break
        (out_dir / f"{sid}.json").write_text(json.dumps(
            {"subset_id": sid, "group": subs[sid]["group"], "size": subs[sid]["size"],
             "fields": subs[sid]["fields"], "by_k": rec}, ensure_ascii=False))
    print(f"shard {args.shard}/{args.nshard} 完成 {len(sids)} 个子集")


if __name__ == "__main__":
    main()
