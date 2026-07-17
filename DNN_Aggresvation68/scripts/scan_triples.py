"""DNN68 Part B 三阶：估计器扫全部 C(44,3)=13244 三元组，K=200 微调，逐 conf v̂。
按分片并行。之后 analyze 用 67 号 pair/single 真值算三阶增量协同、选 top 认证。
用法：scan_triples.py --shard I --nshard N
输出：outputs/triples/{i}_{j}_{k}.json（精简：每 conf 的 v̂@K200）
"""
from __future__ import annotations

import argparse, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KMAX = 200
BATCH = 256


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
    conf_names = [di["confidential"][j] for j in range(nC)]
    tr, te = di["train_data"], di["test_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV); Yte = te[:, ci]
    oracle = torch.load(ROOT / "outputs/oracle_mlp_seed0.pt", map_location=DEV, weights_only=False)["state"]

    out_dir = ROOT / "outputs/triples"; out_dir.mkdir(exist_ok=True)
    triples = [t for idx, t in enumerate(combinations(range(nG), 3)) if idx % args.nshard == args.shard]

    for (i, j, k) in triples:
        fn = out_dir / f"{i:02d}_{j:02d}_{k:02d}.json"
        if fn.exists():
            continue
        mS = torch.zeros(1, nG, device=DEV); mS[0, [i, j, k]] = 1.0
        m_tr = mS.expand(len(Xtr), -1); m_te = mS.expand(len(Xte), -1)
        torch.manual_seed(0); np.random.seed(0)
        model = MLPOracle(nG, nC).to(DEV); model.load_state_dict(oracle)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        step = 0; rng = np.random.RandomState(0); n = len(Xtr)
        while step < KMAX:
            order = rng.permutation(n)
            for b in range(0, n, BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
                model.train(); opt.zero_grad()
                loss = ((model(Xtr[ix], m_tr[ix]) - Ytr[ix]) ** 2).mean()
                loss.backward(); opt.step(); step += 1
                if step >= KMAX:
                    break
        model.eval()
        with torch.no_grad():
            r2 = per_conf_r2(model(Xte, m_te).cpu().numpy(), Yte)
        fn.write_text(json.dumps({"ijk": [i, j, k],
                                  "v": {conf_names[c]: float(r2[c]) for c in range(nC)}}))
    print(f"shard {args.shard}/{args.nshard} 完成 {len(triples)} 三元组")


if __name__ == "__main__":
    main()
