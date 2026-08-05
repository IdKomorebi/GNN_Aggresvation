# -*- coding: utf-8 -*-
"""H-1 第二级：对第一级筛出的候选做 K=K* 微调复核。

第一级（scan_triples_k0.py）用免费的 K=0 全扫 13244 个三元组并按 ŝyn3 排序；
本脚本只对 top-R 比例的候选付微调成本。K* 取 F-1 的结论（76 号 outputs/kstar.json）。

红线：候选选择、排序、复核全程只用 oracle 估计，零重训真值。

用法：python scripts/scan_triples_kstar.py --topfrac 0.30 --shard 0 --nshard 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

R77 = Path(__file__).resolve().parents[1]
R76 = R77.parent / "DNN_Aggresvation76"
R75 = R77.parent / "DNN_Aggresvation75"
R69 = R77.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R77 / "src"))

from src.data_processing import prepare_data   # noqa: E402
from src.oracle import MLPOracle               # noqa: E402
from runlog import log                          # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, LR, WD = 256, 1e-3, 5e-4


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def rank_candidates(topfrac: float):
    """用第一级的 K=0 估计算 ŝyn3，返回 top-R 的三元组列表（纯 oracle）。"""
    low = pd.read_csv(R77 / "outputs/lowfour_k0.csv")
    tri = pd.read_csv(R77 / "outputs/triples_k0.csv")
    l1 = low[low["size"] == 1].set_index(["key", "conf"]).est
    l2 = low[low["size"] == 2].set_index(["key", "conf"]).est

    def key(*ix):
        return "_".join(f"{x:02d}" for x in sorted(ix))

    v1 = {k: v for k, v in l1.items()}
    v2 = {k: v for k, v in l2.items()}
    syn = []
    for r in tri.itertuples():
        i, j, k, c = r.i, r.j, r.k, r.conf
        pairs = [v2.get((key(i, j), c)), v2.get((key(i, k), c)), v2.get((key(j, k), c))]
        if any(p is None for p in pairs):
            syn.append(np.nan)
        else:
            syn.append(r.est - max(pairs))
    tri["syn3_est_k0"] = syn
    tri = tri.dropna(subset=["syn3_est_k0"])
    tri.to_csv(R77 / "outputs/triples_syn3_k0.csv", index=False)

    # 候选以【三元组】为单位：取该三元组在 12 个 conf 上的最大 ŝyn3 作为排序键
    per_tri = tri.groupby(["i", "j", "k"]).syn3_est_k0.max().sort_values(ascending=False)
    n_keep = max(1, int(round(len(per_tri) * topfrac)))
    return list(per_tri.index[:n_keep]), len(per_tri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topfrac", type=float, default=0.30)
    ap.add_argument("--kstar", type=int, default=None, help="缺省读 76 号 kstar.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--scheme", default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    K = args.kstar
    if K is None:
        kp = R76 / "outputs/kstar.json"
        K = int(json.load(open(kp)).get("uniform") or 50) if kp.exists() else 50
    print(f"使用 K* = {K}")

    cand, n_all = rank_candidates(args.topfrac)
    mine = [c for idx, c in enumerate(cand) if idx % args.nshard == args.shard]
    print(f"候选 {len(cand)}/{n_all}（top {args.topfrac:.0%}），本分片 {len(mine)}")

    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
    gi, ci = np.array(di["general_indices"]), np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    CONF = di["confidential"]
    xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)
    xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
    yte = di["test_data"][:, ci]

    ck = torch.load(R75 / "outputs" / f"oracle_{args.scheme}_seed{args.seed}.pt",
                    map_location=DEV, weights_only=False)

    rows = []
    for n_done, (i, j, k) in enumerate(mine, 1):
        mask = torch.zeros(1, nG, device=DEV)
        mask[0, [i, j, k]] = 1.0
        mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MLPOracle(nG, nC)
        model.load_state_dict(ck["state"])
        model = model.to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        rng = np.random.RandomState(args.seed)
        step = 0
        while step < K:
            order = rng.permutation(len(xtr))
            for b in range(0, len(order), BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
                model.train()
                opt.zero_grad()
                loss = ((model(xtr[ix], mtr[ix]) - ytr[ix]) ** 2).mean()
                loss.backward()
                opt.step()
                step += 1
                if step >= K:
                    break
        model.eval()
        with torch.no_grad():
            r2 = per_conf_r2(model(xte, mte).cpu().numpy(), yte)
        for q, c in enumerate(CONF):
            rows.append(dict(i=i, j=j, k=k, conf=c, est=float(r2[q]), K=K))
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(mine)} [{time.time()-t0:.0f}s]", flush=True)

    out = R77 / "outputs" / f"triples_kstar_shard{args.shard}of{args.nshard}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    el = time.time() - t0
    print(f"完成 {len(mine)} 个候选 @ K={K} [{el:.0f}s] -> {out.name}", flush=True)
    log("H-1", "DONE", note=f"第二级 K={K} 复核 shard{args.shard}/{args.nshard}："
                            f"{len(mine)} 个候选", elapsed_s=el)


if __name__ == "__main__":
    main()
