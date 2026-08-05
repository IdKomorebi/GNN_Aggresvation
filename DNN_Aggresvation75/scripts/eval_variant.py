# -*- coding: utf-8 -*-
"""DNN75：在**扩充后的**评测集（~1576 子集）上评估一个方案，K∈{0,10,50,200}。

与 73/74 的差别只有评测集：从 394（89% 是 |S|<=3、中大带每带仅 10 个）
换成 outputs/evalset.json（中大带每带 44/49/44），否则本子项目的尺寸效应测不出来。
微调协议（同掩码、同 batch、同 Adam/lr、同 K 网格）与 73/74 逐项一致。

用法：python scripts/eval_variant.py --scheme small50 --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R75 = Path(__file__).resolve().parents[1]
R69 = R75.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R75 / "scripts"))

from src.oracle import MLPOracle                              # noqa: E402
from train_variant import load_data, per_conf_r2              # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID, BATCH, LR, WD = [0, 10, 50, 200], 256, 1e-3, 5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    di = load_data()
    gi, ci = np.array(di["general_indices"]), np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    name2local = {n: i for i, n in enumerate(di["general"])}
    CONF = di["confidential"]

    xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)
    xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
    yte = di["test_data"][:, ci]

    ck = torch.load(R75 / "outputs" / f"oracle_{args.scheme}_seed{args.seed}.pt",
                    map_location=DEV, weights_only=False)

    def build():
        m = MLPOracle(nG, nC)
        m.load_state_dict(ck["state"])
        return m.to(DEV)

    evalset = json.load(open(R75 / "outputs/evalset.json"))
    sids = sorted(evalset)
    rows = []
    t0 = time.time()
    for n_done, sid in enumerate(sids, 1):
        meta = evalset[sid]
        sel = [name2local[f] for f in meta["fields"]]
        mask = torch.zeros(1, nG, device=DEV)
        mask[0, sel] = 1.0
        mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build()
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        def snap(k):
            model.eval()
            with torch.no_grad():
                r2 = per_conf_r2(model(xte, mte).cpu().numpy(), yte)
            for j, c in enumerate(CONF):
                rows.append(dict(sid=sid, group=meta["group"], size=meta["size"],
                                 K=k, conf=c, est=float(r2[j])))

        snap(0)
        rng = np.random.RandomState(args.seed)
        step = 0
        while step < max(KGRID):
            order = rng.permutation(len(xtr))
            for b in range(0, len(order), BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
                model.train()
                opt.zero_grad()
                loss = ((model(xtr[ix], mtr[ix]) - ytr[ix]) ** 2).mean()
                loss.backward()
                opt.step()
                step += 1
                if step in KGRID:
                    snap(step)
                if step >= max(KGRID):
                    break
        if n_done % 400 == 0:
            print(f"  {args.scheme}/seed{args.seed}: {n_done}/{len(sids)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    out = R75 / "outputs" / f"est_{args.scheme}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{args.scheme} seed{args.seed}] 完成 {len(sids)} 子集 "
          f"[{time.time()-t0:.0f}s] -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
