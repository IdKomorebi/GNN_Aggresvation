# -*- coding: utf-8 -*-
"""F-1：二阶协同的**微调相变曲线**——在细 K 网格上扫全部 990 个低阶子集。

问题
----
75_Correction §4.2 用 K∈{0,10,50,200} 定位到"发现最强协同"的相变落在 (10, 50]：
真 top10 协同的估计排名中位数 7724 → 697 → 6。格点太粗，K* 不明。
而 K* 同时决定微调协议与高阶扫描第二级的预算，必须测准。

设计
----
- 子集：全部 990 个低阶子集（44 单 + 946 对）。算
  `ŝyn(i,j,c) = v̂(ij) − max(v̂(i), v̂(j))` 需要三者在**同一 K**，故必须一起扫。
- K 网格：0,5,10,…,80（17 个 checkpoint），相变两侧留余量。
- 方案：{uniform, none, bern50} × seed {0,1,2}。**必须含 none**——主张二的对照：
  "少步数发现"到底是随机失活起点独有，还是微调本身给的。
- 微调协议与 75 号 `eval_variant.py` 逐项一致（BATCH 256 / Adam lr 1e-3 wd 5e-4 /
  同种子），唯一差别是 K 网格与子集范围。

用法：python scripts/fine_ksweep.py --scheme uniform --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R76 = Path(__file__).resolve().parents[1]
R75 = R76.parent / "DNN_Aggresvation75"
R69 = R76.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R76 / "src"))

from src.data_processing import prepare_data   # noqa: E402
from src.oracle import MLPOracle               # noqa: E402
from runlog import log                          # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = list(range(0, 81, 5))                   # 0,5,...,80
BATCH, LR, WD = 256, 1e-3, 5e-4
import yaml                                     # noqa: E402


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    cfg = yaml.safe_load((R76 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    di = prepare_data(cfg)
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

    # 只取 990 个低阶子集（|S|<=2），它们构成二阶协同的完整闭包
    ev = json.load(open(R75 / "outputs/evalset.json"))
    sids = sorted(s for s, m in ev.items() if m["size"] <= 2)
    assert len(sids) == 990, f"低阶子集数应为 990，实际 {len(sids)}"

    rows = []
    for n_done, sid in enumerate(sids, 1):
        sel = [name2local[f] for f in ev[sid]["fields"]]
        mask = torch.zeros(1, nG, device=DEV)
        mask[0, sel] = 1.0
        mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MLPOracle(nG, nC)
        model.load_state_dict(ck["state"])
        model = model.to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        def snap(k):
            model.eval()
            with torch.no_grad():
                r2 = per_conf_r2(model(xte, mte).cpu().numpy(), yte)
            for j, c in enumerate(CONF):
                rows.append(dict(sid=sid, size=ev[sid]["size"], K=k, conf=c,
                                 est=float(r2[j])))

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
        if n_done % 300 == 0:
            print(f"  {args.scheme}/seed{args.seed}: {n_done}/{len(sids)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    out = R76 / "outputs" / f"fine_{args.scheme}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    el = time.time() - t0
    print(f"[{args.scheme} seed{args.seed}] 完成 {len(sids)} 子集 × {len(KGRID)} 个 K "
          f"[{el:.0f}s] -> {out.name}", flush=True)
    log("F-1", "DONE", note=f"{args.scheme} seed{args.seed}：990 子集 × {len(KGRID)} 个 K",
        elapsed_s=el)


if __name__ == "__main__":
    main()
