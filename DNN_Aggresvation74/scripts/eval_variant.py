# -*- coding: utf-8 -*-
"""DNN74：在 394 个分层子集上评估一个图 oracle 变体，K∈{0,10,50,200}。

评测协议与 DNN73/scripts/eval_variants.py **逐项一致**（同子集、同掩码、同 batch、
同 Adam/lr、同 K 网格），唯一差别是被评模型的图架构。这样新变体的每一格
都能与 73 号的 `est_gnn_ours_seed0.csv` / `est_mlp_ours_seed0.csv` 直接对比。

评测子集（`pick_subsets`，原样照抄 73 号，seed 0）：
    全部 44 单字段 + 全部 50 宽尺寸随机集合 + 200 随机字段对 + 100 随机三元组 = 394 个
真值：69 号 `truth_long.csv` 的逐集合重训（dnn 口径）。

用法：
    python scripts/eval_variant.py --variant A1_gcn_dynamic --seed 0
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

R74 = Path(__file__).resolve().parents[1]
R69 = R74.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R74 / "src"))

from src.data_processing import prepare_data                      # noqa: E402
from src.oracle import MLPOracle                                  # noqa: E402
from graph_oracle import MaskedGraphOracle                    # noqa: E402

sys.path.insert(0, str(R74 / "scripts"))
from train_variant import build_graph_inputs, per_conf_r2         # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID, BATCH, LR, WD = [0, 10, 50, 200], 256, 1e-3, 5e-4


def pick_subsets(reg, seed=0):
    """原样照抄 DNN73/scripts/eval_variants.py:pick_subsets —— 必须一字不差，
    否则新旧结果无法逐格对比。"""
    rng = np.random.RandomState(seed)
    by = {}
    for sid, m in reg.items():
        by.setdefault(m["group"], []).append(sid)
    sel = sorted(by["single"]) + sorted(by["wide_random"])
    sel += list(rng.choice(sorted(by["pair"]), 200, replace=False))
    tri = sorted(by.get("triple_top", [])) + sorted(by.get("triple_rand", []))
    sel += list(rng.choice(tri, 100, replace=False))
    return sorted(set(sel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load((R74 / "base.yaml").read_text())
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

    ck = torch.load(R74 / "outputs" / f"oracle_{args.variant}{args.tag}_seed{args.seed}.pt",
                    map_location=DEV, weights_only=False)
    mt, em = build_graph_inputs(cfg, nG)

    def build():
        m = (MLPOracle(nG, nC) if args.variant == "MLP"
             else MaskedGraphOracle(mt, em, nG, nC, **ck["vcfg"]))
        m.load_state_dict(ck["state"])
        return m.to(DEV)

    reg = json.load(open(R69 / "outputs/subsets.json"))
    sids = pick_subsets(reg)
    rows = []
    t0 = time.time()
    for n_done, sid in enumerate(sids, 1):
        fields = reg[sid]["fields"]
        sel = [name2local[f] for f in fields]
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
                rows.append(dict(sid=sid, group=reg[sid]["group"], size=len(fields),
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
        if n_done % 100 == 0:
            print(f"  {args.variant}/seed{args.seed}: {n_done}/{len(sids)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    out = R74 / "outputs" / f"est_{args.variant}{args.tag}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{args.variant}{args.tag} seed{args.seed}] 完成 {len(sids)} 子集 "
          f"[{time.time()-t0:.0f}s] -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
