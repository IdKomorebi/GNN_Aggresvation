# -*- coding: utf-8 -*-
"""DNN73：在同一批子集上评估各训练方案变体，K∈{0,10,50,200}。

评测协议与 69 号 estimate_oracle_worker 逐项一致（同掩码、同 batch、同 Adam/lr、同 K 网格），
唯一差别是被评模型来自不同【训练掩码分布】。

评测子集（分层抽样，覆盖全尺寸谱）：
  全部 44 单字段 + 全部 50 宽尺寸随机集合 + 200 随机字段对 + 100 随机三元组 = 394 个
真值：69 号 truth_long.csv 的逐集合重训（dnn 口径）。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

R69 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69")
OUT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation73/outputs")
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import MLPOracle, GNNOracle, build_priors

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID, BATCH, LR = [0, 10, 50, 200], 256, 1e-3


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def pick_subsets(reg, seed=0):
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
    ap.add_argument("--arch", required=True, choices=["mlp", "gnn"])
    ap.add_argument("--scheme", required=True, choices=["none", "bern50", "ours"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((R69 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi, ci = np.array(di["general_indices"]), np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    name2local = {n: i for i, n in enumerate(di["general"])}
    CONF = di["confidential"]
    xtr = torch.as_tensor(di["train_data"][:, gi], dtype=torch.float32, device=DEV)
    ytr = torch.as_tensor(di["train_data"][:, ci], dtype=torch.float32, device=DEV)
    xte = torch.as_tensor(di["test_data"][:, gi], dtype=torch.float32, device=DEV)
    yte = di["test_data"][:, ci]

    ck = torch.load(OUT / f"oracle_{args.arch}_{args.scheme}_seed{args.seed}.pt",
                    map_location=DEV, weights_only=False)
    if args.arch == "gnn":
        mt = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
        em = build_edge_mask(mt, top_k=cfg["graph"]["top_k"], threshold=cfg["graph"]["threshold"],
                             symmetrize=True, n_general=nG, bipartite=True)
        A_gg, prior_cg = build_priors(mt, em, nG)

    def build():
        if args.arch == "mlp":
            m = MLPOracle(nG, nC)
        else:
            m = GNNOracle(A_gg, prior_cg, nC, hidden=ck["gnn_hidden"], n_layers=ck["gnn_layers"])
        m.load_state_dict(ck["state"]); return m.to(DEV)

    reg = json.load(open(R69 / "outputs/subsets.json"))
    sids = pick_subsets(reg)
    rows = []
    t0 = time.time()
    for n_done, sid in enumerate(sids, 1):
        fields = reg[sid]["fields"]
        sel = [name2local[f] for f in fields]
        mask = torch.zeros(1, nG, device=DEV); mask[0, sel] = 1.0
        mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = build()
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

        def snap(k):
            model.eval()
            with torch.no_grad():
                r2 = per_conf_r2(model(xte, mte).cpu().numpy(), yte)
            for j, c in enumerate(CONF):
                rows.append(dict(sid=sid, group=reg[sid]["group"], size=len(fields),
                                 K=k, conf=c, est=float(r2[j])))

        snap(0)
        rng = np.random.RandomState(args.seed); step = 0
        while step < max(KGRID):
            order = rng.permutation(len(xtr))
            for b in range(0, len(order), BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
                model.train(); opt.zero_grad()
                loss = ((model(xtr[ix], mtr[ix]) - ytr[ix]) ** 2).mean()
                loss.backward(); opt.step(); step += 1
                if step in KGRID: snap(step)
                if step >= max(KGRID): break
        if n_done % 100 == 0:
            print(f"  {args.arch}/{args.scheme}: {n_done}/{len(sids)} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"est_{args.arch}_{args.scheme}_seed{args.seed}.csv", index=False)
    print(f"[{args.arch}/{args.scheme}] 完成 {len(sids)} 子集 [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
