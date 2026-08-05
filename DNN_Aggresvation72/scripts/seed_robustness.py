# -*- coding: utf-8 -*-
"""DNN72 Part B：oracle 评估的种子稳健性 + 贪心序列单调性复核。

背景：71 Part B 出现 k=16 最坏泄露(0.923) > k=12(0.906) 的非单调；且 71 的贪心
tie-break 依赖 set 迭代顺序（PYTHONHASHSEED，不可复现），72 起改用确定性排序。
本脚本对 72 确定性低阶贪心序列的 k∈{8,12,16,20,24}，用 3 个 oracle 种子
（checkpoint seed = 微调 seed = s）各评一次 K=200，报告最坏/平均泄露的均值±标准差，
判定非单调是否在种子方差内。
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

R69 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69")
OUT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation72/outputs")
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import GNNOracle, build_priors

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, LR, K, TAU = 256, 1e-3, 200, 0.5


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


cfg = yaml.safe_load((R69 / "base.yaml").read_text())
cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
torch.manual_seed(42); np.random.seed(42)
data = prepare_data(cfg)
gidx, cidx = np.asarray(data["general_indices"]), np.asarray(data["confidential_indices"])
nG, nC = data["n_general"], data["n_confidential"]
name2local = {n: i for i, n in enumerate(data["general"])}
xtr = torch.as_tensor(data["train_data"][:, gidx], dtype=torch.float32, device=DEV)
ytr = torch.as_tensor(data["train_data"][:, cidx], dtype=torch.float32, device=DEV)
xte = torch.as_tensor(data["test_data"][:, gidx], dtype=torch.float32, device=DEV)
yte = data["test_data"][:, cidx]
metric = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
gc = cfg["graph"]
edge_mask = build_edge_mask(metric, top_k=gc["top_k"], threshold=gc["threshold"],
                            symmetrize=True, n_general=nG, bipartite=True)
a_gg, prior_cg = build_priors(metric, edge_mask, nG)
CKPTS = {s: torch.load(R69 / f"outputs/oracle_gnn_seed{s}.pt", map_location=DEV, weights_only=False)
         for s in [0, 1, 2]}


def eval_shared(fields, seed):
    sel = [name2local[f] for f in fields]
    mask = torch.zeros(1, nG, device=DEV); mask[0, sel] = 1.0
    mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
    ck = CKPTS[seed]
    torch.manual_seed(seed); np.random.seed(seed)
    model = GNNOracle(a_gg, prior_cg, nC, hidden=ck.get("gnn_hidden", 128),
                      n_layers=ck.get("gnn_layers", 3))
    model.load_state_dict(ck["state"]); model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    rng = np.random.RandomState(seed); step = 0
    while step < K:
        order = rng.permutation(len(xtr))
        for b in range(0, len(order), BATCH):
            ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
            model.train(); opt.zero_grad()
            loss = ((model(xtr[ix], mtr[ix]) - ytr[ix]) ** 2).mean()
            loss.backward(); opt.step(); step += 1
            if step >= K: break
    model.eval()
    with torch.no_grad():
        return per_conf_r2(model(xte, mte).cpu().numpy(), yte)


# ---- 72 确定性低阶贪心序列（与 direct_greedy.py 相同）----
reg = json.load(open(R69 / "outputs/subsets.json"))
tl = pd.read_csv(R69 / "outputs/truth_long.csv")
truth = {(r.sid, r.conf): r.dnn for r in tl.itertuples()}
CONFS = sorted(tl.conf.unique())
single_lk, pair_lk, FIELDS = {}, {}, []
for sid, m in reg.items():
    if m["group"] == "single":
        single_lk[m["fields"][0]] = max(truth.get((sid, c), 0.0) for c in CONFS)
        FIELDS.append(m["fields"][0])
    elif m["group"] == "pair":
        pair_lk[tuple(sorted(m["fields"]))] = max(truth.get((sid, c), 0.0) for c in CONFS)
FIELDS = sorted(set(FIELDS))
dsingle = {f: v for f, v in single_lk.items() if v > TAU}
dpair = {fs: v for fs, v in pair_lk.items() if v > TAU}


def cov(P):
    P = set(P)
    return sum(v for f, v in dsingle.items() if f in P) + \
        sum(v for (a, b), v in dpair.items() if a in P or b in P)


P, rem, fc = [], set(FIELDS), 0.0
for _ in range(24):
    bf, bg = None, -1
    for f in sorted(rem):
        g = cov(P + [f]) - fc
        if g > bg: bg, bf = g, f
    P.append(bf); rem.discard(bf); fc = cov(P)
json.dump(P, open(OUT / "protection_low24.json", "w"))

rows = []
t0 = time.time()
for k in [8, 12, 16, 20, 24]:
    shared = [f for f in FIELDS if f not in set(P[:k])]
    for s in [0, 1, 2]:
        r2 = eval_shared(shared, s)
        rows.append(dict(k=k, seed=s, worst=float(r2.max()), mean=float(r2.mean())))
        print(f"k={k} seed={s}: worst={r2.max():.3f} mean={r2.mean():.3f} [{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "seed_robustness.csv", index=False)
agg = df.groupby("k").agg(worst_mean=("worst", "mean"), worst_std=("worst", "std"),
                          mean_mean=("mean", "mean"), mean_std=("mean", "std"))
print("\n=== 3 种子聚合 ===")
print(agg.round(4).to_string())
agg.to_csv(OUT / "seed_robustness_agg.csv")
