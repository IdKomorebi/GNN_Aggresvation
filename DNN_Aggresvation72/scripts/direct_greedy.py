# -*- coding: utf-8 -*-
"""DNN72 Part A：oracle 直接贪心防护（补 71 号防护缺口）。

贪心目标从"低阶覆盖收益"改为直接最小化 oracle 实测的最坏机密泄露：
  每步从剩余字段中选使 max_c R2(共享∖{f}) 下降最多的 f。
两级预算：内循环 K=50 选字段（70 号已证 K50 排序进平台，0.8s/集合），
选定后 K=200 复核记录。两条曲线：
  - direct : 纯 oracle 贪心，从 k=0 起；
  - hybrid : 低阶贪心热启动到 k*=24（71 号协议重算，确定性），之后 oracle 贪心续跑。
终止：k 到 KCAP=30 或 K200 最坏泄露 ≤ TAU。
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
BATCH, LR, SEED = 256, 1e-3, 0
TAU, KCAP = 0.5, 30
K_INNER, K_CERT = 50, 200


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

ckpt = torch.load(R69 / "outputs/oracle_gnn_seed0.pt", map_location=DEV, weights_only=False)
metric = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
gc = cfg["graph"]
edge_mask = build_edge_mask(metric, top_k=gc["top_k"], threshold=gc["threshold"],
                            symmetrize=True, n_general=nG, bipartite=True)
a_gg, prior_cg = build_priors(metric, edge_mask, nG)


def eval_shared(fields, K):
    if not fields:
        return np.zeros(nC)
    sel = [name2local[f] for f in fields]
    mask = torch.zeros(1, nG, device=DEV); mask[0, sel] = 1.0
    mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GNNOracle(a_gg, prior_cg, nC, hidden=ckpt.get("gnn_hidden", 128),
                      n_layers=ckpt.get("gnn_layers", 3))
    model.load_state_dict(ckpt["state"]); model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    rng = np.random.RandomState(SEED); step = 0
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


# ---- 低阶贪心前缀（71 号协议重算，确定性）----
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


def low_order_greedy(budget):
    P, rem, fc = [], set(FIELDS), 0.0
    for _ in range(budget):
        bf, bg = None, -1
        for f in sorted(rem):
            g = cov(P + [f]) - fc
            if g > bg: bg, bf = g, f
        P.append(bf); rem.discard(bf); fc = cov(P)
    return P


LOW24 = low_order_greedy(24)
print("低阶贪心 k*=24 前缀:", LOW24, flush=True)


def oracle_greedy(P0, strategy_name, kcap=KCAP):
    """从防护集 P0 出发，oracle 贪心续跑，返回逐步记录。"""
    P = list(P0)
    rows = []
    t0 = time.time()
    # 起点复核
    shared = [f for f in FIELDS if f not in set(P)]
    r2 = eval_shared(shared, K_CERT)
    rows.append(dict(strategy=strategy_name, k=len(P), chosen=None,
                     worst_k200=float(r2.max()), mean_k200=float(r2.mean()),
                     elapsed=time.time() - t0))
    print(f"[{strategy_name}] k={len(P)} worst={r2.max():.3f} mean={r2.mean():.3f}", flush=True)
    while len(P) < kcap and rows[-1]["worst_k200"] > TAU:
        cand = [f for f in FIELDS if f not in set(P)]
        best_f, best_w = None, 1e9
        for f in cand:
            shared_f = [x for x in cand if x != f]
            w = float(eval_shared(shared_f, K_INNER).max())
            if w < best_w:
                best_w, best_f = w, f
        P.append(best_f)
        shared = [f for f in FIELDS if f not in set(P)]
        r2 = eval_shared(shared, K_CERT)
        rows.append(dict(strategy=strategy_name, k=len(P), chosen=best_f,
                         worst_k200=float(r2.max()), mean_k200=float(r2.mean()),
                         elapsed=time.time() - t0))
        print(f"[{strategy_name}] k={len(P)} +{best_f} worst_k50={best_w:.3f} "
              f"worst_k200={r2.max():.3f} mean={r2.mean():.3f} [{time.time()-t0:.0f}s]", flush=True)
    return P, rows


all_rows = []
# 混合：低阶 24 + oracle 续跑（便宜，先跑先看）
P_h, rows_h = oracle_greedy(LOW24, "hybrid")
all_rows += rows_h
json.dump(P_h, open(OUT / "protection_hybrid.json", "w"))
pd.DataFrame(all_rows).to_csv(OUT / "direct_greedy_curve.csv", index=False)

# 纯 oracle 贪心：从 0 起
P_d, rows_d = oracle_greedy([], "direct")
all_rows += rows_d
json.dump(P_d, open(OUT / "protection_direct.json", "w"))
pd.DataFrame(all_rows).to_csv(OUT / "direct_greedy_curve.csv", index=False)
print("\nPart A 完成。")
