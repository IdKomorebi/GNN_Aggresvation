# -*- coding: utf-8 -*-
"""DNN71 Part B：用 69 号 oracle 验证防护集的"真实"泄露（闭环）。

Part A 在低阶层面用贪心覆盖危险单/对；但 70 号证明"覆盖所有危险对"只保证最强对被压住，
仍留累积残差。本脚本对"共享集合 = 一般字段 ∖ 防护集 P"用 GNN oracle（K=200 微调）
估计每个机密字段的 R²，取 max over conf 作为最坏真实泄露，比较：
  贪心防护集 vs 度基线 vs 单泄露基线 vs 随机，在各预算 k 下的最坏剩余泄露。
验证：(1) 贪心是否在真实泄露上也最快下降；(2) 低阶覆盖阈值 τ 与真实泄露的残差差距。

复用 69 号 oracle 推理代码（同 checkpoint / 数据切分 / K 网格）。
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

R69 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69")
OUT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation71/outputs")
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import GNNOracle, build_priors

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KMAX, BATCH, LR, SEED = 200, 256, 1e-3, 0


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


# ---- 数据与 oracle ----
cfg = yaml.safe_load((R69 / "base.yaml").read_text())
cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
torch.manual_seed(42); np.random.seed(42)
data = prepare_data(cfg)
gidx, cidx = np.asarray(data["general_indices"]), np.asarray(data["confidential_indices"])
nG, nC = data["n_general"], data["n_confidential"]
name2local = {n: i for i, n in enumerate(data["general"])}
CONF_NAMES = data["confidential"]
xtr = torch.as_tensor(data["train_data"][:, gidx], dtype=torch.float32, device=DEV)
ytr = torch.as_tensor(data["train_data"][:, cidx], dtype=torch.float32, device=DEV)
xte = torch.as_tensor(data["test_data"][:, gidx], dtype=torch.float32, device=DEV)
yte = data["test_data"][:, cidx]

ckpt = torch.load(R69 / "outputs/oracle_gnn_seed0.pt", map_location=DEV, weights_only=False)
metric = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
gc = cfg["graph"]
edge_mask = build_edge_mask(metric, top_k=gc["top_k"], threshold=gc["threshold"], symmetrize=True, n_general=nG, bipartite=True)
a_gg, prior_cg = build_priors(metric, edge_mask, nG)


def build_oracle():
    m = GNNOracle(a_gg, prior_cg, nC, hidden=ckpt.get("gnn_hidden", 128), n_layers=ckpt.get("gnn_layers", 3))
    m.load_state_dict(ckpt["state"]); return m.to(DEV)


def eval_shared(fields, K=KMAX):
    """对共享集合 fields 微调 oracle K 步，返回逐 conf R²。"""
    sel = [name2local[f] for f in fields]
    mask = torch.zeros(1, nG, device=DEV); mask[0, sel] = 1.0
    mtr, mte = mask.expand(len(xtr), -1), mask.expand(len(xte), -1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_oracle()
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
        r2 = per_conf_r2(model(xte, mte).cpu().numpy(), yte)
    return r2


# ---- 载入 Part A 的防护集顺序 ----
# 重算贪心/基线顺序（与 protection_greedy 一致，τ=0.5）
reg = json.load(open(R69 / "outputs/subsets.json"))
tl = pd.read_csv(R69 / "outputs/truth_long.csv")
truth = {(r.sid, r.conf): r.dnn for r in tl.itertuples()}
CONFS = sorted(tl.conf.unique())
single_lk, pair_lk, FIELDS = {}, {}, []
for sid, m in reg.items():
    if m["group"] == "single":
        single_lk[m["fields"][0]] = max(truth.get((sid, c), 0.0) for c in CONFS); FIELDS.append(m["fields"][0])
    elif m["group"] == "pair":
        pair_lk[tuple(sorted(m["fields"]))] = max(truth.get((sid, c), 0.0) for c in CONFS)
FIELDS = sorted(set(FIELDS))
TAU = 0.5
dsingle = {f: v for f, v in single_lk.items() if v > TAU}
dpair = {fs: v for fs, v in pair_lk.items() if v > TAU}


def cov(P):
    P = set(P)
    return sum(v for f, v in dsingle.items() if f in P) + sum(v for (a, b), v in dpair.items() if a in P or b in P)


def greedy_order(budget):
    P, rem, fc = [], set(FIELDS), 0.0
    for _ in range(budget):
        bf, bg = None, -1
        for f in rem:
            g = cov(P + [f]) - fc
            if g > bg: bg, bf = g, f
        P.append(bf); rem.discard(bf); fc = cov(P)
    return P


deg = {f: 0.0 for f in FIELDS}
for f, v in dsingle.items(): deg[f] += v
for (a, b), v in dpair.items(): deg[a] += v; deg[b] += v
orders = {
    "greedy": greedy_order(44),
    "degree": sorted(FIELDS, key=lambda f: -deg[f]),
    "single_leak": sorted(FIELDS, key=lambda f: -single_lk[f]),
    "random": list(np.random.RandomState(0).permutation(FIELDS)),
}

# ---- 验证：各策略在预算 k 下，共享补集的最坏真实泄露 ----
budgets = [0, 4, 8, 12, 16, 20, 24]
rows = []
t0 = time.time()
for strat, order in orders.items():
    for k in budgets:
        P = set(order[:k])
        shared = [f for f in FIELDS if f not in P]
        if not shared:
            r2max, r2mean = 0.0, 0.0
        else:
            r2 = eval_shared(shared)
            r2max, r2mean = float(r2.max()), float(r2.mean())
        rows.append(dict(strategy=strat, k=k, n_shared=len(shared),
                         worst_conf_r2=r2max, mean_conf_r2=r2mean))
        print(f"{strat} k={k}: |共享|={len(shared)} 最坏泄露(oracle)={r2max:.3f} 均值={r2mean:.3f} [{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "protection_validation.csv", index=False)
print("\nPart B 完成。")
