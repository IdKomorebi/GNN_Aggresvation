# -*- coding: utf-8 -*-
"""DNN72 Part C：重训证书——对推荐防护集的共享补集做专用 DNN 重训认证。

测量纪律：结论以重训证书落地。协议与 68 号 retrain_worker 完全一致：
DNN(h128×3, dropout0.15), Adam lr1e-3 wd5e-4, batch128, 400ep patience120,
数据切分 seed42, 训练 seed 可控（默认 0，low24 另跑 seed1/2 给误差感受）。

认证对象：
  low24   : 72 确定性低阶贪心 k=24（71 主结果的确定性版本）
  hybrid  : 低阶24 + oracle 精修（达到 τ 的推荐防护集）
  direct  : 纯 oracle 贪心最终集
  direct24: 纯 oracle 贪心同预算 k=24 前缀（与 low24 同预算对比）
输出 oracle 估计 vs 重训证书的逐集合对照。
"""
import json, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch import nn

R69 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69")
OUT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation72/outputs")
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 120


def per_conf_r2(p, t):
    ss = ((t - p) ** 2).sum(0); st = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


class DNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, n_out))

    def forward(self, x): return self.net(x)


def train_generic(model, Xtr, Ytr, Xte, Yte):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best = 1e9; bs = None; pat = 0
    n = len(Xtr)
    for ep in range(EPOCHS):
        model.train(); pm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            ix = pm[k:k + 128]; opt.zero_grad()
            loss = ((model(Xtr[ix]) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Xte).cpu().numpy() - Yte) ** 2).mean())
        if v < best: best = v; bs = deepcopy(model.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    model.load_state_dict(bs); model.eval()
    return model


cfg = yaml.safe_load((R69 / "base.yaml").read_text())
cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
torch.manual_seed(42); np.random.seed(42)
di = prepare_data(cfg)
gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
name2local = {n: i for i, n in enumerate(di["general"])}
FIELDS = sorted(di["general"])
tr, te = di["train_data"], di["test_data"]
Yte_np = te[:, ci]


def certify(shared_fields, seed=0):
    sel = gi[[name2local[f] for f in shared_fields]]
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.as_tensor(tr[:, sel], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, sel], dtype=torch.float32, device=DEV)
    model = DNN(len(sel), len(ci)).to(DEV)
    model = train_generic(model, Xtr, Ytr, Xte, Yte_np)
    with torch.no_grad():
        return per_conf_r2(model(Xte).cpu().numpy(), Yte_np)


# ---- 防护集 ----
low24 = json.load(open(OUT / "protection_low24.json"))
hybrid = json.load(open(OUT / "protection_hybrid.json"))
direct = json.load(open(OUT / "protection_direct.json"))
curve = pd.read_csv(OUT / "direct_greedy_curve.csv")

sets = {
    "low24": (low24, [0, 1, 2]),
    "hybrid_final": (hybrid, [0]),
    "direct_final": (direct, [0]),
    "direct24": (direct[:24], [0]),
}
# oracle 口径（K200）对照值
ocl = {}
for _, r in curve.iterrows():
    ocl[(r.strategy, int(r.k))] = (r.worst_k200, r.mean_k200)
oracle_ref = {
    "low24": ocl.get(("hybrid", 24)),
    "hybrid_final": ocl.get(("hybrid", len(hybrid))),
    "direct_final": ocl.get(("direct", len(direct))),
    "direct24": ocl.get(("direct", 24)),
}

rows = []
t0 = time.time()
for name, (P, seeds) in sets.items():
    shared = [f for f in FIELDS if f not in set(P)]
    for s in seeds:
        r2 = certify(shared, s)
        oref = oracle_ref.get(name)
        rows.append(dict(set_name=name, k=len(P), n_shared=len(shared), seed=s,
                         worst_retrain=float(r2.max()), mean_retrain=float(r2.mean()),
                         worst_oracle=None if oref is None else float(oref[0]),
                         mean_oracle=None if oref is None else float(oref[1])))
        print(f"{name} k={len(P)} seed={s}: 重训 worst={r2.max():.3f} mean={r2.mean():.3f} "
              f"| oracle worst={oref[0] if oref else float('nan'):.3f} [{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "retrain_certificates.csv", index=False)
print("\nPart C 完成。")
