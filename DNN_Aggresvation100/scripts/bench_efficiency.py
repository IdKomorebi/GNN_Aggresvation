# -*- coding: utf-8 -*-
"""效率对比（必须对最强基线）：逐个专用重训 / GPU 并行专用重训 / 通用模型（L0、L0ensx）。

单价来源：
  并行重训：本号 RUNLOG 中 TRUTH DONE 的实测墙钟（每卡一个分片，B=1024），另测 B=1/64/256 小批次单价；
  逐个重训：独占单卡实测（一个集合一个 DNN，300 epoch，每 epoch 验证），每数据集 3 个集合（规模 22）；
  通用模型：RUNLOG 中 EST DONE 的 ms/集合（每卡一个分片）；训练成本：uniform 每种子训练秒数（本号复训实测）。
查询规模：1、10³、10⁴、10⁵、完整 K=1（41+820=861 个集合）、K=2（11,521）、K=3（112,791）。
总时间 = 训练成本（通用模型）+ Q × 单价；盈亏平衡 Q* = 训练成本 / (并行单价 − 查询单价)。
单 GPU 口径（多卡时三种方法同比例缩短）。
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT.parent / "DNN_Aggresvation98/src")); sys.path.insert(0, str(ROOT / "src"))
from common100 import load
from batch_truth import train_batched

ev = [json.loads(l) for l in open(ROOT / "outputs/events.jsonl", encoding="utf-8")]
dev = torch.device("cuda"); out = {}
cached = json.load(open(ROOT / "outputs/analysis/efficiency.json")) if (ROOT / "outputs/analysis/efficiency.json").exists() else {}
for ds in ["pjm", "caiso"]:
    D = load(ds); rng = np.random.RandomState(1)
    tr = [e for e in ev if e["stage"] == "TRUTH" and e["status"] == "DONE" and e["note"].startswith(f"{ds}_k4 ")]
    n_each = [int(e["note"].split("n=")[1].split()[0]) for e in tr]
    par = np.sum([e["sec"] for e in tr]) / np.sum(n_each)
    small = {}
    for B in ([] if ds in cached else [1, 64, 256]):
        M = np.zeros((B, 44), np.float32)
        for b in range(B): M[b, rng.choice(D["active"], rng.randint(1, 5), replace=False)] = 1
        torch.cuda.synchronize(); t = time.time(); train_batched(M, D, 0, dev); torch.cuda.synchronize(); small[B] = (time.time() - t) / B
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev)
    fi, vi = torch.as_tensor(D["fit_idx"], device=dev), torch.as_tensor(D["val_idx"], device=dev); seq = []
    if ds in cached:   # 复用上次独占实测的小批次与逐个重训计时
        small = {int(k): v for k, v in cached[ds]["parallel_small"].items()}; seq = [cached[ds]["sequential"]]
    for r in ([] if ds in cached else range(3)):
        sel = rng.choice(D["active"], 22, replace=False); X = Xtr[:, sel]
        f = nn.Sequential(nn.Linear(22, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 12)).to(dev)
        opt = torch.optim.Adam(f.parameters(), 1e-3, weight_decay=5e-4); torch.cuda.synchronize(); t = time.time()
        for ep in range(300):
            f.train(); o = fi[torch.randperm(len(fi), device=dev)]
            for s in range(0, len(o), 128):
                ix = o[s:s + 128]; opt.zero_grad(); ((f(X[ix]) - Ytr[ix]) ** 2).mean().backward(); opt.step()
            f.eval()
            with torch.no_grad(): float(((f(X[vi]) - Ytr[vi]) ** 2).mean())
        torch.cuda.synchronize(); seq.append(time.time() - t)
    est = {}
    for m in ["L0", "L0ensx", "direct", "L1x"]:
        es = [e for e in ev if e["stage"] == "EST" and e["status"] == "DONE" and e["note"].startswith(f"{ds}_k4 {m} ")]
        if es: est[m] = float(np.mean([e["ms_per_set"] for e in es])) / 1e3
    orc = [e for e in ev if e["stage"] == "ORACLE" and e["status"] == "DONE" and e["note"].startswith(f"{ds} main")]
    train_sec = float(np.mean([e["train_sec"] for e in orc])) if orc else np.nan
    out[ds] = dict(parallel_B1024=par, parallel_small=small, sequential=float(np.mean(seq)), est=est, uniform_train_sec_per_seed=train_sec)

rows = []
Qs = {"1": 1, "10^3": 1e3, "10^4": 1e4, "10^5": 1e5, "完整K=1": 861, "完整K=2": 11521, "完整K=3": 112791}
for ds, o in out.items():
    ts = o["uniform_train_sec_per_seed"]
    for qn, Q in Qs.items():
        pu = o["parallel_small"][1] if Q < 64 else (o["parallel_small"][64] if Q < 256 else (o["parallel_small"][256] if Q < 1024 else o["parallel_B1024"]))
        r = dict(数据集=ds, 查询规模=qn, 逐个重训_小时=Q * o["sequential"] / 3600, 并行重训_分钟=Q * pu / 60)
        for m, cost in [("L0", 1), ("L0ensx", 3)]:
            if m in o["est"]: r[f"{m}_分钟(含训练)"] = (cost * ts + Q * o["est"][m]) / 60
        rows.append(r)
    for m, cost in [("L0", 1), ("L0ensx", 3)]:
        if m in o["est"]:
            gap = o["parallel_B1024"] - o["est"][m]
            out[ds][f"盈亏平衡查询数_{m}_相对并行重训"] = float(cost * ts / gap) if gap > 0 else float("inf")
            out[ds][f"盈亏平衡查询数_{m}_相对逐个重训"] = float(cost * ts / (o["sequential"] - o["est"][m]))
df = pd.DataFrame(rows); df.to_csv(ROOT / "outputs/analysis/efficiency.csv", index=False)
json.dump(out, open(ROOT / "outputs/analysis/efficiency.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)); print(df.to_markdown(index=False, floatfmt=".3f"))
