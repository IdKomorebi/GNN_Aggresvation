"""DNN68 评测：难度分带专家（按 |S| 路由）vs uniform oracle，在密集真值集上比 K0 与短微调。

真值集合并：60 号 50 随机 + 67 号 990 单/对 + 68 号 85 中段密集。逐 conf MAE，逐尺寸带分解。
配置：uniform（63）/ equal / matched / tilt。K∈{0,10,50}。
路由：|S| 落哪个带用哪个专家（E1=1-5,E2=6-9,E3=10-13,E4=14-17,E5=18-44）。
输出：outputs/banded_eval.csv + banded_curve.png（逐带 K0 误差柱 + 稀释/再分配对比）。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = [0, 10, 50]
BATCH = 256
BANDS = [("E1", 1, 5), ("E2", 6, 9), ("E3", 10, 13), ("E4", 14, 17), ("E5", 18, 44)]
CONFIGS = {"equal": 256, "matched": 104, "tilt": {"E1": 64, "E2": 128, "E3": 128, "E4": 96, "E5": 64}}


def route_band(size):
    for b, lo, hi in BANDS:
        if lo <= size <= hi:
            return b, lo, hi
    raise ValueError(size)


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def load_all_truth(di):
    """合并 60/67/68 的重训真值：{frozenset(fields): {conf: v}}。"""
    truth = {}
    srcs = [
        (ROOT.parent / "DNN_Aggresvation67/outputs/retrain", "*_dnn_seed0.json"),
        (ROOT / "outputs/retrain", "*_dnn_seed0.json"),
    ]
    for d, pat in srcs:
        for f in Path(d).glob(pat):
            j = json.loads(f.read_text())
            truth[frozenset(j["fields"])] = j["per_conf_r2"]
    # 60 号：从 subsets.json + random_eval（v_best 是 12-conf 平均，无逐 conf）——
    # 逐 conf 真值只用 67/68（单/对/中段密集），足够覆盖各带；60 大端用 67/68 不足处补
    return truth


def finetune_eval(state, hidden, nG, nC, Xtr, Ytr, Xte, Yte, mS, conf_names):
    torch.manual_seed(0); np.random.seed(0)
    model = MLPOracle(nG, nC, hidden=hidden).to(DEV); model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    m_tr = mS.expand(len(Xtr), -1); m_te = mS.expand(len(Xte), -1)

    def snap():
        model.eval()
        with torch.no_grad():
            r2 = per_conf_r2(model(Xte, m_te).cpu().numpy(), Yte)
        return {conf_names[j]: float(r2[j]) for j in range(nC)}
    out = {0: snap()}
    n = len(Xtr); step = 0; rng = np.random.RandomState(0); kset = set(KGRID)
    while step < max(KGRID):
        order = rng.permutation(n)
        for b in range(0, n, BATCH):
            ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
            model.train(); opt.zero_grad()
            loss = ((model(Xtr[ix], m_tr[ix]) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step(); step += 1
            if step in kset:
                out[step] = snap()
            if step >= max(KGRID):
                break
    return out


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    general = di["general"]; name2idx = {n: i for i, n in enumerate(general)}
    conf_names = [di["confidential"][j] for j in range(nC)]
    tr, te = di["train_data"], di["test_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV); Yte = te[:, ci]

    truth = load_all_truth(di)
    # 评测子集：所有有逐 conf 真值的（去掉单字段，聚焦 size>=2 的路由差异；单字段也留作 E1）
    items = [(fs, tv) for fs, tv in truth.items()]
    print(f"评测子集数（有逐 conf 真值）: {len(items)}")

    # 预载各专家权重
    experts = {}
    for cname, hspec in CONFIGS.items():
        experts[cname] = {}
        for b, lo, hi in BANDS:
            h = hspec[b] if isinstance(hspec, dict) else hspec
            pt = ROOT / "outputs" / f"expert_{cname}_{b}_h{h}_seed0.pt"
            experts[cname][b] = (torch.load(pt, map_location=DEV, weights_only=False)["state"], h)
    uni = torch.load(ROOT / "outputs/oracle_mlp_seed0.pt", map_location=DEV, weights_only=False)["state"]

    rows = []
    for fs, tv in items:
        idx = [name2idx[f] for f in fs]; size = len(idx)
        mS = torch.zeros(1, nG, device=DEV); mS[0, idx] = 1.0
        b, lo, hi = route_band(size)
        # uniform
        u = finetune_eval(uni, 256, nG, nC, Xtr, Ytr, Xte, Yte, mS, conf_names)
        # 各分带配置（路由到对应带专家）
        res = {"uniform": u}
        for cname in CONFIGS:
            st, h = experts[cname][b]
            res[cname] = finetune_eval(st, h, nG, nC, Xtr, Ytr, Xte, Yte, mS, conf_names)
        for method, byk in res.items():
            for K in KGRID:
                mae = np.mean([abs(byk[K][c] - tv[c]) for c in conf_names])
                rows.append({"size": size, "band": b, "method": method, "K": K, "mae": mae})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/banded_eval.csv", index=False)

    print("\n=== K0 逐带 MAE（越低越好）===")
    piv = df[df.K == 0].pivot_table(index="band", columns="method", values="mae")
    piv = piv.reindex([b for b, _, _ in BANDS])
    print(piv[["uniform", "equal", "matched", "tilt"]].round(4).to_string())
    print("\n=== 全体 mean MAE vs K ===")
    g = df.groupby(["method", "K"])["mae"].mean().unstack("K")
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
