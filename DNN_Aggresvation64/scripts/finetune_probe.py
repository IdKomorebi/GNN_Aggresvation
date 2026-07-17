"""DNN64 最小验证：逐查询微调的收敛曲线（暖启动是否"聪明"）。

对每个真值子集 S（固定掩码 m_S），三条臂各微调，记录 R²(K) 与累计耗时：
  cold       : 随机初始化 MLPOracle，只在 S 遮蔽的训练集上训（≈每子集重训，上界基准）
  warm_full  : 从 63 号平均 oracle 全参微调（"朴素暖启动"）
  warm_head  : 从 63 号平均 oracle，冻结主干只调输出头（最省的"聪明微调"）

微调数据 = 现成训练集按 m_S 遮蔽（零成本，无需任何重训真值）；真值仅作靶子算误差。
K 定义为梯度步数，checkpoint 在 [0,1,2,5,10,20,50,100,200]。
用法：finetune_probe.py --subset-id rs00 --seed 0
输出：outputs/ft/{subset}_{seed}.json
"""
from __future__ import annotations

import argparse, json, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.oracle import MLPOracle

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
BATCH = 256
# 各臂：(暖启动?, 只调头?, 学习率)
ARMS = {
    "cold":         (False, False, 2e-3),
    "warm_full":    (True,  False, 1e-3),
    "warm_full_hi": (True,  False, 5e-3),   # 排除"暖启动只是学习率太小"的混淆
    "warm_head":    (True,  True,  1e-3),
}


def per_conf_r2_mean(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.clip(1 - ss / st, 0, None).mean())


def load_subset(sid):
    subs = json.load(open(ROOT.parent / "DNN_Aggresvation60/outputs/subsets.json"))
    if sid in subs:
        return subs[sid]["fields"], subs[sid]["size"], "60"
    raise KeyError(sid)


def run_arm(arm, Xtr, Ytr, Xte, Yte, mS, nG, nC, seed, oracle_state):
    """返回 {K: r2}, {K: cum_time}。"""
    warm, head_only, lr = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPOracle(nG, nC).to(DEV)
    if warm:
        model.load_state_dict(oracle_state)
    if head_only:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.net[-1].parameters():          # 只调最后一层 Linear
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)

    m_tr = mS.expand(len(Xtr), -1)
    m_te = mS.expand(len(Xte), -1)

    def eval_r2():
        model.eval()
        with torch.no_grad():
            return per_conf_r2_mean(model(Xte, m_te).cpu().numpy(), Yte)

    r2, tcum = {}, {}
    r2[0] = eval_r2(); tcum[0] = 0.0
    n = len(Xtr); step = 0; t0 = time.time()
    rng = np.random.RandomState(seed)
    kset = set(KGRID)
    while step < max(KGRID):
        order = rng.permutation(n)
        for b in range(0, n, BATCH):
            ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
            model.train(); opt.zero_grad()
            loss = ((model(Xtr[ix], m_tr[ix]) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step()
            step += 1
            if step in kset:
                tcum[step] = time.time() - t0
                r2[step] = eval_r2()
            if step >= max(KGRID):
                break
    return r2, tcum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    general = di["general"]; name2idx = {n: i for i, n in enumerate(general)}
    tr, te = di["train_data"], di["test_data"]
    Xtr = torch.as_tensor(tr[:, gi], dtype=torch.float32, device=DEV)
    Ytr = torch.as_tensor(tr[:, ci], dtype=torch.float32, device=DEV)
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV)
    Yte = te[:, ci]

    fields, size, _ = load_subset(args.subset_id)
    idx = [name2idx[f] for f in fields]
    mS = torch.zeros(1, nG, device=DEV); mS[0, idx] = 1.0

    oracle_state = torch.load(ROOT / "outputs/oracle_mlp_seed0.pt",
                              map_location=DEV, weights_only=False)["state"]

    out = {"subset_id": args.subset_id, "size": size, "seed": args.seed, "arms": {}}
    for arm in ARMS:
        r2, tcum = run_arm(arm, Xtr, Ytr, Xte, Yte, mS, nG, nC, args.seed, oracle_state)
        out["arms"][arm] = {"r2": r2, "time": tcum}
    (ROOT / "outputs/ft").mkdir(exist_ok=True)
    (ROOT / "outputs/ft" / f"{args.subset_id}_seed{args.seed}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    w0 = out["arms"]["warm_full"]["r2"][0]
    w10 = out["arms"]["warm_full"]["r2"].get(10)
    c200 = out["arms"]["cold"]["r2"][200]
    print(f"[{args.subset_id} size={size} seed{args.seed}] "
          f"warm K0={w0:.3f} K10={w10:.3f} | cold K200={c200:.3f}")


if __name__ == "__main__":
    main()
