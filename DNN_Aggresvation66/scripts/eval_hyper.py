"""DNN66 评测：掩码条件化 HyperOracle 的 K0 保真度 + adapter 精修曲线。

在 60 号 50 个固定真值子集上：
  hyper_k0      : 直接前向（H1：是否优于 63 MLP oracle 的 0.089）
  film_lr1e-2   : 冻结全部，只微调 hypernet(m) 生成的 FiLM 参数（~1.5k 个），大 lr
  film_lr1e-3   : 同上，小 lr
  full_lr1e-3   : 从 HyperOracle 全参微调（对照）
判据（预注册）：H1 hyper K0 < 0.089；H2 film 精修更快/更稳地压偏差（对照 naive 暖启动
K10=0.072、K500≈0.019，及 64 号全参大 lr 崩溃）。
用法：eval_hyper.py --init outputs/hyper_seed0.pt
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.hyper_oracle import HyperOracle

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KGRID = [0, 1, 2, 5, 10, 20, 50, 100, 200]
BATCH = 256


def per_conf_r2_mean(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.clip(1 - ss / st, 0, None).mean())


def finetune(model_state, nG, nC, Xtr, Ytr, Xte, Yte, mS, mode, lr, seed):
    """mode: film(只调生成的 FiLM 张量) / full(全参)。返回 {K: r2}。"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = HyperOracle(nG, nC).to(DEV); model.load_state_dict(model_state)
    m_tr = mS.expand(len(Xtr), -1); m_te = mS.expand(len(Xte), -1)

    if mode == "film":
        model.eval()  # 冻结 dropout/参数行为之外，仍手动控制 train()
        with torch.no_grad():
            film0 = model.film_params(mS)          # (1,H) ×6
        film = [f.clone().detach().requires_grad_(True) for f in film0]
        params = film
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        film = None
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0 if mode == "film" else 5e-4)

    def fwd(X, m):
        if mode == "film":
            fb = [f.expand(len(X), -1) for f in film]
            return model(X, m, film=fb)
        return model(X, m)

    def r2():
        model.eval()
        with torch.no_grad():
            return per_conf_r2_mean(fwd(Xte, m_te).cpu().numpy(), Yte)

    out = {0: r2()}
    n = len(Xtr); step = 0; rng = np.random.RandomState(seed); kset = set(KGRID)
    while step < max(KGRID):
        order = rng.permutation(n)
        for b in range(0, n, BATCH):
            ix = torch.as_tensor(order[b:b + BATCH], device=DEV)
            if mode == "full":
                model.train()
            opt.zero_grad()
            loss = ((fwd(Xtr[ix], m_tr[ix]) - Ytr[ix]) ** 2).mean()
            loss.backward(); opt.step(); step += 1
            if step in kset:
                out[step] = r2()
            if step >= max(KGRID):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="outputs/hyper_seed0.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
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
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV); Yte = te[:, ci]

    state = torch.load(ROOT / args.init, map_location=DEV, weights_only=False)["state"]
    subs = json.load(open(ROOT.parent / "DNN_Aggresvation60/outputs/subsets.json"))
    rv = pd.read_csv(ROOT.parent / "DNN_Aggresvation60/outputs/random_eval.csv").set_index("subset_id")
    rs = [k for k, v in subs.items() if v["group"] == "random_eval"]

    ARMS = [("film_lr1e-2", "film", 1e-2), ("film_lr1e-3", "film", 1e-3),
            ("full_lr1e-3", "full", 1e-3)]
    recs = []
    for sid in rs:
        idx = [name2idx[f] for f in subs[sid]["fields"]]
        mS = torch.zeros(1, nG, device=DEV); mS[0, idx] = 1.0
        truth = float(rv.loc[sid, "v_best"])
        for tag, mode, lr in ARMS:
            for seed in args.seeds:
                r2 = finetune(state, nG, nC, Xtr, Ytr, Xte, Yte, mS, mode, lr, seed)
                for K in KGRID:
                    recs.append({"subset": sid, "size": subs[sid]["size"], "seed": seed,
                                 "arm": tag, "K": K, "r2": r2[K], "truth": truth,
                                 "err": r2[K] - truth})
    df = pd.DataFrame(recs)
    df.to_csv(ROOT / "outputs/hyper_eval.csv", index=False)

    g = df.groupby(["arm", "K"])["err"].apply(lambda e: np.abs(e).mean()).unstack("K")
    print("=== mean|v̂(K) − truth|（50 子集）===")
    print(g.round(4).to_string())
    k0 = g.iloc[0][0]
    print(f"\nH1（K0 条件化质量）: hyper K0={k0:.4f} vs 63 MLP oracle 0.089 "
          f"-> {'✅更好' if k0 < 0.089 else '❌未超'}")


if __name__ == "__main__":
    main()
