#!/usr/bin/env python3
"""DNN49 低数据评估:对某个低数据比例的模型,算 6 种敏感度方法的排名 + 各自全字段
上界 + top-k 曲线。一个 frac 一个进程(四卡并行)。

6 方法(均 a_c=per-confidential test R²≥0 加权):mask/single/attn/ig/shapley/lrp。
(gate 需联合训练且 top-k 不适用,不纳入。)

用法:evaluate_lowdata.py --config configs/frac_XX.yaml --model <model.pt>
输出:outputs/lowdata/<frac>/{ranking.csv, topk.csv}
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from compute_sensitivity import load_model_and_test, per_conf_r2  # noqa


def all_method_scores(model, X, y, data_info, a_c, a_t, device):
    gi = data_info["general_indices"]; ci = data_info["confidential_indices"]
    ng = len(gi); out = {}
    with torch.no_grad():
        base = model(X); base_mse = ((base - y) ** 2).mean(0)
    # mask
    s = np.zeros(ng)
    for k, j in enumerate(gi):
        Xm = X.clone(); Xm[:, j, :] = 0.0
        with torch.no_grad(): mse = ((model(Xm) - y) ** 2).mean(0)
        s[k] = float((a_t * (mse - base_mse).clamp(min=0)).sum())
    out["mask"] = s
    # single
    s = np.zeros(ng)
    for k, j in enumerate(gi):
        Xs = torch.zeros_like(X); Xs[:, j, :] = X[:, j, :]
        with torch.no_grad():
            r2 = torch.tensor(np.clip(per_conf_r2(model(Xs), y), 0, None), device=device)
        s[k] = float((a_t * r2).sum())
    out["single"] = s
    # attn
    with torch.no_grad(): _ = model(X)
    att = torch.stack([la.mean(0) for la in model.layer_attentions], 0).mean(0)  # (N,N)
    out["attn"] = np.array([float((a_c * att[ci, j].cpu().numpy()).sum()) for j in gi])
    # ig
    steps = 32; bl = torch.zeros_like(X); tot = torch.zeros_like(X)
    for st in range(1, steps + 1):
        Xi = (bl + st / steps * (X - bl)).clone().requires_grad_(True)
        g, = torch.autograd.grad((model(Xi) * a_t).sum(), Xi); tot += g.detach()
    ig = ((X - bl) * tot / steps).abs().mean(0).squeeze(-1).cpu().numpy()
    out["ig"] = np.array([ig[j] for j in gi])
    # lrp (freeze attention, input×grad)
    model.freeze_attention = True
    Xi = X.clone().requires_grad_(True)
    g, = torch.autograd.grad((model(Xi) * a_t).sum(), Xi)
    rel = (Xi * g).abs().mean(0).squeeze(-1).detach().cpu().numpy()
    model.freeze_attention = False
    out["lrp"] = np.array([rel[j] for j in gi])
    # shapley
    rng = np.random.default_rng(42); n_perm = 60
    gia = np.array(gi); phi = np.zeros(ng)
    def value(active):
        Xs = torch.zeros_like(X); on = gia[active]
        if len(on): Xs[:, on, :] = X[:, on, :]
        with torch.no_grad(): r2 = np.clip(per_conf_r2(model(Xs), y), 0, None)
        return float((a_c * r2).sum())
    v0 = value(np.zeros(ng, bool))
    for _ in range(n_perm):
        perm = rng.permutation(ng); act = np.zeros(ng, bool); prev = v0
        for idx in perm:
            act[idx] = True; v = value(act); phi[idx] += v - prev; prev = v
    out["shapley"] = phi / n_perm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frac_tag = Path(args.config).stem
    model, X, y, data_info = load_model_and_test(Path(args.config), Path(args.model), device)
    gi = data_info["general_indices"]; gen_names = data_info["general"]
    with torch.no_grad():
        a_c = np.clip(per_conf_r2(model(X), y), 0, None)
    a_t = torch.tensor(a_c, dtype=torch.float32, device=device)

    scores = all_method_scores(model, X, y, data_info, a_c, a_t, device)

    # 排名表(独立标准化 max=1)
    rank = pd.DataFrame({"general_field": gen_names})
    for m, s in scores.items():
        mx = s.max(); rank[f"{m}_norm"] = s / mx if mx > 0 else s
    rank = rank.set_index("general_field")

    # top-k 曲线(同一 frac 模型)
    def mean_r2(positions):
        Xk = torch.zeros_like(X); on = [gi[p] for p in positions]
        if on: Xk[:, on, :] = X[:, on, :]
        with torch.no_grad(): return float(np.mean(per_conf_r2(model(Xk), y)))
    upper = mean_r2(list(range(len(gi))))
    topk = {"k": list(range(1, len(gi) + 1))}
    name2pos = {gen_names[p]: p for p in range(len(gi))}
    for m in scores:
        order = rank.sort_values(f"{m}_norm", ascending=False).index.tolist()
        pos = [name2pos[n] for n in order]
        topk[m] = [mean_r2(pos[:k]) for k in range(1, len(gi) + 1)]

    out_dir = ROOT / "outputs/lowdata" / frac_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rank.to_csv(out_dir / "ranking.csv")
    tdf = pd.DataFrame(topk).set_index("k"); tdf.to_csv(out_dir / "topk.csv")
    (out_dir / "upper.txt").write_text(f"{upper:.6f}\n")
    print(f"[{frac_tag}] 全字段上界 mean R²={upper:.4f}  top10: " +
          ", ".join(f"{m}={tdf[m].iloc[9]:.3f}" for m in scores))


if __name__ == "__main__":
    main()
