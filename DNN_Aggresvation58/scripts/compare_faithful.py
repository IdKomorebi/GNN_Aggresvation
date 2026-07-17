#!/usr/bin/env python3
"""DNN58 对比：置零版(旧) vs 遮蔽训练版(新) 的 v(S) 忠实性、准确率、Shapley 敏感度。

- 准确率：全字段 12-conf 平均 R²。
- v(S) 忠实性：① 随机子集大小 k 的 leakage 曲线(两版对比)；② 单字段 leakage 散点。
  遮蔽版按构造对任意 S 忠实；看旧版是否在小 S 处系统性失真。
- 敏感度：两版各做 Shapley，比 Spearman / top-10 重合，看忠实 v(S) 是否改变敏感度排名。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PERM = 50


def per_conf_r2(pred, tgt):
    ss = ((tgt - pred) ** 2).sum(0); st = ((tgt - tgt.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]; nG = di["n_general"]; gi = np.array(di["general_indices"])
    gnames = di["general"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"], symmetrize=True,
                                n_general=nG, bipartite=True)

    def build():
        return InferenceDrivenGNN(
            metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"], n_general=nG,
            confidential_indices=di["confidential_indices"], hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"], dropout=m["dropout"], input_dim=1, architecture=m["architecture"],
            attention_dropout=m["attention_dropout"], attention_dim=m["attention_dim"],
            attention_temperature=m["attention_temperature"], edge_alpha_temperature=m["edge_alpha_temperature"],
            alpha_init_std=m["alpha_init_std"], prior_scale_init=m["prior_scale_init"],
            gate_bias_init=m["gate_bias_init"], prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
            input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(DEV)

    old = build(); old.load_state_dict(torch.load(ROOT / "old_model/gcn_dynamic_zeromask.pt", map_location=DEV)); old.eval()
    new = build(); new.load_state_dict(torch.load(ROOT / "trained/gcn_dynamic_masked.pt", map_location=DEV)); new.eval()

    fte, tte = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    Xte = torch.as_tensor(fte, dtype=torch.float32, device=DEV); Yte = tte

    def r2vec(model, active):
        Xs = torch.zeros_like(Xte); on = gi[list(active)]
        if len(on): Xs[:, on, :] = Xte[:, on, :]
        with torch.no_grad():
            return per_conf_r2(model(Xs).cpu().numpy(), Yte)

    # a_c 用各自全字段
    ac = {"old": r2vec(old, set(range(nG))), "new": r2vec(new, set(range(nG)))}
    acc = {k: float(v.mean()) for k, v in ac.items()}

    def leak(model, active, aw):
        return float((aw * r2vec(model, active)).sum())

    # ---- v(S) 忠实性：随机子集大小 k 的 leakage 曲线 ----
    rng = np.random.default_rng(0)
    ks = [1, 2, 4, 8, 16, 24, 32, 44]
    curve = {"old": [], "new": []}
    for k in ks:
        lo, ln = [], []
        for _ in range(20):
            S = set(rng.choice(nG, k, replace=False).tolist())
            lo.append(leak(old, S, ac["old"] / ac["old"].sum()))
            ln.append(leak(new, S, ac["new"] / ac["new"].sum()))
        curve["old"].append(np.mean(lo)); curve["new"].append(np.mean(ln))

    # ---- 单字段 leakage 散点 ----
    single_old = np.array([leak(old, {i}, ac["old"] / ac["old"].sum()) for i in range(nG)])
    single_new = np.array([leak(new, {i}, ac["new"] / ac["new"].sum()) for i in range(nG)])

    # ---- Shapley 两版 ----
    def shapley(model, aw):
        rng2 = np.random.default_rng(42); phi = np.zeros(nG)
        v0 = leak(model, set(), aw)
        for _ in range(N_PERM):
            perm = rng2.permutation(nG); act = set(); prev = v0
            for idx in perm:
                act.add(int(idx)); v = leak(model, act, aw); phi[idx] += v - prev; prev = v
        return phi / N_PERM
    sh_old = shapley(old, ac["old"] / ac["old"].sum())
    sh_new = shapley(new, ac["new"] / ac["new"].sum())
    sp = spearmanr(sh_old, sh_new).correlation
    t_old = set(np.argsort(-sh_old)[:10]); t_new = set(np.argsort(-sh_new)[:10])
    ov = len(t_old & t_new) / 10.0

    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    pd.DataFrame({"general_field": gnames, "shapley_old": sh_old, "shapley_new": sh_new,
                  "single_old": single_old, "single_new": single_new}).to_csv(
        out / "old_vs_masked.csv", index=False)
    summary = {"acc_old": acc["old"], "acc_new": acc["new"],
               "shapley_spearman_old_vs_new": float(sp), "top10_overlap": float(ov),
               "vS_curve_k": ks, "vS_old": curve["old"], "vS_new": curve["new"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ks, curve["old"], "-o", color="#9e9e9e", label="zero-mask (old)")
    axes[0].plot(ks, curve["new"], "-o", color="#c0392b", label="masked-trained (new)")
    axes[0].set_xlabel("random subset size k"); axes[0].set_ylabel("leakage v(S) (a_c weighted R2)")
    axes[0].set_title("v(S) vs subset size: is zero-mask distorted at small S?")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].scatter(single_old, single_new, s=22, color="#4c72b0")
    lim = max(single_old.max(), single_new.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", alpha=0.4)
    axes[1].set_xlabel("single-field leakage (zero-mask)")
    axes[1].set_ylabel("single-field leakage (masked-trained)")
    axes[1].set_title("single-field v({g}): old vs masked"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "faithfulness.png", dpi=140); plt.close(fig)

    print("=== ① 准确率(全字段 12-conf 平均 R²) ===")
    print(f"  置零版(旧)   {acc['old']:.4f}")
    print(f"  遮蔽版(新)   {acc['new']:.4f}   Δ={acc['new']-acc['old']:+.4f}")
    print("\n=== ② v(S) 忠实性：随机子集 leakage 曲线 ===")
    print(f"  {'k':>4s}" + "".join(f"{k:>8d}" for k in ks))
    print(f"  {'old':>4s}" + "".join(f"{v:>8.3f}" for v in curve['old']))
    print(f"  {'new':>4s}" + "".join(f"{v:>8.3f}" for v in curve['new']))
    print("\n=== ③ Shapley 敏感度：忠实 v(S) 是否改变排名 ===")
    print(f"  Spearman(old, new) = {sp:.3f}   top-10 重合 = {ov:.2f}")
    print(f"\n图: {out}/faithfulness.png ; 表: old_vs_masked.csv")


if __name__ == "__main__":
    main()
