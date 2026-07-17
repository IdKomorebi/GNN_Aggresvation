#!/usr/bin/env python3
"""DNN56 最小验证：general 字段对的"协同泄露"是否存在（单看都低、组合起来高）。

用最佳推断模型 gcn_dynamic 当 oracle：
- leakage(S) = 只给攻击者字段集合 S（其余 general 置 0）时，12-conf 的 a_c 加权平均 R²。
- 充分性视角（真·协同）:
    single(g)      = leakage({g})
    pair(g1,g2)    = leakage({g1,g2})
    synergy        = pair − max(single(g1), single(g2))   （互补增益）
    目标 = "both single 低 & pair 高" = 两个看着无害的字段合起来才泄露。
- 必要性视角（冗余/替代）:
    eff(g)         = base − leakage(all\{g})              （去掉 g 的损失）
    eff_pair       = base − leakage(all\{g1,g2})
    superadd       = eff_pair − eff(g1) − eff(g2)         （>0：两者互为替代，需同时删）
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT))

from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples


def load():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"],
                                symmetrize=True, n_general=di["n_general"], bipartite=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"],
        n_general=di["n_general"], confidential_indices=di["confidential_indices"],
        hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"],
        input_dim=1, architecture=m["architecture"], attention_dropout=m["attention_dropout"],
        attention_dim=m["attention_dim"], attention_temperature=m["attention_temperature"],
        edge_alpha_temperature=m["edge_alpha_temperature"], alpha_init_std=m["alpha_init_std"],
        prior_scale_init=m["prior_scale_init"], gate_bias_init=m["gate_bias_init"],
        prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
        input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(dev)
    model.load_state_dict(torch.load(ROOT / "inference_model/model.pt", map_location=dev))
    model.eval()
    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    X = torch.as_tensor(feat, dtype=torch.float32, device=dev)
    y = torch.as_tensor(tgt, dtype=torch.float32, device=dev)
    return model, X, y, di, dev


def per_conf_r2(pred, y):
    p, t = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
    ss_res = ((t - p) ** 2).sum(0); ss_tot = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    model, X, y, di, dev = load()
    gi = np.array(di["general_indices"]); names = di["general"]; nG = di["n_general"]
    with torch.no_grad():
        a_c = np.clip(per_conf_r2(model(X), y), 0.0, None)
    aw = a_c / a_c.sum()
    print(f"oracle gcn_dynamic: full-field 12-conf 平均 R² = {a_c.mean():.4f}")

    def leak(active_local):                       # active_local: general 局部索引集合
        Xs = torch.zeros_like(X)
        on = gi[list(active_local)]
        if len(on) > 0:
            Xs[:, on, :] = X[:, on, :]
        with torch.no_grad():
            r2 = np.clip(per_conf_r2(model(Xs), y), 0.0, None)
        return float((aw * r2).sum())

    def leak_mask(remove_local):                  # 全字段去掉 remove_local
        Xs = X.clone()
        Xs[:, gi[list(remove_local)], :] = 0.0
        with torch.no_grad():
            r2 = np.clip(per_conf_r2(model(Xs), y), 0.0, None)
        return float((aw * r2).sum())

    base = float((aw * a_c).sum())
    all_set = set(range(nG))

    # ---- 单字段 ----
    single = np.array([leak({i}) for i in range(nG)])           # 充分性
    eff = np.array([base - leak_mask({i}) for i in range(nG)])  # 必要性

    # ---- 字段对 ----
    pairs = list(itertools.combinations(range(nG), 2))
    rows = []
    for i, j in pairs:
        pl = leak({i, j})
        el = base - leak_mask({i, j})
        synergy = pl - max(single[i], single[j])               # 充分性互补增益
        superadd = el - eff[i] - eff[j]                        # 必要性超可加
        rows.append((i, j, pl, el, synergy, superadd))
    df = pd.DataFrame(rows, columns=["i", "j", "pair_leak", "pair_effect", "synergy", "superadd"])
    df["gi"] = [names[i] for i in df["i"]]; df["gj"] = [names[j] for j in df["j"]]
    df["single_i"] = single[df["i"].to_numpy()]
    df["single_j"] = single[df["j"].to_numpy()]
    df["max_single"] = np.maximum(df["single_i"], df["single_j"])
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    df.sort_values("synergy", ascending=False).to_csv(out / "pair_synergy.csv", index=False)

    # ---- 真·协同：both single 低 & pair 高 ----
    LOW = 0.30
    true_syn = df[(df["single_i"] < LOW) & (df["single_j"] < LOW)].sort_values("pair_leak", ascending=False)
    print(f"\n=== 充分性协同 top8（按 synergy=pair−max(single) 排序）===")
    print(f"{'g1':26s}{'g2':22s}{'s_i':>7s}{'s_j':>7s}{'pair':>7s}{'syn':>7s}")
    for _, r in df.sort_values("synergy", ascending=False).head(8).iterrows():
        print(f"{r['gi'][:25]:26s}{r['gj'][:21]:22s}{r['single_i']:>7.3f}{r['single_j']:>7.3f}"
              f"{r['pair_leak']:>7.3f}{r['synergy']:>7.3f}")
    print(f"\n=== 真·协同（两个 single 都 <{LOW}，但 pair 高）top8 ===")
    if len(true_syn):
        print(f"{'g1':26s}{'g2':22s}{'s_i':>7s}{'s_j':>7s}{'pair':>7s}")
        for _, r in true_syn.head(8).iterrows():
            print(f"{r['gi'][:25]:26s}{r['gj'][:21]:22s}{r['single_i']:>7.3f}{r['single_j']:>7.3f}{r['pair_leak']:>7.3f}")
    else:
        print("  （无）")

    print(f"\n=== 必要性超可加 top6（冗余/替代对：需同时删）===")
    print(f"{'g1':26s}{'g2':22s}{'eff_i':>7s}{'eff_j':>7s}{'eff_pair':>9s}{'super':>8s}")
    for _, r in df.sort_values("superadd", ascending=False).head(6).iterrows():
        print(f"{r['gi'][:25]:26s}{r['gj'][:21]:22s}{eff[int(r['i'])]:>7.3f}{eff[int(r['j'])]:>7.3f}"
              f"{r['pair_effect']:>9.3f}{r['superadd']:>8.3f}")

    # ---- 定量判据 ----
    n_true = int(((df["max_single"] < LOW) & (df["pair_leak"] > 0.5)).sum())
    n_big_syn = int((df["synergy"] > 0.1).sum())
    med_syn = float(df["synergy"].median())
    summary = {
        "full_field_R2": base,
        "n_pairs": len(pairs),
        "synergy_median": med_syn,
        "synergy_max": float(df["synergy"].max()),
        "n_pairs_synergy>0.1": n_big_syn,
        "n_true_synergy(max_single<0.3 & pair>0.5)": n_true,
        "superadd_max": float(df["superadd"].max()),
        "n_superadd>0.05": int((df["superadd"] > 0.05).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 图1：max_single vs pair_leak 散点 ----
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(df["max_single"], df["pair_leak"], c=df["synergy"], cmap="viridis", s=14)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="pair = max(single) (无协同)")
    ax.axvline(LOW, color="red", ls=":", alpha=0.6, label=f"max_single={LOW}")
    ax.set_xlabel("max(single leak of the two fields)")
    ax.set_ylabel("pair leak (both fields together)")
    ax.set_title("DNN56: pairwise synergy — points top-left = individually low, jointly high")
    fig.colorbar(sc, label="synergy = pair − max(single)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "synergy_scatter.png", dpi=140); plt.close(fig)

    print("\n=== 定量判据 ===")
    print(f"  字段对总数: {len(pairs)}")
    print(f"  synergy 中位数={med_syn:.4f}  最大={df['synergy'].max():.4f}")
    print(f"  synergy>0.1 的对数: {n_big_syn}")
    print(f"  真·协同(两 single<0.3 且 pair>0.5)的对数: {n_true}")
    print(f"  必要性超可加>0.05 的对数: {summary['n_superadd>0.05']}  最大={summary['superadd_max']:.4f}")
    print(f"\n图: {out}/synergy_scatter.png ; 表: pair_synergy.csv ; summary.json")


if __name__ == "__main__":
    main()
