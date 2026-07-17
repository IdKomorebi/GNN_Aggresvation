#!/usr/bin/env python3
"""DNN54 去风险实验：纯图扩散能否揭示直接相关性漏掉的"间接泄露"，并由推断模型验证。

思路
----
- 关联先验图（avg 相关系数，沿用模型的 bipartite 图结构）。confidential 节点按推断
  准确率 a_c 作种子，往外扩散（PPR，无参数，只有 restart β）。
    direct  d_g = 1 跳：g 与 confidential 的直接相关连接（A @ e）。
    diffused s_g = 多跳 PPR：g 经 general-general 间接连到 confidential 的总可达性。
- 推断模型(gcn_dynamic)当 oracle，给出"真实泄露"参照：
    mask_g   : 遮蔽 g 后各 confidential 推断 loss 增幅（necessity，a_c 加权）。
    single_g : 只留 g 时各 confidential 的推断 R²（sufficiency，a_c 加权）。
- 决定性判据：
    (1) 排名一致性 Spearman(direct, diffused, mask, single)。
    (2) top-k 推断 oracle：按各排名取 top-k general（其余置 0），测 12-conf 平均 R²。
        若 diffused 的 top-k 明显优于 direct、并逼近 model-based mask/single → 扩散把
        间接泄露捞了出来，值得做；若 diffused≈direct → 扩散无增量；若更差 → 扩散有害。
    (3) 被扩散"提拔"的字段（high s, low d）是否被模型证实真在泄露（mask/single 高）。
"""
from __future__ import annotations

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


def load_model():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
    m = cfg["model"]; g = cfg["graph"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"],
                                symmetrize=True, n_general=di["n_general"], bipartite=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"],
        n_general=di["n_general"], confidential_indices=di["confidential_indices"],
        hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"],
        input_dim=1, architecture=m["architecture"], attention_dropout=m["attention_dropout"],
        attention_dim=m["attention_dim"], attention_temperature=m["attention_temperature"],
        edge_alpha_temperature=m["edge_alpha_temperature"], alpha_init_std=m["alpha_init_std"],
        prior_scale_init=m["prior_scale_init"], gate_bias_init=m["gate_bias_init"],
        prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
        input_encoder="mlp", bipartite=True, unified_aggregation="gcn_dynamic").to(device)
    model.load_state_dict(torch.load(ROOT / "inference_model/model.pt", map_location=device))
    model.eval()
    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    X = torch.as_tensor(feat, dtype=torch.float32, device=device)
    y = torch.as_tensor(tgt, dtype=torch.float32, device=device)
    return model, X, y, di, mt, edge_mask, device


def per_conf_r2(pred, y):
    p, t = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    model, X, y, di, mt, edge_mask, device = load_model()
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    gen_names = di["general"]; nG = di["n_general"]
    cols = di["all_columns"]

    with torch.no_grad():
        base_pred = model(X)
    a_c = np.clip(per_conf_r2(base_pred, y), 0.0, None)            # (C,) 推断准确率权重
    a_t = torch.as_tensor(a_c, dtype=torch.float32, device=device)
    print(f"推断模型 gcn_dynamic: 12-conf 平均 R² = {a_c.mean():.4f}")

    # ---------- 关联先验图（avg 相关系数 × bipartite 图结构，对称）----------
    avg_corr = np.abs(mt.mean(axis=2)) * edge_mask
    np.fill_diagonal(avg_corr, 0.0)
    A = 0.5 * (avg_corr + avg_corr.T)                              # 对称
    deg = A.sum(1)
    e = np.zeros(len(cols)); e[ci] = a_c; e = e / max(e.sum(), 1e-12)   # confidential 种子

    # direct = 1 跳；diffused = 多跳 PPR
    direct_full = A @ e
    def ppr(beta, iters=200):
        s = e.copy()
        for _ in range(iters):
            s = (1 - beta) * e + beta * (A @ (s / np.maximum(deg, 1e-12)))
        return s
    BETAS = [0.5, 0.7, 0.85]
    diffused_full = {b: ppr(b) for b in BETAS}

    d_direct = direct_full[gi]
    d_diff = {b: diffused_full[b][gi] for b in BETAS}

    # ---------- 推断模型 oracle：mask（necessity）、single（sufficiency）----------
    base_mse = ((base_pred - y) ** 2).mean(0)
    mask_score = np.zeros(nG); single_score = np.zeros(nG)
    for k, j in enumerate(gi):
        Xm = X.clone(); Xm[:, j, :] = 0.0
        with torch.no_grad():
            mse = ((model(Xm) - y) ** 2).mean(0)
        mask_score[k] = float((a_t * (mse - base_mse).clamp(min=0)).sum())
        Xs = torch.zeros_like(X); Xs[:, j, :] = X[:, j, :]
        with torch.no_grad():
            r2 = torch.as_tensor(np.clip(per_conf_r2(model(Xs), y), 0, None),
                                 dtype=torch.float32, device=device)
        single_score[k] = float((a_t * r2).sum())

    # ---------- 排名一致性 ----------
    from scipy.stats import spearmanr
    series = {"direct": d_direct, "diffused(.7)": d_diff[0.7],
              "mask(model)": mask_score, "single(model)": single_score}
    names = list(series)
    print("\n=== 排名一致性 (Spearman) ===")
    print(" " * 14 + "".join(f"{n[:12]:>14s}" for n in names))
    for a in names:
        row = "".join(f"{spearmanr(series[a], series[b]).correlation:>14.3f}" for b in names)
        print(f"{a:14s}{row}")

    # ---------- top-k 推断 oracle ----------
    def topk_curve(score):
        order = np.argsort(-score)                                # general 局部索引降序
        ks = list(range(1, nG + 1))
        out = []
        for kk in ks:
            keep = set(order[:kk].tolist())
            Xs = torch.zeros_like(X)
            for kk2, j in enumerate(gi):
                if kk2 in keep:
                    Xs[:, j, :] = X[:, j, :]
            with torch.no_grad():
                r2 = np.clip(per_conf_r2(model(Xs), y), 0, None)
            out.append(float((a_c * r2).sum() / a_c.sum()))
        return ks, out

    rng = np.random.default_rng(0)
    rankings = {"direct(corr)": d_direct, "diffused(.7)": d_diff[0.7],
                "mask(model)": mask_score, "single(model)": single_score,
                "random": rng.random(nG)}
    curves = {n: topk_curve(s) for n, s in rankings.items()}
    full = a_c.mean()

    # ---------- 被扩散"提拔"的字段 ----------
    def rank(s):  # 1 = 最高
        return pd.Series(-s).rank().to_numpy()
    rd = rank(d_direct); rs = rank(d_diff[0.7])
    promote = rd - rs                                             # >0：扩散把它提上来
    df = pd.DataFrame({"general_field": gen_names,
                       "direct": d_direct, "diffused.7": d_diff[0.7],
                       "rank_direct": rd.astype(int), "rank_diffused": rs.astype(int),
                       "rank_gain": (rd - rs).astype(int),
                       "mask_model": mask_score, "single_model": single_score})
    df = df.sort_values("rank_gain", ascending=False)
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    df.to_csv(out / "diffusion_vs_direct.csv", index=False)

    print("\n=== 被扩散最提拔的 6 个字段（low direct → high diffused），看模型是否证实泄露 ===")
    print(f"{'field':32s}{'rk_dir':>7s}{'rk_dif':>7s}{'mask':>9s}{'single':>9s}")
    for _, r in df.head(6).iterrows():
        print(f"{r['general_field'][:31]:32s}{int(r['rank_direct']):>7d}{int(r['rank_diffused']):>7d}"
              f"{r['mask_model']:>9.4f}{r['single_model']:>9.4f}")
    # single 的中位数作"是否真泄露"参照
    print(f"\n(参照) single_model 全体: 中位数={np.median(single_score):.4f} 最大={single_score.max():.4f}")

    # ---------- 图1：top-k oracle ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"direct(corr)": "#9e9e9e", "diffused(.7)": "#4c72b0",
              "mask(model)": "#e15759", "single(model)": "#59a14f", "random": "#cccccc"}
    styles = {"direct(corr)": "--", "diffused(.7)": "-", "mask(model)": "-.",
              "single(model)": ":", "random": ":"}
    for n, (ks, vv) in curves.items():
        ax.plot(ks, vv, styles[n], color=colors[n], lw=2 if "diffused" in n or "direct" in n else 1.5, label=n)
    ax.axhline(full, ls=":", color="#333", lw=1, label=f"full-field = {full:.4f}")
    ax.set_xlabel("top-k general fields revealed"); ax.set_ylabel("mean conf R2 (a_c weighted)")
    ax.set_title("DNN54: top-k inference oracle — graph-only (direct/diffused) vs model-based")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(1, nG)
    fig.tight_layout(); fig.savefig(out / "topk_oracle.png", dpi=140); plt.close(fig)

    # ---------- 图2：diffused vs direct 散点（颜色=模型 single）----------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(rd, rs, c=single_score, cmap="viridis", s=40)
    for i in range(nG):
        if rd[i] - rs[i] >= 8:                                   # 被明显提拔的标注
            ax.annotate(gen_names[i][:14], (rd[i], rs[i]), fontsize=6)
    ax.plot([1, nG], [1, nG], "k--", alpha=0.4)
    ax.set_xlabel("rank by direct correlation (1=top)")
    ax.set_ylabel("rank by diffused PPR (1=top)")
    ax.invert_xaxis(); ax.invert_yaxis()
    fig.colorbar(sc, label="single sufficiency R2 (model oracle)")
    ax.set_title("DNN54: does diffusion promote fields the model confirms leak?")
    fig.tight_layout(); fig.savefig(out / "diffusion_scatter.png", dpi=140); plt.close(fig)

    # ---------- 决定性数值小结 ----------
    auc = {n: np.trapz(vv, ks) / nG for n, (ks, vv) in curves.items()}
    # 小 k 区(前 10)平均，最能体现"挑得准"
    smallk = {n: np.mean(vv[:10]) for n, (ks, vv) in curves.items()}
    summary = {
        "inference_model_meanR2": float(a_c.mean()),
        "spearman_direct_vs_diffused": float(spearmanr(d_direct, d_diff[0.7]).correlation),
        "spearman_diffused_vs_mask": float(spearmanr(d_diff[0.7], mask_score).correlation),
        "spearman_direct_vs_mask": float(spearmanr(d_direct, mask_score).correlation),
        "spearman_diffused_vs_single": float(spearmanr(d_diff[0.7], single_score).correlation),
        "spearman_direct_vs_single": float(spearmanr(d_direct, single_score).correlation),
        "topk_smallk_mean(first10)": {n: float(v) for n, v in smallk.items()},
        "topk_auc": {n: float(v) for n, v in auc.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== top-k 小 k 区(前10平均) ===")
    for n, v in sorted(smallk.items(), key=lambda x: -x[1]):
        print(f"  {n:16s} {v:.4f}")
    print("\n=== 关键 Spearman ===")
    print(f"  direct↔diffused = {summary['spearman_direct_vs_diffused']:.3f}")
    print(f"  diffused↔mask   = {summary['spearman_diffused_vs_mask']:.3f}   "
          f"direct↔mask = {summary['spearman_direct_vs_mask']:.3f}")
    print(f"  diffused↔single = {summary['spearman_diffused_vs_single']:.3f}   "
          f"direct↔single = {summary['spearman_direct_vs_single']:.3f}")
    print(f"\n图: {out}/topk_oracle.png, diffusion_scatter.png ; 表: diffusion_vs_direct.csv")


if __name__ == "__main__":
    main()
