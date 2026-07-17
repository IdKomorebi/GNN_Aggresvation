#!/usr/bin/env python3
"""DNN52 单方法诊断：为 outputs/<enc>/<mode>/ 生成详细记录。

对每个方法都产出：
  vs_probe.png / vs_probe.csv   ── 该方法 12-conf test R² vs DNN probe 上界
有静态先验(gcn_static/gat_static)：
  alpha_bar.png                 ── 学到的 5 个相关性度量全局静态权重(柱状图)
有动态先验(gcn_dynamic/gat_dynamic)：
  gate_by_edge_long.png         ── 每条 confidential 入边的平均门控(长图)
有 GAT(gat_*)：
  attention_report.md + attention_entropy.png
                                ── 注意力是否起作用 / 是否很平均(归一化熵)

用法：diagnose.py --encoder mlp --mode gat_dynamic
"""
from __future__ import annotations

import argparse
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
from train_variant import ENCODER_TAG

METRICS = ["pearson", "spearman", "kendall", "nmi", "distance_corr"]
STATIC = {"gcn_static", "gat_static"}
DYNAMIC = {"gcn_dynamic", "gat_dynamic"}
GAT = {"gat_noprior", "gat_static", "gat_dynamic"}


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def load(encoder, mode, device):
    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(_resolve(cfg["dataset"]["csv_path"]))
    di = prepare_data(cfg)
    mt = np.load(_resolve(cfg["correlation"]["cache_dir"]) / "metric_tensor.npy")
    g, m = cfg["graph"], cfg["model"]
    edge_mask = build_edge_mask(mt, top_k=g["top_k"], threshold=g["threshold"],
                                symmetrize=True, n_general=di["n_general"], bipartite=True)
    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=di["n_nodes"],
        n_general=di["n_general"], confidential_indices=di["confidential_indices"],
        hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"],
        input_dim=1, architecture=m["architecture"], attention_dropout=m["attention_dropout"],
        attention_dim=m["attention_dim"], attention_temperature=m["attention_temperature"],
        edge_alpha_temperature=m["edge_alpha_temperature"], alpha_init_std=m["alpha_init_std"],
        prior_scale_init=m["prior_scale_init"], gate_bias_init=m["gate_bias_init"],
        prior_log_eps=m["prior_log_eps"], target_specific_heads=True,
        input_encoder=encoder, bipartite=True, unified_aggregation=mode).to(device)
    out_dir = ROOT / "outputs" / ENCODER_TAG[encoder] / mode
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    model.eval()
    feat, tgt = _build_windowed_samples(di["test_data"], di["confidential_indices"], 1)
    X = torch.as_tensor(feat, dtype=torch.float32, device=device)
    y = torch.as_tensor(tgt, dtype=torch.float32, device=device)
    return model, X, y, di, out_dir


def per_conf_r2(pred, y):
    p, t = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def probe_map():
    df = pd.read_csv(ROOT / "outputs/_probe_reference/target_probe_results.csv")
    return dict(zip(df["confidential_field"], df["probe_r2"]))


def plot_vs_probe(out_dir, conf, model_r2, probe, mode, enc_tag):
    order = sorted(range(len(conf)), key=lambda i: -model_r2[i])
    names = [conf[i] for i in order]
    mr = [model_r2[i] for i in order]
    pr = [probe.get(conf[i], np.nan) for i in order]
    pd.DataFrame({"confidential_field": names, "model_r2": mr, "probe_r2": pr,
                  "gap": [a - b for a, b in zip(mr, pr)]}).to_csv(out_dir / "vs_probe.csv", index=False)
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - w / 2, mr, w, label=f"{mode}", color="#4c72b0")
    ax.bar(x + w / 2, pr, w, label="DNN probe (upper bound)", color="#9e9e9e")
    ax.set_xticks(x); ax.set_xticklabels([n[:22] for n in names], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("test R2"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{enc_tag} / {mode}: per-confidential R2 vs DNN probe "
                 f"(mean {np.mean(mr):.4f} vs {np.nanmean(pr):.4f})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "vs_probe.png", dpi=130); plt.close(fig)


def plot_alpha_bar(out_dir, model, mode, enc_tag):
    alpha = model.get_alpha_general().detach().cpu().numpy()
    pd.DataFrame({"metric": METRICS, "alpha": alpha}).to_csv(out_dir / "alpha_static.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(METRICS, alpha, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"])
    for i, v in enumerate(alpha):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("global static alpha"); ax.set_ylim(0, max(alpha) * 1.2)
    ax.set_title(f"{enc_tag} / {mode}: global static prior weights (alpha)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "alpha_bar.png", dpi=130); plt.close(fig)


def plot_gate_long(out_dir, model, di, mode, enc_tag):
    gate = model.get_unified_gate()
    if gate is None:
        return
    gate = gate.cpu().numpy()
    support = model.agg_support.cpu().numpy()
    cols = di["all_columns"]
    rows = []
    for c in di["confidential_indices"]:
        for j in range(di["n_general"]):
            if support[c, j] > 0:
                rows.append({"target": cols[c], "source": cols[j], "gate": float(gate[c, j])})
    df = pd.DataFrame(rows).sort_values("gate", ascending=False)
    df.to_csv(out_dir / "gate_by_edge.csv", index=False)
    labels = (df["target"].str[:16] + " <- " + df["source"].str[:16]).tolist()
    vals = df["gate"].to_numpy()
    fig_h = max(6, 0.18 * len(df))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    yy = np.arange(len(df))
    ax.barh(yy, vals, color="#4c72b0")
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("mean dynamic gate (→1 rely on q·k / →0 rely on prior)")
    ax.set_title(f"{enc_tag} / {mode}: per confidential-edge dynamic gate "
                 f"(mean {vals.mean():.3f}, std {vals.std():.3f})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "gate_by_edge_long.png", dpi=130); plt.close(fig)


def attention_report(out_dir, model, di, mode, enc_tag):
    att = model.get_unified_attention()
    if att is None:
        return
    att = att.cpu().numpy()
    support = model.agg_support.cpu().numpy()
    cols = di["all_columns"]
    rows = []
    for c in di["confidential_indices"]:
        w = att[c].copy()
        valid = support[c] > 0
        deg = int(valid.sum())
        wv = w[valid]
        wv = wv / max(wv.sum(), 1e-12)
        ent = float(-np.sum(wv * np.log(wv + 1e-12)))
        ent_norm = ent / np.log(deg) if deg > 1 else 0.0
        rows.append({"target": cols[c], "degree": deg,
                     "entropy_norm": round(ent_norm, 4),
                     "max_weight": round(float(wv.max()), 4),
                     "uniform_weight": round(1.0 / deg, 4) if deg else np.nan,
                     "top_source": cols[np.where(valid)[0][np.argmax(wv)]]})
    df = pd.DataFrame(rows).sort_values("entropy_norm")
    df.to_csv(out_dir / "attention_entropy.csv", index=False)

    # 熵柱状图（0=完全集中, 1=完全平均）
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(df))
    colors = ["#59a14f" if e < 0.9 else "#e15759" for e in df["entropy_norm"]]
    ax.barh(yy, df["entropy_norm"], color=colors)
    ax.axvline(0.95, color="gray", ls="--", lw=1, label="near-uniform (0.95)")
    ax.set_yticks(yy); ax.set_yticklabels(df["target"].str[:24], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("normalized attention entropy (0=concentrated, 1=uniform)")
    ax.set_title(f"{enc_tag} / {mode}: confidential attention entropy")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "attention_entropy.png", dpi=130); plt.close(fig)

    mean_ent = float(df["entropy_norm"].mean())
    n_uniform = int((df["entropy_norm"] >= 0.95).sum())
    verdict = ("注意力基本平均/退化（接近均匀），未形成区分性路由"
               if mean_ent >= 0.95 else
               "注意力有区分性（明显偏离均匀），起到了选择邻居的作用"
               if mean_ent < 0.85 else
               "注意力弱区分（略偏离均匀）")
    md = [f"# 注意力记录 — {enc_tag} / {mode}", "",
          f"- confidential 目标数: {len(df)}",
          f"- 平均归一化注意力熵: **{mean_ent:.4f}** (1=完全平均, 0=完全集中)",
          f"- 接近均匀(熵≥0.95)的目标数: **{n_uniform}/{len(df)}**",
          f"- 判定: **{verdict}**", "",
          "注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；",
          "越集中(熵小)说明注意力真正在选择信息量大的 general 源。", "",
          "| target | degree | entropy_norm | max_weight | uniform_weight | top_source |",
          "|---|---:|---:|---:|---:|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['target']} | {r['degree']} | {r['entropy_norm']:.4f} | "
                  f"{r['max_weight']:.4f} | {r['uniform_weight']:.4f} | {r['top_source']} |")
    (out_dir / "attention_report.md").write_text("\n".join(md))
    return mean_ent, n_uniform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODER_TAG))
    ap.add_argument("--mode", required=True,
                    choices=["gcn_noprior", "gcn_static", "gcn_dynamic",
                             "gat_noprior", "gat_static", "gat_dynamic"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_tag = ENCODER_TAG[args.encoder]

    model, X, y, di, out_dir = load(args.encoder, args.mode, device)
    with torch.no_grad():
        pred = model(X)                       # 触发 forward，填充注意力/门控诊断
    r2 = np.clip(per_conf_r2(pred, y), -1, None)
    conf = di["confidential"]
    plot_vs_probe(out_dir, conf, r2, probe_map(), args.mode, enc_tag)

    if args.mode in STATIC:
        plot_alpha_bar(out_dir, model, args.mode, enc_tag)
    if args.mode in DYNAMIC:
        plot_gate_long(out_dir, model, di, args.mode, enc_tag)
    if args.mode in GAT:
        attention_report(out_dir, model, di, args.mode, enc_tag)
    print(f"[diag] {enc_tag}/{args.mode}  done  -> {out_dir}")


if __name__ == "__main__":
    main()
