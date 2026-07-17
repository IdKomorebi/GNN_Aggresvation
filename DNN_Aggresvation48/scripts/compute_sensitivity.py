#!/usr/bin/env python3
"""DNN47 推断驱动敏感度——四种方法之一，输出每个 general 字段的原始分数。

四种方法（都用推断准确率 a_c = per-confidential test R²≥0 加权）：
  mask   : 必要性。遮蔽 general j 后，各 confidential 的 loss 增幅（a_c 加权）。
  single : 充分性。只保留 general j（其余 general 置 0）时，对各 confidential 的
           推断 R²≥0（a_c 加权）。
  attn   : 路径归因（简化）。训练好的模型逐层 general→confidential 注意力（已含
           gate 混合的先验强度）对 j 的平均权重（a_c 加权）。
  ig     : Integrated Gradients。a_c 加权输出对 general 输入的积分梯度幅值。

用法：compute_sensitivity.py --method {mask,single,attn,ig} --model <model.pt路径>
输出：outputs/sensitivity/<method>_raw.csv  (general_field, raw_score)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT))

from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.train import _build_windowed_samples


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def load_model_and_test(cfg_path: Path, model_pt: Path, device: torch.device):
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["dataset"]["csv_path"] = str(_resolve(cfg["dataset"]["csv_path"]))
    data_info = prepare_data(cfg)

    mt = np.load(_resolve(cfg["correlation"]["cache_dir"]) / "metric_tensor.npy")
    g = cfg.get("graph", {})
    m = cfg.get("model", {})
    edge_mask = build_edge_mask(
        mt, top_k=int(g.get("top_k", 10)), threshold=float(g.get("threshold", 0.08)),
        symmetrize=bool(g.get("symmetrize", True)),
        selection=str(g.get("selection", "threshold_or_topk")),
        min_k=None if g.get("min_k") is None else int(g["min_k"]),
        max_k=None if g.get("max_k") is None else int(g["max_k"]),
        n_general=data_info["n_general"], bipartite=bool(m.get("bipartite", False)),
    )
    win = int(m.get("window_size", 1))
    model = InferenceDrivenGNN(
        metric_tensor=mt, edge_mask=edge_mask, n_nodes=data_info["n_nodes"],
        n_general=data_info["n_general"], confidential_indices=data_info["confidential_indices"],
        hidden_dim=int(m.get("hidden_dim", 32)), num_layers=int(m.get("num_layers", 2)),
        dropout=float(m.get("dropout", 0.05)), input_dim=win,
        architecture=str(m.get("architecture", "gcn")),
        attention_dropout=float(m.get("attention_dropout", 0.05)),
        attention_dim=int(m.get("attention_dim", m.get("hidden_dim", 32))),
        attention_temperature=float(m.get("attention_temperature", 1.0)),
        edge_alpha_temperature=float(m.get("edge_alpha_temperature", 1.0)),
        alpha_init_std=float(m.get("alpha_init_std", 0.0)),
        prior_scale_init=float(m.get("prior_scale_init", 1.0)),
        gate_bias_init=float(m.get("gate_bias_init", -1.0)),
        prior_log_eps=float(m.get("prior_log_eps", 1e-4)),
        target_specific_heads=bool(m.get("target_specific_heads", True)),
        input_encoder=str(m.get("input_encoder", "linear")),
        input_encoder_dropout=float(m.get("input_encoder_dropout", 0.0)),
        bipartite=bool(m.get("bipartite", False)),
        use_input_gate=bool(m.get("use_input_gate", False)),
    ).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()

    feat, tgt = _build_windowed_samples(
        data_info["test_data"], data_info["confidential_indices"], win)
    X = torch.as_tensor(feat, dtype=torch.float32, device=device)   # (B, N, win)
    y = torch.as_tensor(tgt, dtype=torch.float32, device=device)    # (B, C)
    return model, X, y, data_info


def per_conf_r2(pred: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    p, t = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["mask", "single", "attn", "ig", "gate", "shapley", "lrp"])
    ap.add_argument("--config", default=str(SCRIPTS.parent / "configs/multi_graphcombo.yaml"))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pt = Path(args.model) if args.model else Path(sorted(
        glob.glob(str(ROOT / "outputs/tuning/multi_graphcombo/*/training/model.pt")))[-1])
    model, X, y, data_info = load_model_and_test(Path(args.config), model_pt, device)
    gen_idx = data_info["general_indices"]
    gen_names = data_info["general"]

    with torch.no_grad():
        base_pred = model(X)
    a_c = np.clip(per_conf_r2(base_pred, y), 0.0, None)              # (C,) 推断准确率权重
    a_t = torch.as_tensor(a_c, dtype=torch.float32, device=device)
    scores = np.zeros(len(gen_idx), dtype=np.float64)

    if args.method == "mask":
        base_mse = ((base_pred - y) ** 2).mean(0)                   # (C,)
        for k, j in enumerate(gen_idx):
            Xm = X.clone(); Xm[:, j, :] = 0.0
            with torch.no_grad():
                mse = ((model(Xm) - y) ** 2).mean(0)
            scores[k] = float((a_t * (mse - base_mse).clamp(min=0)).sum())

    elif args.method == "single":
        for k, j in enumerate(gen_idx):
            Xs = torch.zeros_like(X); Xs[:, j, :] = X[:, j, :]     # 只留 general j
            with torch.no_grad():
                r2 = torch.as_tensor(np.clip(per_conf_r2(model(Xs), y), 0, None),
                                     dtype=torch.float32, device=device)
            scores[k] = float((a_t * r2).sum())

    elif args.method == "attn":
        with torch.no_grad():
            _ = model(X)
        # layer_attentions: list of (B,N,N)；取每层 general→confidential 注意力均值
        ci = data_info["confidential_indices"]
        att = torch.stack([la.mean(0) for la in model.layer_attentions], 0)  # (L,N,N)
        att = att.mean(0)                                          # (N,N) 跨层平均
        for k, j in enumerate(gen_idx):
            # confidential c 从 general j 的注意力，按 a_c 加权
            col = att[ci, j].detach().cpu().numpy()                # (C,)
            scores[k] = float((a_c * col).sum())

    elif args.method == "ig":
        steps = 32
        baseline = torch.zeros_like(X)
        total = torch.zeros_like(X)
        for s in range(1, steps + 1):
            Xi = (baseline + (s / steps) * (X - baseline)).clone().requires_grad_(True)
            out = (model(Xi) * a_t).sum()                          # a_c 加权输出
            grad, = torch.autograd.grad(out, Xi)
            total += grad.detach()
        ig = ((X - baseline) * (total / steps)).abs().mean(0).squeeze(-1)  # (N,)
        ig = ig.detach().cpu().numpy()
        for k, j in enumerate(gen_idx):
            scores[k] = float(ig[j])

    elif args.method == "gate":
        # 读取「与模型联合训练 + L1」得到的输入门控 g（需 --config/--model 指向 gate 训练）
        if not getattr(model, "use_input_gate", False):
            raise SystemExit("gate 方法需 use_input_gate=True 的模型，请用 gate 训练的 config/model")
        scores = model.input_gate.detach().abs().cpu().numpy()
        print(f"  [gate] g 统计: min={scores.min():.3f} max={scores.max():.3f} "
              f"mean={scores.mean():.3f} std={scores.std():.3f}")

    elif args.method == "lrp":
        # 节点级 LRP-0：冻结注意力(固定路由)，relevance = 输入 × 梯度，沿推断图回流。
        model.freeze_attention = True
        Xi = X.clone().requires_grad_(True)
        out = (model(Xi) * a_t).sum()                  # a_c 加权输出
        grad, = torch.autograd.grad(out, Xi)
        rel = (Xi * grad).abs().mean(0).squeeze(-1)    # (N,) 每节点相关性
        model.freeze_attention = False
        rel = rel.detach().cpu().numpy()
        for k, j in enumerate(gen_idx):
            scores[k] = float(rel[j])

    elif args.method == "shapley":
        # 蒙特卡洛 Shapley:value(S)=只用 S 的 general 推断的 a_c 加权 R²
        rng = np.random.default_rng(42)
        n_perm = 60
        gi = np.array(gen_idx)
        n_g = len(gi)
        phi = np.zeros(n_g)

        def value(active: np.ndarray) -> float:
            Xs = torch.zeros_like(X)
            on = gi[active]
            if len(on) > 0:
                Xs[:, on, :] = X[:, on, :]
            with torch.no_grad():
                p = model(Xs)
            r2 = np.clip(per_conf_r2(p, y), 0.0, None)
            return float((a_c * r2).sum())

        v_empty = value(np.zeros(n_g, dtype=bool))
        for it in range(n_perm):
            perm = rng.permutation(n_g)
            active = np.zeros(n_g, dtype=bool)
            prev = v_empty
            for idx in perm:
                active[idx] = True
                v = value(active)
                phi[idx] += v - prev
                prev = v
        scores = phi / n_perm

    out_dir = ROOT / "outputs/sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"general_field": gen_names, "raw_score": scores})
    df.to_csv(out_dir / f"{args.method}_raw.csv", index=False)
    print(f"[{args.method}] 已输出 {out_dir / f'{args.method}_raw.csv'}  "
          f"(top3: {df.sort_values('raw_score', ascending=False)['general_field'].head(3).tolist()})")


if __name__ == "__main__":
    main()
