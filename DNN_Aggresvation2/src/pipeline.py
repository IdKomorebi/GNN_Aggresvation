from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .correlation_metrics import compute_metric_tensor
from .models import CorrelationGateDNN, FixedGraphRiskGNN, SingleFeatureDNN
from .node_features import NODE_STAT_NAMES, compute_node_stats
from .utils import (
    chronological_split,
    ensure_dir,
    load_yaml,
    make_run_dir,
    r2_score_np,
    read_numeric_table,
    save_json,
    save_yaml,
    set_seed,
    standardize_train_test,
    torch_device,
)


@dataclass
class ProjectData:
    df: pd.DataFrame
    columns: list[str]
    confidential: list[str]
    general: list[str]
    column_to_idx: dict[str, int]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def prepare_data(cfg: dict[str, Any]) -> ProjectData:
    fields_cfg = cfg.get("fields", {})
    drop_columns = _as_list(fields_cfg.get("drop_columns"))
    df = read_numeric_table(
        cfg["dataset"]["processed_csv"],
        drop_columns=drop_columns,
        drop_constant_columns=bool(fields_cfg.get("drop_constant_columns", False)),
    )
    confidential_raw = _as_list(cfg.get("fields", {}).get("confidential"))
    confidential = [c for c in confidential_raw if c in df.columns]
    missing = sorted(set(confidential_raw) - set(confidential))
    if missing:
        print(f"Warning: confidential fields not found after preprocessing: {missing}")
    general = [c for c in df.columns if c not in set(confidential)]
    columns = confidential + [c for c in general if c not in set(confidential)]
    df = df[columns].copy()
    return ProjectData(
        df=df,
        columns=columns,
        confidential=confidential,
        general=[c for c in columns if c not in set(confidential)],
        column_to_idx={c: i for i, c in enumerate(columns)},
    )


def _standardized_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_x = df[feature_cols].to_numpy(dtype=float)
    raw_y = df[target_col].to_numpy(dtype=float).reshape(-1, 1)
    valid_y = np.isfinite(raw_y.reshape(-1))
    raw_x = raw_x[valid_y]
    raw_y = raw_y[valid_y]
    train_idx, test_idx = chronological_split(len(raw_y), train_ratio)
    x_train, x_test, _, _ = standardize_train_test(raw_x[train_idx], raw_x[test_idx])
    y_train, y_test, _, _ = standardize_train_test(raw_y[train_idx], raw_y[test_idx])
    return x_train.astype(np.float32), x_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32)


def _eval_torch_regression(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return r2_score_np(y, pred)


def train_correlation_gate_for_target(
    target: str,
    data: ProjectData,
    metric_tensor: np.ndarray,
    metrics: list[str],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    pre_cfg = cfg["pretrain"]
    x_train, x_test, y_train, y_test = _standardized_xy(
        data.df,
        data.general,
        target,
        train_ratio=float(pre_cfg.get("train_ratio", 0.8)),
    )
    target_idx = data.column_to_idx[target]
    general_indices = [data.column_to_idx[c] for c in data.general]
    corr_vectors = metric_tensor[general_indices, target_idx, :].astype(np.float32)

    model = CorrelationGateDNN(
        in_dim=len(data.general),
        hidden_dims=pre_cfg.get("hidden_dims", [64, 32, 16]),
        correlation_vectors=corr_vectors,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(pre_cfg.get("lr", 1e-3)))
    batch_size = int(pre_cfg.get("batch_size", 64))
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    lambda_gate = float(pre_cfg.get("lambda_gate", 0.002))
    log_rows = []
    best_r2 = -1e9
    best_alpha: np.ndarray | None = None
    best_gates: np.ndarray | None = None

    for epoch in range(1, int(pre_cfg.get("epochs", 160)) + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mse = nn.functional.mse_loss(pred, yb)
            gate_penalty = torch.mean(torch.abs(model.get_gates()))
            loss = mse + lambda_gate * gate_penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.size(0)

        test_r2 = _eval_torch_regression(model, x_test, y_test, device)
        with torch.no_grad():
            alpha_np = model.metric_weights().detach().cpu().numpy()
            gates_np = model.get_gates().detach().cpu().numpy()
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_alpha = alpha_np.copy()
            best_gates = gates_np.copy()
        log_rows.append({"epoch": epoch, "loss": total / len(loader.dataset), "test_r2": test_r2})

    assert best_alpha is not None and best_gates is not None
    target_dir = ensure_dir(output_dir / target)
    pd.DataFrame(log_rows).to_csv(target_dir / "log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "metric": metrics,
            "alpha": best_alpha,
        }
    ).to_csv(target_dir / "metric_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "feature": data.general,
            "gate": best_gates,
            **{metric: corr_vectors[:, i] for i, metric in enumerate(metrics)},
        }
    ).sort_values("gate", ascending=False).to_csv(target_dir / "feature_gates.csv", index=False, encoding="utf-8-sig")
    return {
        "target": target,
        "test_r2": float(best_r2),
        "alpha": {metric: float(best_alpha[i]) for i, metric in enumerate(metrics)},
    }


def pretrain_global_metric_weights(
    data: ProjectData,
    metric_tensor: np.ndarray,
    metrics: list[str],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> np.ndarray:
    rows = []
    alphas = []
    weights = []
    for target in data.confidential:
        print(f"Pretraining correlation weights for target: {target}")
        result = train_correlation_gate_for_target(target, data, metric_tensor, metrics, cfg, device, output_dir)
        rows.append(result)
        alpha = np.asarray([result["alpha"][m] for m in metrics], dtype=float)
        alphas.append(alpha)
        weights.append(max(float(result["test_r2"]), 0.0))

    alpha_arr = np.vstack(alphas)
    rel_weights = np.asarray(weights, dtype=float)
    if bool(cfg["pretrain"].get("r2_weighted_average", True)) and rel_weights.sum() > 1e-12:
        global_alpha = (alpha_arr * rel_weights[:, None]).sum(axis=0) / rel_weights.sum()
    else:
        global_alpha = alpha_arr.mean(axis=0)
    global_alpha = np.clip(global_alpha, 0.0, None)
    global_alpha = global_alpha / max(float(global_alpha.sum()), 1e-12)

    payload = {
        "metrics": metrics,
        "global_alpha": {metric: float(global_alpha[i]) for i, metric in enumerate(metrics)},
        "target_weights": rows,
        "r2_weighted_average": bool(cfg["pretrain"].get("r2_weighted_average", True)),
    }
    save_json(output_dir / "correlation_weights.json", payload)
    pd.DataFrame(
        {
            "metric": metrics,
            "global_alpha": global_alpha,
        }
    ).to_csv(output_dir / "global_metric_weights.csv", index=False, encoding="utf-8-sig")
    return global_alpha.astype(np.float32)


def build_fixed_graph(
    data: ProjectData,
    metric_tensor: np.ndarray,
    metrics: list[str],
    global_alpha: np.ndarray,
    cfg: dict[str, Any],
    output_dir: Path,
) -> np.ndarray:
    graph_cfg = cfg["graph"]
    score = np.tensordot(metric_tensor, global_alpha, axes=([2], [0]))
    np.fill_diagonal(score, 0.0)
    n = len(data.columns)
    mask = score >= float(graph_cfg.get("threshold", 0.08))
    top_k = int(graph_cfg.get("top_k", 10))
    for i in range(n):
        order = np.argsort(score[i])[::-1]
        order = [j for j in order if j != i][:top_k]
        mask[i, order] = True
    if bool(graph_cfg.get("symmetrize", True)):
        mask = mask | mask.T
    adj = score * mask
    if bool(graph_cfg.get("add_self_loop", False)):
        np.fill_diagonal(adj, 1.0)
    row_sum = adj.sum(axis=1, keepdims=True)
    normalized = np.divide(adj, row_sum + 1e-12, out=np.zeros_like(adj), where=row_sum > 0)

    edge_rows = []
    for i, src in enumerate(data.columns):
        for j, dst in enumerate(data.columns):
            if i == j or adj[i, j] <= 0:
                continue
            row = {
                "source": src,
                "target": dst,
                "raw_weight": float(adj[i, j]),
                "normalized_weight": float(normalized[i, j]),
            }
            for m_idx, metric in enumerate(metrics):
                row[metric] = float(metric_tensor[i, j, m_idx])
            edge_rows.append(row)
    pd.DataFrame(edge_rows).sort_values("raw_weight", ascending=False).to_csv(
        output_dir / "edge_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(output_dir / "adjacency.npy", normalized.astype(np.float32))
    return normalized.astype(np.float32)


def select_general_anchors(
    data: ProjectData,
    metric_tensor: np.ndarray,
    global_alpha: np.ndarray,
    cfg: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    anchor_cfg = cfg["anchors"]
    high_count = int(anchor_cfg.get("high_anchor_count", anchor_cfg.get("general_anchor_count", 10)))
    low_count = int(anchor_cfg.get("low_anchor_count", 0))
    score = np.tensordot(metric_tensor, global_alpha, axes=([2], [0]))
    confidential_indices = [data.column_to_idx[c] for c in data.confidential]
    rows = []
    for field in data.general:
        idx = data.column_to_idx[field]
        conf_scores = score[idx, confidential_indices]
        rows.append(
            {
                "field": field,
                "max_confidential_edge": float(np.max(conf_scores)) if len(conf_scores) else 0.0,
                "mean_top3_confidential_edge": float(np.mean(np.sort(conf_scores)[-3:])) if len(conf_scores) else 0.0,
            }
        )
    anchor_df = pd.DataFrame(rows)
    anchor_df["anchor_score"] = 0.7 * anchor_df["max_confidential_edge"] + 0.3 * anchor_df["mean_top3_confidential_edge"]
    anchor_df = anchor_df.sort_values("anchor_score", ascending=False).reset_index(drop=True)
    anchor_df.to_csv(output_dir / "general_anchor_candidates.csv", index=False, encoding="utf-8-sig")

    high = anchor_df.head(max(high_count, 0)).copy()
    high["anchor_role"] = "high"
    used = set(high["field"].astype(str))
    low_pool = anchor_df[~anchor_df["field"].astype(str).isin(used)].sort_values("anchor_score", ascending=True)
    low = low_pool.head(max(low_count, 0)).copy()
    low["anchor_role"] = "low"

    selected = pd.concat([high, low], ignore_index=True)
    selected["anchor_rank"] = np.arange(1, len(selected) + 1)
    selected.to_csv(output_dir / "selected_general_anchors.csv", index=False, encoding="utf-8-sig")
    return selected


def train_single_feature_probe(
    df: pd.DataFrame,
    feature: str,
    target: str,
    cfg: dict[str, Any],
    device: torch.device,
) -> float:
    anchor_cfg = cfg["anchors"]
    x_train, x_test, y_train, y_test = _standardized_xy(
        df,
        [feature],
        target,
        train_ratio=float(cfg["pretrain"].get("train_ratio", 0.8)),
    )
    model = SingleFeatureDNN(anchor_cfg.get("probe_hidden_dims", [32, 16])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(anchor_cfg.get("probe_lr", 1e-3)))
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=int(anchor_cfg.get("probe_batch_size", 64)),
        shuffle=True,
    )
    for _ in range(int(anchor_cfg.get("probe_epochs", 80))):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return max(0.0, _eval_torch_regression(model, x_test, y_test, device))


def compute_anchor_labels(
    data: ProjectData,
    anchors: list[str] | pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    probe_cache_dirs: list[str | Path] | None = None,
) -> pd.DataFrame:
    if isinstance(anchors, pd.DataFrame):
        anchor_df = anchors.copy()
    else:
        anchor_df = pd.DataFrame({"field": anchors, "anchor_role": "high"})
    if "anchor_role" not in anchor_df.columns:
        anchor_df["anchor_role"] = "high"
    if "anchor_score" not in anchor_df.columns:
        anchor_df["anchor_score"] = np.nan

    rows = []
    detail_rows = []
    probe_cache: dict[str, dict[str, float]] = {}
    for cache_dir in probe_cache_dirs or []:
        detail_path = Path(cache_dir) / "general_anchor_target_r2.csv"
        if not detail_path.exists():
            continue
        cache_df = pd.read_csv(detail_path)
        if not {"feature", "target", "r2"}.issubset(cache_df.columns):
            continue
        for _, cache_row in cache_df.iterrows():
            feature_key = str(cache_row["feature"])
            target_key = str(cache_row["target"])
            probe_cache.setdefault(feature_key, {})[target_key] = float(cache_row["r2"])

    max_w = float(cfg["anchors"].get("q_max_weight", 0.7))
    top3_w = float(cfg["anchors"].get("q_top3_weight", 0.3))
    low_max_graph = float(cfg["anchors"].get("low_anchor_max_graph_score", 0.25))
    low_max_probe_q = float(cfg["anchors"].get("low_anchor_max_probe_q", 0.25))
    for _, anchor_row in anchor_df.iterrows():
        feature = str(anchor_row["field"])
        anchor_role = str(anchor_row.get("anchor_role", "high"))
        anchor_score = float(anchor_row["anchor_score"]) if pd.notna(anchor_row.get("anchor_score", np.nan)) else np.nan
        cached_target_scores = probe_cache.get(feature, {})
        can_reuse_cache = all(target in cached_target_scores for target in data.confidential)
        if can_reuse_cache:
            print(f"Reusing cached q label for general anchor: {feature}")
        else:
            print(f"Measuring q label for general anchor: {feature}")

        r2_values = []
        for target in data.confidential:
            if can_reuse_cache:
                r2 = cached_target_scores[target]
                from_cache = True
            else:
                r2 = train_single_feature_probe(data.df, feature, target, cfg, device)
                from_cache = False
            r2_values.append(r2)
            detail_rows.append({
                "feature": feature,
                "anchor_role": anchor_role,
                "target": target,
                "r2": float(r2),
                "from_cache": from_cache,
            })
        r2_sorted = np.sort(np.asarray(r2_values, dtype=float))
        raw_q_value = max_w * float(r2_sorted[-1]) + top3_w * float(np.mean(r2_sorted[-3:]))
        raw_q_value = float(np.clip(raw_q_value, 0.0, 1.0))
        if anchor_role == "low":
            confirmed_low = (
                (not np.isfinite(anchor_score) or anchor_score <= low_max_graph)
                and raw_q_value <= low_max_probe_q
            )
            use_in_gnn = bool(confirmed_low)
            label_source = "low_probe_confirmed" if confirmed_low else "low_probe_rejected"
        else:
            use_in_gnn = True
            label_source = "high_probe_q"
        rows.append(
            {
                "field": feature,
                "anchor_role": anchor_role,
                "anchor_score": anchor_score,
                "q_label": raw_q_value,
                "raw_probe_q_label": raw_q_value,
                "max_r2": float(r2_sorted[-1]),
                "top3_mean_r2": float(np.mean(r2_sorted[-3:])),
                "use_in_gnn": use_in_gnn,
                "label_source": label_source,
            }
        )
    labels = pd.DataFrame(rows).sort_values("q_label", ascending=False).reset_index(drop=True)
    labels.to_csv(output_dir / "general_anchor_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(output_dir / "general_anchor_target_r2.csv", index=False, encoding="utf-8-sig")
    return labels


def train_fixed_gnn(
    data: ProjectData,
    adjacency: np.ndarray,
    anchor_labels: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    gnn_cfg = cfg["gnn"]
    stats = compute_node_stats(data.df, data.columns)
    node_feature_cols = list(NODE_STAT_NAMES)
    s_init = np.asarray([1.0 if c in set(data.confidential) else 0.0 for c in data.columns], dtype=np.float32)
    feature_matrix = stats[node_feature_cols].to_numpy(dtype=np.float32)
    if bool(gnn_cfg.get("include_initial_sensitivity", True)):
        feature_matrix = np.column_stack([s_init, feature_matrix]).astype(np.float32)
        node_feature_cols = ["s_init", *node_feature_cols]

    labels = np.full(len(data.columns), np.nan, dtype=np.float32)
    weights = np.zeros(len(data.columns), dtype=np.float32)
    label_mode = str(gnn_cfg.get("label_mode", "sensitivity"))
    if label_mode == "sensitivity":
        for field in data.confidential:
            idx = data.column_to_idx[field]
            labels[idx] = 1.0
            weights[idx] = float(gnn_cfg.get("confidential_label_weight", 1.0))
    for _, row in anchor_labels.iterrows():
        if "use_in_gnn" in anchor_labels.columns and not bool(row.get("use_in_gnn", True)):
            continue
        field = str(row["field"])
        idx = data.column_to_idx[field]
        labels[idx] = float(row["q_label"])
        weights[idx] = float(gnn_cfg.get("anchor_label_weight", 1.0))

    mask = np.isfinite(labels) & (weights > 0)
    if mask.sum() == 0:
        raise ValueError("No GNN supervision labels were produced.")

    model = FixedGraphRiskGNN(
        in_dim=feature_matrix.shape[1],
        adjacency=adjacency,
        hidden_dim=int(gnn_cfg.get("hidden_dim", 48)),
        num_layers=int(gnn_cfg.get("num_layers", 2)),
        dropout=float(gnn_cfg.get("dropout", 0.05)),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(gnn_cfg.get("lr", 0.005)),
        weight_decay=float(gnn_cfg.get("weight_decay", 1e-4)),
    )
    x_t = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    m_t = torch.as_tensor(mask, dtype=torch.bool, device=device)

    log_rows = []
    for epoch in range(1, int(gnn_cfg.get("epochs", 1200)) + 1):
        model.train()
        pred = model(x_t)
        sq = (pred[m_t] - y_t[m_t]) ** 2
        loss = torch.mean(sq * w_t[m_t])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == int(gnn_cfg.get("epochs", 1200)):
            log_rows.append({"epoch": epoch, "loss": float(loss.detach().cpu())})

    model.eval()
    with torch.no_grad():
        scores = model(x_t).detach().cpu().numpy()
    pd.DataFrame(log_rows).to_csv(output_dir / "log.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state": model.state_dict(),
            "node_feature_cols": node_feature_cols,
            "columns": data.columns,
            "label_mode": label_mode,
        },
        output_dir / "model.pt",
    )

    out = pd.DataFrame(
        {
            "field": data.columns,
            "field_type": ["confidential" if c in set(data.confidential) else "general" for c in data.columns],
            "s_init": s_init,
            "label": labels,
            "is_labeled": mask,
            "score": scores,
        }
    ).sort_values("score", ascending=False)
    if label_mode == "q_inference":
        out["sensitivity_score_posthoc"] = out["s_init"] + (1.0 - out["s_init"]) * out["score"]
    out.to_csv(output_dir / "node_scores.csv", index=False, encoding="utf-8-sig")
    save_json(
        output_dir / "metrics.json",
        {
            "label_mode": label_mode,
            "labeled_nodes": int(mask.sum()),
            "confidential_nodes": len(data.confidential),
            "general_anchor_nodes": int(
                anchor_labels["use_in_gnn"].sum() if "use_in_gnn" in anchor_labels.columns else len(anchor_labels)
            ),
            "selected_general_anchor_nodes": len(anchor_labels),
            "final_loss": float(log_rows[-1]["loss"]) if log_rows else None,
        },
    )
    return out


def maybe_plot(run_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    gnn_dir = run_dir / "gnn"
    score_path = gnn_dir / "node_scores.csv"
    log_path = gnn_dir / "log.csv"
    if score_path.exists():
        scores = pd.read_csv(score_path)
        plt.figure(figsize=(9, 5))
        for field_type, group in scores.groupby("field_type"):
            plt.hist(group["score"], bins=15, alpha=0.65, label=field_type)
        plt.xlabel("Score")
        plt.ylabel("Node count")
        plt.title("GNN Risk Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(gnn_dir / "score_distribution.png", dpi=180)
        plt.close()
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        plt.figure(figsize=(8, 4))
        plt.plot(log_df["epoch"], log_df["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GNN Training Loss")
        plt.tight_layout()
        plt.savefig(gnn_dir / "loss.png", dpi=180)
        plt.close()


def run_pipeline(config_path: str | Path) -> Path:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("runtime", {}).get("random_state", 42)))
    device = torch_device(str(cfg.get("runtime", {}).get("device", "cpu")))
    run_dir = make_run_dir(cfg["dataset"].get("output_root", "outputs"))
    save_yaml(run_dir / "config_used.yaml", cfg)

    print(f"Run directory: {run_dir}")
    data = prepare_data(cfg)
    save_json(
        run_dir / "field_split.json",
        {
            "columns": data.columns,
            "confidential": data.confidential,
            "general": data.general,
            "general_count": len(data.general),
            "confidential_count": len(data.confidential),
        },
    )

    metrics = list(cfg["relations"].get("metrics", []))
    rel_dir = ensure_dir(run_dir / "relationships")
    print(f"Computing relationship tensor for {len(data.columns)} fields...")
    metric_tensor = compute_metric_tensor(
        data.df,
        data.columns,
        metrics=metrics,
        sample_size=int(cfg["relations"].get("sample_size", 3000)),
        expensive_sample_size=int(cfg["relations"].get("expensive_sample_size", 1200)),
        random_state=int(cfg.get("runtime", {}).get("random_state", 42)),
    )
    np.save(rel_dir / "metric_tensor.npy", metric_tensor)
    save_json(rel_dir / "metrics.json", {"metrics": metrics, "columns": data.columns})

    pretrain_dir = ensure_dir(run_dir / "pretrain")
    global_alpha = pretrain_global_metric_weights(data, metric_tensor, metrics, cfg, device, pretrain_dir)

    graph_dir = ensure_dir(run_dir / "graph")
    adjacency = build_fixed_graph(data, metric_tensor, metrics, global_alpha, cfg, graph_dir)

    labels_dir = ensure_dir(run_dir / "labels")
    anchors = select_general_anchors(data, metric_tensor, global_alpha, cfg, labels_dir)
    save_json(
        labels_dir / "selected_general_anchors.json",
        {
            "high_anchors": anchors.loc[anchors["anchor_role"] == "high", "field"].astype(str).tolist(),
            "low_anchors": anchors.loc[anchors["anchor_role"] == "low", "field"].astype(str).tolist(),
            "anchors": anchors["field"].astype(str).tolist(),
        },
    )
    anchor_labels = compute_anchor_labels(data, anchors, cfg, device, labels_dir)

    gnn_dir = ensure_dir(run_dir / "gnn")
    train_fixed_gnn(data, adjacency, anchor_labels, cfg, gnn_dir, device)
    maybe_plot(run_dir)
    print(f"Finished: {run_dir}")
    return run_dir
