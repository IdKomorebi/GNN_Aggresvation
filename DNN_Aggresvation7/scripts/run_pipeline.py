"""
DNN_Aggresvation7 主流水线脚本。

完整流程：
1. 数据加载与预处理
2. 计算5种相关性指标张量
3. 构建图结构（KNN+阈值边掩码）
4. 训练端到端推断驱动edge-specific GNN/GAT（任务：还原Confidential真实值）
5. 遮蔽重要性法提取General字段敏感度
6. 导出结果，并绘制DNN probe R² vs 当前模型R²三线表
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 动态配置项目根目录
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.correlation import compute_metric_tensor
from src.data_processing import prepare_data
from src.model import InferenceDrivenGNN, build_edge_mask
from src.sensitivity import compute_masking_sensitivity
from src.train import train_model


def _load_config(config_path: str) -> dict:
    """加载YAML配置文件。"""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # 简易YAML解析降级
        config: dict = {}
        current_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip()
                if not clean or clean.startswith("#"):
                    continue
                if clean.endswith(":") and "  " not in line[:2]:
                    current_section = clean[:-1].strip()
                    config[current_section] = {}
                    continue
                if ":" in clean:
                    parts = clean.split(":", 1)
                    k = parts[0].strip().lstrip("- ")
                    v = parts[1].split("#")[0].strip()
                    if (v.startswith('"') and v.endswith('"')) or (
                        v.startswith("'") and v.endswith("'")
                    ):
                        v = v[1:-1]
                    elif v.startswith("[") and v.endswith("]"):
                        v = [
                            item.strip().strip("'\"")
                            for item in v[1:-1].split(",")
                            if item.strip()
                        ]
                    else:
                        try:
                            v = float(v) if "." in v else int(v)
                        except ValueError:
                            if v.lower() == "true":
                                v = True
                            elif v.lower() == "false":
                                v = False
                    if current_section:
                        config[current_section][k] = v
                    else:
                        config[k] = v
        return config


def _resolve_path(path: str) -> str:
    """将相对路径解析为项目根目录的绝对路径。"""
    if os.path.isabs(path):
        return path
    return str((PROJECT_ROOT / path).resolve())


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _load_probe_r2(cfg: dict) -> dict[str, float]:
    """读取子项目4单目标DNN probe结果，作为字段可推断性参照。"""
    eval_cfg = cfg.get("evaluation", {})
    probe_path_cfg = eval_cfg.get(
        "probe_results_csv",
        "../DNN_Aggresvation4/outputs/target_probe_20260521_113054/target_probe_results.csv",
    )
    probe_path = Path(_resolve_path(probe_path_cfg))
    if not probe_path.exists():
        print(f"  [提示] 未找到DNN probe结果，跳过R²对比表: {probe_path}")
        return {}
    df = pd.read_csv(probe_path)
    if "confidential_field" not in df.columns or "probe_r2" not in df.columns:
        print(f"  [提示] DNN probe结果列不完整，跳过R²对比表: {probe_path}")
        return {}
    return dict(zip(df["confidential_field"], df["probe_r2"]))


def _save_r2_comparison_table(
    run_dir: Path,
    data_info: dict,
    train_results: dict,
    probe_r2: dict[str, float],
) -> None:
    """绘制三线表风格PNG：单目标DNN probe R² vs 当前模型测试R²。"""
    if not probe_r2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = []
    used_names = set(train_results["loss_target_names"])
    for field in data_info["confidential"]:
        model_value = train_results["per_target_r2"].get(field)
        probe_value = probe_r2.get(field)
        rows.append(
            {
                "field": field,
                "probe_r2": probe_value,
                "model_r2": model_value,
                "delta": None
                if probe_value is None or model_value is None
                else model_value - probe_value,
                "used": field in used_names,
            }
        )

    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_dir / "r2_probe_vs_model.csv", index=False)

    fig_h = max(5.2, 0.38 * (len(rows) + 3))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")

    title = "Per-Confidential R2: Single-target DNN Probe vs Current Model"
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=14, fontweight="bold")

    headers = ["Confidential field", "DNN probe R2", "Model test R2", "Model - Probe", "In loss"]
    xs = [0.02, 0.54, 0.68, 0.82, 0.94]
    y_top = 0.90
    row_h = 0.82 / (len(rows) + 1)

    ax.hlines(y_top + row_h * 0.45, 0.01, 0.99, colors="black", linewidth=1.4)
    ax.hlines(y_top - row_h * 0.45, 0.01, 0.99, colors="black", linewidth=0.9)
    ax.hlines(y_top - row_h * (len(rows) + 0.55), 0.01, 0.99, colors="black", linewidth=1.4)

    for x, header in zip(xs, headers):
        ha = "left" if header == "Confidential field" else "center"
        ax.text(x, y_top, header, ha=ha, va="center", fontsize=10.5, fontweight="bold")

    for idx, row in enumerate(rows, start=1):
        y = y_top - row_h * idx
        probe_text = "" if row["probe_r2"] is None else f"{row['probe_r2']:.4f}"
        model_text = "" if row["model_r2"] is None else f"{row['model_r2']:.4f}"
        delta_text = "" if row["delta"] is None else f"{row['delta']:+.4f}"
        delta_color = "#166534" if row["delta"] is not None and row["delta"] >= 0 else "#991b1b"
        if row["delta"] is None:
            delta_color = "black"

        ax.text(xs[0], y, row["field"], ha="left", va="center", fontsize=9.2)
        ax.text(xs[1], y, probe_text, ha="center", va="center", fontsize=9.2)
        ax.text(xs[2], y, model_text, ha="center", va="center", fontsize=9.2)
        ax.text(xs[3], y, delta_text, ha="center", va="center", fontsize=9.2, color=delta_color)
        ax.text(xs[4], y, "yes" if row["used"] else "no", ha="center", va="center", fontsize=9.2)

    ax.text(
        0.01,
        0.025,
        "DNN probe uses the DNN_Aggresvation4 single-target MLP diagnostic; current model R2 is measured on this run's test set.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#4b5563",
    )

    fig.savefig(table_dir / "r2_probe_vs_model_booktabs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  R²三线对比表已保存至: {table_dir / 'r2_probe_vs_model_booktabs.png'}")


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        _save_json(path.with_suffix(".json"), data)


def _load_or_compute_metric_tensor(
    cfg: dict,
    data_info: dict,
    metrics: list[str],
    seed: int,
    run_dir: Path,
) -> np.ndarray:
    """读取或计算相关性张量，并在当前run目录保存副本。"""
    corr_cfg = cfg.get("correlation", {})
    cache_dir_cfg = corr_cfg.get("cache_dir")
    cache_dir = Path(_resolve_path(cache_dir_cfg)) if cache_dir_cfg else None
    cache_tensor = cache_dir / "metric_tensor.npy" if cache_dir else None
    cache_meta = cache_dir / "metrics.json" if cache_dir else None

    rel_dir = run_dir / "relationships"
    rel_dir.mkdir(parents=True, exist_ok=True)

    if cache_tensor and cache_tensor.exists() and cache_meta and cache_meta.exists():
        with open(cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("metrics") == metrics and meta.get("columns") == data_info["all_columns"]:
            print(f"  复用相关性缓存: {cache_tensor}")
            metric_tensor = np.load(cache_tensor)
            np.save(rel_dir / "metric_tensor.npy", metric_tensor)
            _save_json(rel_dir / "metrics.json", meta)
            return metric_tensor
        print("  相关性缓存字段或指标不匹配，将重新计算。")

    metric_tensor = compute_metric_tensor(
        data_info["raw_df"],
        data_info["all_columns"],
        metrics=metrics,
        sample_size=int(corr_cfg.get("sample_size", 3000)),
        expensive_sample_size=int(corr_cfg.get("expensive_sample_size", 1200)),
        seed=seed,
    )

    meta = {"metrics": metrics, "columns": data_info["all_columns"]}
    np.save(rel_dir / "metric_tensor.npy", metric_tensor)
    _save_json(rel_dir / "metrics.json", meta)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_tensor, metric_tensor)
        _save_json(cache_meta, meta)
        print(f"  相关性缓存已更新: {cache_dir}")

    return metric_tensor


def _save_attention_diagnostics(
    model: InferenceDrivenGNN,
    train_results: dict,
    data_info: dict,
    run_dir: Path,
    device: torch.device,
) -> None:
    """保存GAT最后一层注意力诊断。"""
    if getattr(model, "architecture", "gcn") not in {"gat", "edge_gat", "edge_conf_gat"}:
        return

    sample_count = min(512, len(train_results["test_features"]))
    sample_x = torch.as_tensor(
        train_results["test_features"][:sample_count],
        dtype=torch.float32,
        device=device,
    )
    attention = model.get_mean_attention(sample_x)
    if attention is None:
        return

    diag_dir = run_dir / "attention"
    diag_dir.mkdir(parents=True, exist_ok=True)
    np.save(diag_dir / "mean_attention_last_layer.npy", attention)

    columns = data_info["all_columns"]
    conf_indices = data_info["confidential_indices"]
    rows = []
    entropy_rows = []
    for conf_pos, node_idx in enumerate(conf_indices):
        target_name = data_info["confidential"][conf_pos]
        weights = attention[node_idx].copy()
        valid = weights > 0
        denom = float(np.log(max(int(valid.sum()), 2)))
        entropy = float(-np.sum(weights[valid] * np.log(weights[valid] + 1e-12)))
        entropy_norm = entropy / denom if denom > 0 else 0.0
        entropy_rows.append(
            {
                "target_field": target_name,
                "degree": int(valid.sum()),
                "attention_entropy": entropy,
                "attention_entropy_normalized": entropy_norm,
            }
        )

        order = np.argsort(weights)[::-1]
        rank = 0
        for source_idx in order:
            if weights[source_idx] <= 0:
                continue
            rank += 1
            rows.append(
                {
                    "target_field": target_name,
                    "rank": rank,
                    "source_field": columns[source_idx],
                    "source_type": "confidential"
                    if source_idx in set(conf_indices)
                    else "general",
                    "attention": float(weights[source_idx]),
                }
            )
            if rank >= 12:
                break

    pd.DataFrame(rows).to_csv(diag_dir / "confidential_attention_top_edges.csv", index=False)
    pd.DataFrame(entropy_rows).to_csv(diag_dir / "confidential_attention_entropy.csv", index=False)
    print(f"  GAT注意力诊断已保存至: {diag_dir}")


def _save_edge_alpha_diagnostics(
    model: InferenceDrivenGNN,
    metric_tensor: np.ndarray,
    edge_mask: np.ndarray,
    data_info: dict,
    metrics: list[str],
    run_dir: Path,
) -> None:
    """保存每条边自己的相关系数融合权重。"""
    if not getattr(model, "use_edge_metric_weights", False):
        return

    diag_dir = run_dir / "edge_alpha"
    diag_dir.mkdir(parents=True, exist_ok=True)

    field_names = data_info["all_columns"]
    edge_alpha = model.get_edge_alpha().detach().cpu().numpy()
    edge_weight = np.sum(metric_tensor * edge_alpha, axis=-1) * edge_mask

    rows = []
    n_nodes = edge_mask.shape[0]
    eye = np.eye(n_nodes, dtype=bool)
    valid_edges = edge_mask.astype(bool) & ~eye

    for target_idx in range(n_nodes):
        for source_idx in range(n_nodes):
            if not valid_edges[target_idx, source_idx]:
                continue
            alpha_values = edge_alpha[target_idx, source_idx]
            dominant_idx = int(np.argmax(alpha_values))
            row = {
                "target_idx": target_idx,
                "target_field": field_names[target_idx],
                "source_idx": source_idx,
                "source_field": field_names[source_idx],
                "edge_weight": float(edge_weight[target_idx, source_idx]),
                "dominant_metric": metrics[dominant_idx],
                "dominant_alpha": float(alpha_values[dominant_idx]),
            }
            for metric_idx, metric_name in enumerate(metrics):
                row[f"alpha_{metric_name}"] = float(alpha_values[metric_idx])
                row[f"corr_{metric_name}"] = float(metric_tensor[target_idx, source_idx, metric_idx])
            rows.append(row)

    pd.DataFrame(rows).sort_values(
        ["target_field", "edge_weight"], ascending=[True, False]
    ).to_csv(diag_dir / "edge_alpha_by_edge.csv", index=False)

    summary_rows = []
    valid_alpha = edge_alpha[valid_edges]
    for metric_idx, metric_name in enumerate(metrics):
        values = valid_alpha[:, metric_idx]
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(diag_dir / "edge_alpha_summary.csv", index=False)

    np.save(diag_dir / "edge_alpha_tensor.npy", edge_alpha)
    np.save(diag_dir / "edge_weight_matrix.npy", edge_weight)
    for metric_idx, metric_name in enumerate(metrics):
        pd.DataFrame(
            edge_alpha[:, :, metric_idx],
            index=field_names,
            columns=field_names,
        ).to_csv(diag_dir / f"edge_alpha_matrix_{metric_name}.csv")

    print(f"  边级alpha诊断已保存至: {diag_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNN_Aggresvation7: edge-specific alpha GNN/GAT 推断驱动敏感度评估流水线"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="YAML配置文件路径",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    config_path = os.path.abspath(args.config)
    print(f"加载配置文件: {config_path}")
    cfg = _load_config(config_path)
    experiment_name = str(cfg.get("experiment", {}).get("name", Path(config_path).stem))

    # 解析路径
    cfg["dataset"]["csv_path"] = _resolve_path(cfg["dataset"]["csv_path"])
    output_root = Path(_resolve_path(cfg["dataset"].get("output_dir", "outputs")))

    # 创建运行目录
    safe_experiment = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in experiment_name
    )
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if safe_experiment:
        run_id = f"{run_id}_{safe_experiment}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_yaml(run_dir / "config_used.yaml", cfg)

    # 设置随机种子
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = str(cfg.get("runtime", {}).get("device", "cpu"))
    device = torch.device(device_str)

    print("\n" + "=" * 60)
    print("  DNN_Aggresvation7: edge-specific alpha GNN/GAT 推断驱动敏感度评估")
    print("=" * 60)
    print(f"  实验名称: {experiment_name}")
    print(f"  运行目录: {run_dir}")
    print(f"  随机种子: {seed}")
    print(f"  计算设备: {device}")

    # ================================================================
    # 步骤1: 数据准备
    # ================================================================
    data_info = prepare_data(cfg)

    # 保存字段信息
    _save_json(
        run_dir / "field_split.json",
        {
            "general": data_info["general"],
            "confidential": data_info["confidential"],
            "n_general": data_info["n_general"],
            "n_confidential": data_info["n_confidential"],
            "n_nodes": data_info["n_nodes"],
        },
    )

    # ================================================================
    # 步骤2: 计算相关性张量
    # ================================================================
    corr_cfg = cfg.get("correlation", {})
    metrics = corr_cfg.get("metrics", ["pearson", "spearman", "kendall", "nmi", "distance_corr"])

    print("\n" + "=" * 60)
    print(f"  [相关性计算] 计算 {len(data_info['all_columns'])} 个字段间的 {len(metrics)} 种相关性")
    print("=" * 60)

    metric_tensor = _load_or_compute_metric_tensor(
        cfg=cfg,
        data_info=data_info,
        metrics=metrics,
        seed=seed,
        run_dir=run_dir,
    )
    print(f"  相关性张量形状: {metric_tensor.shape}")

    # ================================================================
    # 步骤3: 构建图结构
    # ================================================================
    graph_cfg = cfg.get("graph", {})
    top_k = int(graph_cfg.get("top_k", 10))
    threshold = float(graph_cfg.get("threshold", 0.08))
    symmetrize = bool(graph_cfg.get("symmetrize", True))

    print(f"\n  [图构建] top_k={top_k}, threshold={threshold}, symmetrize={symmetrize}")
    edge_mask = build_edge_mask(metric_tensor, top_k=top_k, threshold=threshold, symmetrize=symmetrize)
    n_edges = int(edge_mask.sum()) - int(np.trace(edge_mask))
    print(f"  边掩码: {n_edges} 条有效边 (密度: {n_edges / (data_info['n_nodes'] ** 2):.4f})")

    graph_dir = run_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    np.save(graph_dir / "edge_mask.npy", edge_mask)

    # ================================================================
    # 步骤4: 训练GNN
    # ================================================================
    model_cfg = cfg.get("model", {})
    window_size = int(model_cfg.get("window_size", 1))
    architecture = str(model_cfg.get("architecture", "gcn"))
    model = InferenceDrivenGNN(
        metric_tensor=metric_tensor,
        edge_mask=edge_mask,
        n_nodes=data_info["n_nodes"],
        n_general=data_info["n_general"],
        confidential_indices=data_info["confidential_indices"],
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        input_dim=window_size,
        architecture=architecture,
        attention_dropout=float(model_cfg.get("attention_dropout", 0.05)),
        edge_alpha_temperature=float(model_cfg.get("edge_alpha_temperature", 1.0)),
    ).to(device)
    print(f"  模型架构: {architecture}")
    if architecture.startswith("edge_"):
        print(f"  边级alpha温度: {float(model_cfg.get('edge_alpha_temperature', 1.0))}")

    train_results = train_model(
        model=model,
        train_data=data_info["train_data"],
        test_data=data_info["test_data"],
        data_info=data_info,
        cfg=cfg,
        device=device,
    )

    # 保存训练结果
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "epoch": range(1, len(train_results["history"]["train_loss"]) + 1),
            "train_loss": train_results["history"]["train_loss"],
            "test_loss": train_results["history"]["test_loss"],
            "test_mse_all": train_results["history"]["test_mse_all"],
            "test_mse_excluded": train_results["history"]["test_mse_excluded"],
        }
    ).to_csv(train_dir / "log.csv", index=False)

    torch.save(model.state_dict(), train_dir / "model.pt")

    # α权重
    alpha_df = pd.DataFrame(
        {"metric": metrics, "alpha": train_results["final_alpha"]}
    )
    alpha_df.to_csv(train_dir / "alpha_weights.csv", index=False)
    _save_edge_alpha_diagnostics(model, metric_tensor, edge_mask, data_info, metrics, run_dir)

    # 逐目标R²
    pd.DataFrame(
        [
            {
                "confidential_field": k,
                "test_r2": v,
                "test_mse": train_results["per_target_mse"][k],
                "used_in_loss": k in set(train_results["loss_target_names"]),
            }
            for k, v in train_results["per_target_r2"].items()
        ]
    ).sort_values("test_r2", ascending=False).to_csv(
        train_dir / "per_target_r2.csv", index=False
    )

    probe_r2 = _load_probe_r2(cfg)
    _save_r2_comparison_table(run_dir, data_info, train_results, probe_r2)

    _save_attention_diagnostics(model, train_results, data_info, run_dir, device)

    # ================================================================
    # 步骤5: 遮蔽重要性法计算敏感度
    # ================================================================
    print("\n" + "=" * 60)
    print("  [敏感度评估] 遮蔽重要性法")
    print("=" * 60)

    sensitivity_df, baseline_loss = compute_masking_sensitivity(
        model=model,
        test_data=data_info["test_data"],
        data_info=data_info,
        device=device,
        window_size=window_size,
        target_indices=train_results["loss_target_indices"],
    )

    # ================================================================
    # 步骤6: 导出结果
    # ================================================================
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_df.to_csv(results_dir / "sensitivity_scores.csv", index=True)

    # 总结报告
    summary = {
        "experiment": experiment_name,
        "run_dir": str(run_dir),
        "seed": seed,
        "architecture": architecture,
        "n_general": data_info["n_general"],
        "n_confidential": data_info["n_confidential"],
        "n_nodes": data_info["n_nodes"],
        "n_edges": n_edges,
        "train_samples": len(data_info["train_data"]),
        "test_samples": len(data_info["test_data"]),
        "best_test_loss": train_results["best_test_loss"],
        "final_test_mse_targets": train_results["final_test_mse_targets"],
        "final_test_mse_all": train_results["final_test_mse_all"],
        "final_test_mse_excluded": train_results["final_test_mse_excluded"],
        "loss_target_names": train_results["loss_target_names"],
        "excluded_target_names": train_results["excluded_target_names"],
        "baseline_loss_masking": baseline_loss,
        "final_alpha": {m: float(v) for m, v in zip(metrics, train_results["final_alpha"])},
        "per_target_r2": train_results["per_target_r2"],
        "per_target_mse": train_results["per_target_mse"],
        "top10_sensitive_fields": sensitivity_df.head(10)["field"].tolist(),
    }
    _save_json(results_dir / "summary.json", summary)

    # 尝试绘图
    _try_plot(train_results, sensitivity_df, metrics, run_dir)

    print("\n" + "=" * 60)
    print(f"  全部完成！结果已导出至: {run_dir}")
    print("=" * 60)


def _try_plot(train_results: dict, sensitivity_df: pd.DataFrame, metrics: list[str], run_dir: Path) -> None:
    """尝试生成可视化图表（matplotlib不可用时静默跳过）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. 训练/测试损失曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_results["history"]["train_loss"], label="Train Loss", alpha=0.8)
    ax.plot(train_results["history"]["test_loss"], label="Test Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    # 2. α权重柱状图
    fig, ax = plt.subplots(figsize=(6, 4))
    alpha = train_results["final_alpha"]
    ax.bar(metrics, alpha, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"])
    ax.set_ylabel("Weight")
    ax.set_title("Learned Correlation Metric Weights (α)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "alpha_weights.png", dpi=180)
    plt.close(fig)

    # 3. 敏感度排名条形图（前20）
    top = sensitivity_df.head(20).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e15759" if s > 0.1 else "#f28e2b" if s > 0.01 else "#76b7b2" for s in top["sensitivity"]]
    ax.barh(range(len(top) - 1, -1, -1), top["sensitivity"].values, color=colors)
    ax.set_yticks(range(len(top) - 1, -1, -1))
    ax.set_yticklabels(top["field"].values, fontsize=8)
    ax.set_xlabel("Masking Sensitivity")
    ax.set_title("Top 20 General Fields by Sensitivity")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "sensitivity_ranking.png", dpi=180)
    plt.close(fig)

    # 4. 逐目标R²柱状图
    r2_data = train_results["per_target_r2"]
    fields = sorted(r2_data.keys(), key=lambda k: -r2_data[k])
    r2_values = [r2_data[f] for f in fields]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_r2 = ["#e15759" if v > 0.8 else "#f28e2b" if v > 0.5 else "#76b7b2" for v in r2_values]
    ax.barh(range(len(fields) - 1, -1, -1), r2_values, color=colors_r2)
    ax.set_yticks(range(len(fields) - 1, -1, -1))
    ax.set_yticklabels(fields, fontsize=8)
    ax.set_xlabel("Test R²")
    ax.set_title("Per-Confidential Field Reconstruction R²")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_target_r2.png", dpi=180)
    plt.close(fig)

    print(f"  可视化图表已保存至: {plots_dir}")


if __name__ == "__main__":
    main()
