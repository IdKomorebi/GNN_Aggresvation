"""
DNN_Aggresvation3 主流水线脚本。

完整流程：
1. 数据加载与预处理
2. 计算5种相关性指标张量
3. 构建图结构（KNN+阈值边掩码）
4. 训练端到端推断驱动GNN（任务：还原Confidential真实值）
5. 遮蔽重要性法提取General字段敏感度
6. 导出结果
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNN_Aggresvation3: 端到端推断驱动的敏感度评估流水线"
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

    # 解析路径
    cfg["dataset"]["csv_path"] = _resolve_path(cfg["dataset"]["csv_path"])
    output_root = Path(_resolve_path(cfg["dataset"].get("output_dir", "outputs")))

    # 创建运行目录
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = str(cfg.get("runtime", {}).get("device", "cpu"))
    device = torch.device(device_str)

    print("\n" + "=" * 60)
    print("  DNN_Aggresvation3: 端到端推断驱动的敏感度评估")
    print("=" * 60)
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

    rel_dir = run_dir / "relationships"
    rel_dir.mkdir(parents=True, exist_ok=True)

    metric_tensor = compute_metric_tensor(
        data_info["raw_df"],
        data_info["all_columns"],
        metrics=metrics,
        sample_size=int(corr_cfg.get("sample_size", 3000)),
        expensive_sample_size=int(corr_cfg.get("expensive_sample_size", 1200)),
        seed=seed,
    )
    np.save(rel_dir / "metric_tensor.npy", metric_tensor)
    _save_json(rel_dir / "metrics.json", {"metrics": metrics, "columns": data_info["all_columns"]})
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
    ).to(device)

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
        }
    ).to_csv(train_dir / "log.csv", index=False)

    torch.save(model.state_dict(), train_dir / "model.pt")

    # α权重
    alpha_df = pd.DataFrame(
        {"metric": metrics, "alpha": train_results["final_alpha"]}
    )
    alpha_df.to_csv(train_dir / "alpha_weights.csv", index=False)

    # 逐目标R²
    pd.DataFrame(
        [
            {"confidential_field": k, "test_r2": v}
            for k, v in train_results["per_target_r2"].items()
        ]
    ).sort_values("test_r2", ascending=False).to_csv(
        train_dir / "per_target_r2.csv", index=False
    )

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
    )

    # ================================================================
    # 步骤6: 导出结果
    # ================================================================
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_df.to_csv(results_dir / "sensitivity_scores.csv", index=True)

    # 总结报告
    summary = {
        "run_dir": str(run_dir),
        "seed": seed,
        "n_general": data_info["n_general"],
        "n_confidential": data_info["n_confidential"],
        "n_nodes": data_info["n_nodes"],
        "n_edges": n_edges,
        "train_samples": len(data_info["train_data"]),
        "test_samples": len(data_info["test_data"]),
        "best_test_loss": train_results["best_test_loss"],
        "baseline_loss_masking": baseline_loss,
        "final_alpha": {m: float(v) for m, v in zip(metrics, train_results["final_alpha"])},
        "per_target_r2": train_results["per_target_r2"],
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
