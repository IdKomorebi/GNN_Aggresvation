"""
单目标全连接 DNN probe（DNN40）。

目的：
- 每个 Confidential 字段单独训练一个 MLP，只使用原始 44 个 General 字段作为输入
- 作为非图、全连接 DNN baseline，对比相关系数构图 GNN 的推断效果
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import prepare_data


class TargetMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str((PROJECT_ROOT / path).resolve())


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _load_latest_summary_by_experiment(output_root: Path) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for summary_path in sorted(output_root.glob("run_*/results/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        experiment = summary.get("experiment")
        if experiment:
            summaries[experiment] = summary
    return summaries


def _train_one_target(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe_cfg = cfg.get("target_probe", {})
    hidden_dims = [int(v) for v in probe_cfg.get("hidden_dims", [128, 64])]
    dropout = float(probe_cfg.get("dropout", 0.15))
    lr = float(probe_cfg.get("lr", 0.0005))
    weight_decay = float(probe_cfg.get("weight_decay", 0.0005))
    batch_size = int(probe_cfg.get("batch_size", 128))
    epochs = int(probe_cfg.get("epochs", 600))
    patience = int(probe_cfg.get("patience", 80))

    model = TargetMLP(x_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_x = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    test_y = torch.as_tensor(y_test, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_n += batch_x.size(0)
        train_loss = total_loss / total_n
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = loss_fn(test_pred, test_y).item()

        history.append({"epoch": epoch, "train_mse": train_loss, "test_mse": test_loss})
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(test_x).cpu().numpy()

    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    mse = float(np.mean((y_test - y_pred) ** 2))

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_test_mse": best_loss,
        "final_test_mse": mse,
        "test_r2": float(r2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DNN40 single-target MLP probes")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "multi.yaml"),
    )
    args = parser.parse_args()

    cfg = _load_config(os.path.abspath(args.config))
    cfg["dataset"]["csv_path"] = _resolve_path(cfg["dataset"]["csv_path"])
    output_root = Path(_resolve_path(cfg["dataset"].get("output_dir", "outputs")))

    probe_cfg = cfg.setdefault("target_probe", {})
    probe_cfg.setdefault("hidden_dims", [128, 64])
    probe_cfg.setdefault("dropout", 0.15)
    probe_cfg.setdefault("lr", 0.0005)
    probe_cfg.setdefault("weight_decay", 0.0005)
    probe_cfg.setdefault("batch_size", 128)
    probe_cfg.setdefault("epochs", 600)
    probe_cfg.setdefault("patience", 80)

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    device = torch.device(str(cfg.get("runtime", {}).get("device", "cpu")))
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "dnn_probe_general_only" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  DNN_Aggresvation40: 单目标全连接DNN probe")
    print("=" * 72)
    print(f"  运行目录: {run_dir}")
    print(f"  随机种子: {seed}")
    print(f"  设备: {device}")
    print(f"  MLP hidden_dims: {probe_cfg['hidden_dims']}")

    data_info = prepare_data(cfg)
    n_general = data_info["n_general"]
    x_train = data_info["train_data"][:, :n_general]
    x_test = data_info["test_data"][:, :n_general]
    y_train_all = data_info["train_data"][:, data_info["confidential_indices"]]
    y_test_all = data_info["test_data"][:, data_info["confidential_indices"]]

    summaries = _load_latest_summary_by_experiment(output_root)
    baseline_r2 = summaries.get("baseline_gcn", {}).get("per_target_r2", {})
    gat_mask_r2 = summaries.get("gat_loss_mask", {}).get("per_target_r2", {})

    rows = []
    histories_dir = run_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for target_idx, target_name in enumerate(data_info["confidential"]):
        print(f"\n  [Target Probe] {target_name}")
        result = _train_one_target(
            x_train=x_train,
            y_train=y_train_all[:, target_idx],
            x_test=x_test,
            y_test=y_test_all[:, target_idx],
            cfg=cfg,
            seed=seed + target_idx,
            device=device,
        )
        pd.DataFrame(result["history"]).to_csv(
            histories_dir / f"{target_name}.csv", index=False
        )
        torch.save(result["model"].state_dict(), models_dir / f"{target_name}.pt")

        row = {
            "confidential_field": target_name,
            "probe_r2": result["test_r2"],
            "probe_mse": result["final_test_mse"],
            "best_epoch": result["best_epoch"],
            "baseline_gcn_r2": baseline_r2.get(target_name),
            "gat_loss_mask_r2": gat_mask_r2.get(target_name),
        }
        if row["baseline_gcn_r2"] is not None:
            row["probe_minus_baseline_gcn_r2"] = row["probe_r2"] - row["baseline_gcn_r2"]
        if row["gat_loss_mask_r2"] is not None:
            row["probe_minus_gat_loss_mask_r2"] = row["probe_r2"] - row["gat_loss_mask_r2"]
        rows.append(row)

        print(
            f"    R2={result['test_r2']:.4f}, "
            f"MSE={result['final_test_mse']:.6f}, "
            f"best_epoch={result['best_epoch']}"
        )

    result_df = pd.DataFrame(rows).sort_values("probe_r2", ascending=False)
    result_df.to_csv(run_dir / "target_probe_results.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "seed": seed,
        "device": str(device),
        "n_general": data_info["n_general"],
        "n_confidential": data_info["n_confidential"],
        "input_policy": "original_general_only",
        "probe_config": probe_cfg,
        "mean_probe_r2": float(result_df["probe_r2"].mean()),
        "median_probe_r2": float(result_df["probe_r2"].median()),
        "results": result_df.to_dict(orient="records"),
    }
    _save_json(run_dir / "summary.json", summary)

    print("\n" + "=" * 72)
    print("  单目标全连接DNN probe完成")
    print("=" * 72)
    print(result_df[["confidential_field", "probe_r2", "baseline_gcn_r2", "gat_loss_mask_r2"]])
    print(f"\n  结果: {run_dir / 'target_probe_results.csv'}")


if __name__ == "__main__":
    main()
