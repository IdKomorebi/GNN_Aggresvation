"""
训练模块 v3。

改进：
- 支持滑动窗口特征（每个节点输入不再是1个标量，而是近w个时间步的值）
- 损失函数可选MSE或SmoothL1（Huber Loss，对不可推断字段的大误差更鲁棒）
- 保留CosineAnnealing学习率调度器和早停机制
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import InferenceDrivenGNN


def _build_windowed_samples(
    data_matrix: np.ndarray,
    confidential_indices: list[int],
    window_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建滑动窗口训练样本。

    当window_size=1时，退化为原始的逐时间步方式。
    当window_size=w时，每个样本的节点特征为最近w个时间步的值组成的向量。

    参数:
        data_matrix: (T, N) 标准化后的完整数据
        confidential_indices: Confidential节点在列中的索引
        window_size: 滑动窗口大小

    返回:
        node_features: (T-w+1, N, w) float32 — Confidential位置已置零
        targets:       (T-w+1, C) float32 — 最后一个时间步的Confidential真实值
    """
    T, N = data_matrix.shape
    C = len(confidential_indices)
    w = window_size

    n_samples = T - w + 1
    features = np.zeros((n_samples, N, w), dtype=np.float32)
    targets = np.zeros((n_samples, C), dtype=np.float32)

    for i in range(n_samples):
        # 先提取目标值（在修改之前）
        targets[i] = data_matrix[i + w - 1, confidential_indices].copy()
        # 构造节点特征（必须copy，否则会修改原始data_matrix）
        window = data_matrix[i : i + w, :].copy()  # (w, N) — copy!
        feat = window.T  # (N, w)
        # Confidential位置置零
        feat[confidential_indices, :] = 0.0
        features[i] = feat

    return features, targets


def train_model(
    model: InferenceDrivenGNN,
    train_data: np.ndarray,
    test_data: np.ndarray,
    data_info: dict,
    cfg: dict,
    device: torch.device,
) -> dict:
    """训练推断驱动的GNN。"""
    conf_indices = data_info["confidential_indices"]
    window_size = cfg.get("model", {}).get("window_size", 1)
    loss_fn_name = cfg.get("training", {}).get("loss_fn", "mse")

    # 构造带窗口的输入与目标
    train_features, train_targets = _build_windowed_samples(
        train_data, conf_indices, window_size
    )
    test_features, test_targets = _build_windowed_samples(
        test_data, conf_indices, window_size
    )
    print(f"  窗口大小: {window_size}")
    print(f"  训练样本数: {len(train_features)}, 测试样本数: {len(test_features)}")
    print(f"  节点特征维度: {train_features.shape[2]}")
    print(f"  损失函数: {loss_fn_name}")

    # DataLoader
    train_dataset = TensorDataset(
        torch.as_tensor(train_features),
        torch.as_tensor(train_targets),
    )
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集一次性加载
    test_x = torch.as_tensor(test_features, device=device)
    test_y = torch.as_tensor(test_targets, device=device)

    # 损失函数
    if loss_fn_name == "smooth_l1" or loss_fn_name == "huber":
        loss_fn = torch.nn.SmoothL1Loss()
    else:
        loss_fn = torch.nn.MSELoss()

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    # 学习率调度器
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg["training"]["lr"] * 0.01
    )

    patience = cfg["training"].get("patience", 20)
    best_test_loss = float("inf")
    patience_counter = 0
    best_state = None

    history = {"train_loss": [], "test_loss": [], "alpha": []}

    print("\n" + "=" * 60)
    print("  [模型训练] 开始端到端推断驱动GNN训练")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
        train_loss = epoch_loss / n_samples

        # 学习率调度
        scheduler.step()

        # ---- 测试（始终用MSE评估，以便跨版本对比） ----
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = torch.nn.functional.mse_loss(test_pred, test_y).item()
            alpha = model.get_alpha().cpu().numpy()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["alpha"].append(alpha.copy())

        # ---- 早停 ----
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # ---- 日志 ----
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs or patience_counter >= patience:
            alpha_str = ", ".join(f"{v:.4f}" for v in alpha)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:4d} | "
                f"训练损失: {train_loss:.6f} | "
                f"测试MSE: {test_loss:.6f} | "
                f"lr: {current_lr:.6f} | "
                f"alpha: [{alpha_str}]"
            )

        if patience_counter >= patience:
            print(f"  早停触发于 Epoch {epoch} (patience={patience})")
            break

    # 恢复最优模型
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 最终评估
    with torch.no_grad():
        final_pred = model(test_x).cpu().numpy()
        final_alpha = model.get_alpha().cpu().numpy()

    # 逐Confidential字段计算R²
    per_target_r2 = {}
    for c_idx, col_name in enumerate(data_info["confidential"]):
        y_true = test_targets[:, c_idx]
        y_pred = final_pred[:, c_idx]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        per_target_r2[col_name] = float(r2)

    print(f"\n  最优测试损失: {best_test_loss:.6f}")
    print(f"  逐Confidential字段 R²:")
    for name, r2 in sorted(per_target_r2.items(), key=lambda x: -x[1]):
        print(f"    {name:45s} R² = {r2:.4f}")

    return {
        "history": history,
        "best_test_loss": best_test_loss,
        "final_alpha": final_alpha,
        "test_predictions": final_pred,
        "test_targets": test_targets,
        "per_target_r2": per_target_r2,
    }
