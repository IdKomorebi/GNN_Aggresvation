"""
训练模块。

职责：
- 将数据矩阵转换为GNN输入（Confidential位置置零）
- 批量化训练，监督信号为Confidential的真实标准化值
- 早停机制（基于测试集损失）
- 返回训练历史、最优模型参数和学到的α权重
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import InferenceDrivenGNN


def _prepare_node_values(
    data_matrix: np.ndarray,
    confidential_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    将完整数据矩阵转换为GNN输入和监督目标。

    - node_values: Confidential位置置零（攻击者看不到）
    - targets: Confidential位置的真实值

    参数:
        data_matrix: (T, N) 标准化后的完整数据
        confidential_indices: Confidential节点在列中的索引

    返回:
        node_values: (T, N) float32
        targets:     (T, C) float32
    """
    node_values = data_matrix.copy()
    node_values[:, confidential_indices] = 0.0
    targets = data_matrix[:, confidential_indices].copy()
    return node_values.astype(np.float32), targets.astype(np.float32)


def train_model(
    model: InferenceDrivenGNN,
    train_data: np.ndarray,
    test_data: np.ndarray,
    data_info: dict,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    训练推断驱动的GNN。

    返回包含训练历史、最终α权重和测试集预测的字典。
    """
    conf_indices = data_info["confidential_indices"]

    # 构造输入与目标
    train_inputs, train_targets = _prepare_node_values(train_data, conf_indices)
    test_inputs, test_targets = _prepare_node_values(test_data, conf_indices)

    # DataLoader
    train_dataset = TensorDataset(
        torch.as_tensor(train_inputs),
        torch.as_tensor(train_targets),
    )
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集一次性加载（量级不大）
    test_x = torch.as_tensor(test_inputs, device=device)
    test_y = torch.as_tensor(test_targets, device=device)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    epochs = cfg["training"]["epochs"]
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
            loss = torch.nn.functional.mse_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
        train_loss = epoch_loss / n_samples

        # ---- 测试 ----
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
            print(
                f"  Epoch {epoch:4d} | "
                f"训练损失: {train_loss:.6f} | "
                f"测试损失: {test_loss:.6f} | "
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
