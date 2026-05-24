"""
训练模块（DNN_Aggresvation4）。

相对子项目3新增：
- 可配置排除部分Confidential字段的训练损失贡献
- 同时记录训练目标MSE、全目标MSE、被排除目标MSE
- 保留滑动窗口构造的修复版实现，默认window_size=1
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import InferenceDrivenGNN


def _build_windowed_samples(
    data_matrix: np.ndarray,
    confidential_indices: list[int],
    window_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构建滑动窗口训练样本。

    当window_size=1时，退化为逐时间步方式。注意：必须先copy目标，再copy窗口，
    防止numpy view的inplace修改污染原始data_matrix。
    """
    T, N = data_matrix.shape
    C = len(confidential_indices)
    w = window_size

    n_samples = T - w + 1
    features = np.zeros((n_samples, N, w), dtype=np.float32)
    targets = np.zeros((n_samples, C), dtype=np.float32)

    for i in range(n_samples):
        targets[i] = data_matrix[i + w - 1, confidential_indices].copy()
        window = data_matrix[i : i + w, :].copy()
        feat = window.T
        feat[confidential_indices, :] = 0.0
        features[i] = feat

    return features, targets


def resolve_loss_target_indices(data_info: dict, cfg: dict) -> tuple[list[int], list[str], list[str]]:
    """根据配置解析参与训练损失的Confidential目标列。"""
    confidential = data_info["confidential"]
    training_exclude = cfg.get("training", {}).get("loss_exclude_confidential", [])
    fields_exclude = cfg.get("fields", {}).get("loss_exclude_confidential", [])
    excluded = sorted(set(training_exclude) | set(fields_exclude))

    unknown = sorted(set(excluded) - set(confidential))
    if unknown:
        print(f"  警告: loss_exclude_confidential中不存在的字段将被忽略: {unknown}")

    excluded_set = set(excluded)
    indices = [i for i, name in enumerate(confidential) if name not in excluded_set]
    names = [confidential[i] for i in indices]
    effective_excluded = [name for name in confidential if name in excluded_set]

    if not indices:
        raise ValueError("loss_exclude_confidential排除了所有Confidential字段，无法训练。")

    return indices, names, effective_excluded


def _elementwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn_name: str) -> torch.Tensor:
    if loss_fn_name in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred, target, reduction="none")
    return F.mse_loss(pred, target, reduction="none")


def _selected_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_indices: list[int],
    loss_fn_name: str,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = _elementwise_loss(pred, target, loss_fn_name)
    idx = torch.as_tensor(target_indices, dtype=torch.long, device=pred.device)
    selected = losses.index_select(1, idx)
    if target_weights is None:
        return selected.mean()
    weights = target_weights.index_select(0, idx).view(1, -1)
    weighted = selected * weights
    return weighted.sum() / weights.sum().clamp(min=1e-12) / selected.shape[0]


def _mse_for_indices(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_indices: list[int] | None = None,
) -> float:
    if target_indices is None:
        return F.mse_loss(pred, target).item()
    idx = torch.as_tensor(target_indices, dtype=torch.long, device=pred.device)
    return F.mse_loss(pred.index_select(1, idx), target.index_select(1, idx)).item()


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
    loss_target_indices, loss_target_names, excluded_target_names = resolve_loss_target_indices(
        data_info, cfg
    )
    excluded_target_indices = [
        i for i, name in enumerate(data_info["confidential"]) if name in set(excluded_target_names)
    ]

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
    print(f"  参与训练损失的目标数: {len(loss_target_names)} / {len(data_info['confidential'])}")
    if excluded_target_names:
        print(f"  排除训练损失目标: {excluded_target_names}")

    configured_weights = cfg.get("training", {}).get("target_loss_weights", {})
    target_weight_values = np.ones(len(data_info["confidential"]), dtype=np.float32)
    if configured_weights:
        unknown_weights = sorted(set(configured_weights) - set(data_info["confidential"]))
        if unknown_weights:
            print(f"  警告: target_loss_weights中不存在的字段将被忽略: {unknown_weights}")
        for idx, name in enumerate(data_info["confidential"]):
            if name in configured_weights:
                target_weight_values[idx] = float(configured_weights[name])
        shown = {
            name: float(target_weight_values[idx])
            for idx, name in enumerate(data_info["confidential"])
            if target_weight_values[idx] != 1.0 and name in loss_target_names
        }
        if shown:
            print(f"  训练目标loss权重: {shown}")

    train_dataset = TensorDataset(
        torch.as_tensor(train_features),
        torch.as_tensor(train_targets),
    )
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_x = torch.as_tensor(test_features, device=device)
    test_y = torch.as_tensor(test_targets, device=device)
    target_weights_t = torch.as_tensor(target_weight_values, dtype=torch.float32, device=device)

    base_lr = cfg["training"]["lr"]
    alpha_lr_multiplier = float(cfg["training"].get("alpha_lr_multiplier", 1.0))
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    if alpha_lr_multiplier != 1.0:
        alpha_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in {"beta_general", "beta_confidential"}:
                alpha_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
                {
                    "params": alpha_params,
                    "lr": base_lr * alpha_lr_multiplier,
                    "weight_decay": weight_decay,
                },
            ]
        )
        print(f"  alpha参数学习率倍率: {alpha_lr_multiplier:g}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )

    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=base_lr * 0.01
    )

    patience = cfg["training"].get("patience", 20)
    best_test_loss = float("inf")
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_mse_all": [],
        "test_mse_excluded": [],
        "alpha": [],
    }

    print("\n" + "=" * 60)
    print("  [模型训练] 开始端到端推断驱动GNN训练")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = _selected_loss(
                pred,
                batch_y,
                loss_target_indices,
                loss_fn_name,
                target_weights=target_weights_t,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
        train_loss = epoch_loss / n_samples

        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = _mse_for_indices(test_pred, test_y, loss_target_indices)
            test_mse_all = _mse_for_indices(test_pred, test_y)
            test_mse_excluded = (
                _mse_for_indices(test_pred, test_y, excluded_target_indices)
                if excluded_target_indices
                else float("nan")
            )
            alpha = model.get_alpha().cpu().numpy()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_mse_all"].append(test_mse_all)
        history["test_mse_excluded"].append(test_mse_excluded)
        history["alpha"].append(alpha.copy())

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs or patience_counter >= patience:
            alpha_str = ", ".join(f"{v:.4f}" for v in alpha)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:4d} | "
                f"训练目标MSE: {train_loss:.6f} | "
                f"测试目标MSE: {test_loss:.6f} | "
                f"测试全量MSE: {test_mse_all:.6f} | "
                f"lr: {current_lr:.6f} | "
                f"alpha: [{alpha_str}]"
            )

        if patience_counter >= patience:
            print(f"  早停触发于 Epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_pred_t = model(test_x)
        final_pred = final_pred_t.cpu().numpy()
        final_alpha = model.get_alpha().cpu().numpy()
        final_test_mse_all = _mse_for_indices(final_pred_t, test_y)
        final_test_mse_targets = _mse_for_indices(final_pred_t, test_y, loss_target_indices)
        final_test_mse_excluded = (
            _mse_for_indices(final_pred_t, test_y, excluded_target_indices)
            if excluded_target_indices
            else float("nan")
        )

    per_target_r2 = {}
    per_target_mse = {}
    for c_idx, col_name in enumerate(data_info["confidential"]):
        y_true = test_targets[:, c_idx]
        y_pred = final_pred[:, c_idx]
        mse = float(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        per_target_mse[col_name] = mse
        per_target_r2[col_name] = float(r2)

    print(f"\n  最优测试目标MSE: {best_test_loss:.6f}")
    print(f"  最终测试全量MSE: {final_test_mse_all:.6f}")
    if excluded_target_names:
        print(f"  最终被排除目标MSE: {final_test_mse_excluded:.6f}")
    print(f"  逐Confidential字段 R²:")
    for name, r2 in sorted(per_target_r2.items(), key=lambda x: -x[1]):
        marker = " (excluded)" if name in set(excluded_target_names) else ""
        print(f"    {name:45s} R² = {r2:.4f}{marker}")

    return {
        "history": history,
        "best_test_loss": best_test_loss,
        "final_test_mse_all": final_test_mse_all,
        "final_test_mse_targets": final_test_mse_targets,
        "final_test_mse_excluded": final_test_mse_excluded,
        "final_alpha": final_alpha,
        "test_predictions": final_pred,
        "test_targets": test_targets,
        "test_features": test_features,
        "per_target_r2": per_target_r2,
        "per_target_mse": per_target_mse,
        "loss_target_indices": loss_target_indices,
        "loss_target_names": loss_target_names,
        "excluded_target_names": excluded_target_names,
    }
