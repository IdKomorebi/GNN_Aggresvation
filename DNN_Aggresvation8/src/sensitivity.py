"""
遮蔽重要性法计算General字段敏感度（DNN_Aggresvation4）。

敏感度默认使用训练目标集合计算MSE。如果配置排除了不可推断字段，
这些字段仍保留在图中作为节点，但不参与遮蔽敏感度的损失口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .model import InferenceDrivenGNN
from .train import _build_windowed_samples


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_indices: list[int] | None,
) -> float:
    if target_indices is None:
        return F.mse_loss(pred, target).item()
    idx = torch.as_tensor(target_indices, dtype=torch.long, device=pred.device)
    return F.mse_loss(pred.index_select(1, idx), target.index_select(1, idx)).item()


def compute_masking_sensitivity(
    model: InferenceDrivenGNN,
    test_data: np.ndarray,
    data_info: dict,
    device: torch.device,
    window_size: int = 1,
    target_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, float]:
    """遮蔽重要性法计算敏感度。"""
    conf_indices = data_info["confidential_indices"]
    general = data_info["general"]
    n_general = data_info["n_general"]

    base_features, targets = _build_windowed_samples(
        test_data, conf_indices, window_size
    )

    model.eval()

    with torch.no_grad():
        x_base = torch.as_tensor(base_features, dtype=torch.float32, device=device)
        y_true = torch.as_tensor(targets, dtype=torch.float32, device=device)
        pred_base = model(x_base)
        baseline_loss = _masked_mse(pred_base, y_true, target_indices)

    target_desc = "训练目标" if target_indices is not None else "全部Confidential"
    print(f"\n  基线测试损失 ({target_desc}, 所有General可见): {baseline_loss:.6f}")
    print(f"  开始逐一遮蔽 {n_general} 个General字段...")

    rows = []
    for i in range(n_general):
        masked_features = base_features.copy()
        masked_features[:, i, :] = 0.0

        with torch.no_grad():
            x_masked = torch.as_tensor(
                masked_features, dtype=torch.float32, device=device
            )
            pred_masked = model(x_masked)
            masked_loss = _masked_mse(pred_masked, y_true, target_indices)

        loss_increase = masked_loss - baseline_loss
        sensitivity = loss_increase / max(baseline_loss, 1e-12)

        rows.append(
            {
                "field": general[i],
                "sensitivity": sensitivity,
                "masked_loss": masked_loss,
                "loss_increase": loss_increase,
            }
        )

    sensitivity_df = (
        pd.DataFrame(rows)
        .sort_values("sensitivity", ascending=False)
        .reset_index(drop=True)
    )
    sensitivity_df.index = sensitivity_df.index + 1
    sensitivity_df.index.name = "rank"

    print(f"\n  {'排名':<4}  {'字段名称':<42}  {'敏感度':>10}  {'遮蔽后损失':>12}")
    print("  " + "-" * 72)
    for rank, (_, row) in enumerate(sensitivity_df.head(15).iterrows(), 1):
        print(
            f"  {rank:<4}  "
            f"{row['field']:<42}  "
            f"{row['sensitivity']:>10.4f}  "
            f"{row['masked_loss']:>12.6f}"
        )

    return sensitivity_df, baseline_loss
