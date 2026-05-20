"""
遮蔽重要性法计算General字段敏感度。

核心逻辑：
- 基线：所有General字段可见，Confidential字段不可见 → 测量还原误差
- 遮蔽：逐一将每个General字段置零 → 测量还原误差增量
- 敏感度 = (遮蔽后误差 - 基线误差) / 基线误差

敏感度越高，说明该General字段对还原Confidential真实值的贡献越大，
即其隐性泄露风险越高。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .model import InferenceDrivenGNN


def compute_masking_sensitivity(
    model: InferenceDrivenGNN,
    test_data: np.ndarray,
    data_info: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    """
    遮蔽重要性法计算敏感度。

    参数:
        model: 训练好的GNN模型
        test_data: (T_test, N) 标准化后的测试数据
        data_info: 数据信息字典
        device: 计算设备

    返回:
        sensitivity_df: 包含敏感度排名的DataFrame
        baseline_loss: 基线还原误差
    """
    conf_indices = data_info["confidential_indices"]
    general = data_info["general"]
    n_general = data_info["n_general"]

    # 构造基线输入：Confidential位置置零
    node_values = test_data.copy()
    node_values[:, conf_indices] = 0.0
    targets = test_data[:, conf_indices]

    model.eval()

    # 基线损失
    with torch.no_grad():
        x_base = torch.as_tensor(node_values, dtype=torch.float32, device=device)
        y_true = torch.as_tensor(targets, dtype=torch.float32, device=device)
        pred_base = model(x_base)
        baseline_loss = torch.nn.functional.mse_loss(pred_base, y_true).item()

    print(f"\n  基线测试损失 (所有General可见): {baseline_loss:.6f}")
    print(f"  开始逐一遮蔽 {n_general} 个General字段...")

    # 逐一遮蔽每个General字段
    rows = []
    for i in range(n_general):
        masked_values = node_values.copy()
        masked_values[:, i] = 0.0  # 遮蔽第i个General字段

        with torch.no_grad():
            x_masked = torch.as_tensor(
                masked_values, dtype=torch.float32, device=device
            )
            pred_masked = model(x_masked)
            masked_loss = torch.nn.functional.mse_loss(pred_masked, y_true).item()

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

    # 按敏感度降序排列
    sensitivity_df = (
        pd.DataFrame(rows)
        .sort_values("sensitivity", ascending=False)
        .reset_index(drop=True)
    )
    sensitivity_df.index = sensitivity_df.index + 1
    sensitivity_df.index.name = "rank"

    # 打印前15名
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
