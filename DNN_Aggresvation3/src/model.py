"""
端到端推断驱动的GNN模型。

核心思想：
- 任务是用General节点的真实值还原Confidential节点的真实值
- 图边权由5种相关性指标的可学习融合权重α端到端优化
- Confidential节点的输入为零（模拟攻击者无法观测）
- 敏感度不是模型的直接输出，而是训练后通过遮蔽实验获得的副产品
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class InferenceDrivenGNN(nn.Module):
    """
    推断驱动的图神经网络。

    参数:
        metric_tensor: (N, N, K) 相关性张量
        edge_mask:     (N, N) 二值掩码，定义哪些边存在
        n_nodes:       节点总数 N
        n_general:     General节点数 G
        confidential_indices: Confidential节点在columns中的索引列表
        hidden_dim:    隐藏层维度
        num_layers:    消息传递层数
        dropout:       Dropout比率
    """

    def __init__(
        self,
        metric_tensor: np.ndarray,
        edge_mask: np.ndarray,
        n_nodes: int,
        n_general: int,
        confidential_indices: list[int],
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        # 固定张量（不参与梯度）
        self.register_buffer(
            "metric_tensor",
            torch.as_tensor(metric_tensor, dtype=torch.float32),
        )
        self.register_buffer(
            "edge_mask",
            torch.as_tensor(edge_mask, dtype=torch.float32),
        )
        self.register_buffer(
            "conf_indices",
            torch.as_tensor(confidential_indices, dtype=torch.long),
        )

        self.n_nodes = n_nodes
        self.n_general = n_general
        self.n_metrics = metric_tensor.shape[2]

        # 可学习的相关性融合权重 β → α = softmax(β)
        self.beta = nn.Parameter(torch.zeros(self.n_metrics, dtype=torch.float32))

        # 标量(1维) → 隐藏空间(hidden_dim维) 投射
        self.input_proj = nn.Linear(1, hidden_dim)

        # 消息传递层
        self.self_layers = nn.ModuleList()
        self.neigh_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # 输出头：每个Confidential节点预测1个标量
        self.output_head = nn.Linear(hidden_dim, 1)

    def get_alpha(self) -> torch.Tensor:
        """返回softmax归一化后的相关性融合权重。"""
        return torch.softmax(self.beta, dim=0)

    def compute_adjacency(self) -> torch.Tensor:
        """根据当前α计算行归一化的邻接矩阵。"""
        alpha = self.get_alpha()
        # 融合: A_raw[i,j] = Σ_k α_k · R[i,j,k]
        A_raw = torch.einsum("ijk,k->ij", self.metric_tensor, alpha)
        # 应用边掩码
        A = A_raw * self.edge_mask
        # 对角线置零
        A = A * (1.0 - torch.eye(self.n_nodes, device=A.device))
        # 行归一化
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return A / row_sum

    def forward(self, node_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            node_values: (B, N) 每个节点在该batch中各时间步的标量值
                         Confidential节点位置已被置零

        返回:
            (B, C) 对Confidential节点的预测值
        """
        # 标量升维: (B, N) → (B, N, 1) → (B, N, H)
        h = self.input_proj(node_values.unsqueeze(-1))
        h = torch.relu(h)

        # 邻接矩阵 (N, N)
        A = self.compute_adjacency()

        # 消息传递
        for self_layer, neigh_layer in zip(self.self_layers, self.neigh_layers):
            # A: (N,N), h: (B,N,H) → messages: (B,N,H)
            messages = torch.matmul(A, h)
            h = torch.relu(self_layer(h) + neigh_layer(messages))
            h = self.dropout(h)

        # 提取Confidential节点的隐藏表示 (B, C, H) → 预测 (B, C)
        h_conf = h[:, self.conf_indices, :]
        pred = self.output_head(h_conf).squeeze(-1)

        return pred


def build_edge_mask(
    metric_tensor: np.ndarray,
    top_k: int = 10,
    threshold: float = 0.08,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    基于平均相关性构建二值边掩码。

    对每个节点，保留其平均相关性最高的top_k个邻居，
    同时保留所有平均相关性超过threshold的边。
    """
    # 平均相关性作为候选权重
    avg_corr = metric_tensor.mean(axis=2)
    np.fill_diagonal(avg_corr, 0.0)

    n = avg_corr.shape[0]
    mask = avg_corr >= threshold

    for i in range(n):
        order = np.argsort(avg_corr[i])[::-1]
        neighbors = [j for j in order if j != i][:top_k]
        for j in neighbors:
            mask[i, j] = True

    if symmetrize:
        mask = mask | mask.T

    return mask.astype(np.float32)
