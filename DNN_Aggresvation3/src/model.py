"""
端到端推断驱动的GNN模型（v2）。

核心思想：
- 任务是用General节点的真实值还原Confidential节点的真实值
- 图边权由5种相关性指标的可学习融合权重α端到端优化
- Confidential节点的输入为零（模拟攻击者无法观测）
- 敏感度不是模型的直接输出，而是训练后通过遮蔽实验获得的副产品

v2改进：
- 添加BatchNorm + 残差连接，缩小训练-测试gap
- 增加可学习的Confidential节点嵌入（代替固定零向量）
- 更大的输出头（双层MLP）
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class InferenceDrivenGNN(nn.Module):
    """
    推断驱动的图神经网络 v2。

    相比v1的改进：
    1. 每层消息传递后增加BatchNorm，稳定训练
    2. 残差连接：h_{l+1} = BN(ReLU(W_self·h_l + W_neigh·msg_l)) + h_l
    3. 可学习的Confidential节点嵌入（不再是固定零向量）
    4. 双层MLP输出头替代单层Linear
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

        # 固定张量
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
        self.n_confidential = len(confidential_indices)
        self.n_metrics = metric_tensor.shape[2]
        self.hidden_dim = hidden_dim

        # 可学习的相关性融合权重 β → α = softmax(β)
        self.beta = nn.Parameter(torch.zeros(self.n_metrics, dtype=torch.float32))

        # 标量 → 隐藏空间投射（General节点用）
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

        # Confidential节点的可学习嵌入
        self.conf_embedding = nn.Parameter(
            torch.randn(self.n_confidential, hidden_dim) * 0.01
        )

        # 消息传递层 + BatchNorm
        self.self_layers = nn.ModuleList()
        self.neigh_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(n_nodes))

        self.dropout = nn.Dropout(dropout)

        # 双层MLP输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_alpha(self) -> torch.Tensor:
        """返回softmax归一化后的相关性融合权重。"""
        return torch.softmax(self.beta, dim=0)

    def compute_adjacency(self) -> torch.Tensor:
        """根据当前α计算行归一化的邻接矩阵。"""
        alpha = self.get_alpha()
        A_raw = torch.einsum("ijk,k->ij", self.metric_tensor, alpha)
        A = A_raw * self.edge_mask
        A = A * (1.0 - torch.eye(self.n_nodes, device=A.device))
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
        B = node_values.shape[0]

        # General节点：标量升维 (B, N, 1) → (B, N, H)
        h_proj = self.input_proj(node_values.unsqueeze(-1))

        # Confidential节点：使用可学习嵌入替代零向量投影
        # 通过scatter方式避免inplace操作
        conf_emb = self.conf_embedding.unsqueeze(0).expand(B, -1, -1)  # (B, C, H)

        # 构造索引用于scatter
        idx = self.conf_indices.view(1, -1, 1).expand(B, -1, self.hidden_dim)
        h = h_proj.scatter(1, idx, conf_emb)

        # 邻接矩阵
        A = self.compute_adjacency()

        # 消息传递 + 残差连接 + BatchNorm
        for self_layer, neigh_layer, bn in zip(
            self.self_layers, self.neigh_layers, self.batch_norms
        ):
            messages = torch.matmul(A, h)
            h_new = self_layer(h) + neigh_layer(messages)
            h_new = bn(h_new)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            # 残差连接（非inplace）
            h = h_new + h

        # 提取Confidential节点的隐藏表示 → 预测
        h_conf = h[:, self.conf_indices, :]  # (B, C, H)
        pred = self.output_head(h_conf).squeeze(-1)  # (B, C)

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
