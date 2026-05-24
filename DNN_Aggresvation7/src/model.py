"""
端到端推断驱动的GNN模型（DNN_Aggresvation7）。

子项目7在子项目4基础上测试 edge-specific correlation weights：
- `gcn` / `gat`: 子项目4全局alpha基线
- `edge_gcn`: 每条边学习自己的相关性指标融合权重，再做GCN聚合
- `edge_gat`: 每条边学习自己的相关性指标融合权重，并对全图节点使用GAT
- `edge_conf_gat`: General节点使用edge-GCN，Confidential节点使用edge-GAT读取邻居

所有模式都保留：
- 5种相关性指标融合权重。全局模式使用一个alpha，edge模式使用每条边自己的alpha_ij
- Confidential节点可学习嵌入
- BatchNorm、残差连接和双层输出头
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InferenceDrivenGNN(nn.Module):
    """推断驱动GNN，支持全局alpha和边级alpha两类聚合方式。"""

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
        input_dim: int = 1,
        architecture: str = "gcn",
        attention_dropout: float = 0.05,
        edge_alpha_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if architecture not in {"gcn", "gat", "edge_gcn", "edge_gat", "edge_conf_gat"}:
            raise ValueError(f"Unknown architecture: {architecture}")

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
        self.input_dim = input_dim
        self.architecture = architecture
        self.use_edge_metric_weights = architecture.startswith("edge_")
        self.edge_alpha_temperature = max(float(edge_alpha_temperature), 1e-3)

        # 全局相关性融合权重 beta -> alpha = softmax(beta)。
        self.beta = nn.Parameter(torch.zeros(self.n_metrics, dtype=torch.float32))
        # 边级相关性融合权重。形状为(target/source/metric)，第i行聚合第j列源节点。
        self.edge_beta = nn.Parameter(
            torch.zeros(self.n_nodes, self.n_nodes, self.n_metrics, dtype=torch.float32)
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.conf_embedding = nn.Parameter(
            torch.randn(self.n_confidential, hidden_dim) * 0.01
        )

        self.self_layers = nn.ModuleList()
        self.neigh_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.att_target_layers = nn.ModuleList()
        self.att_source_layers = nn.ModuleList()
        self.att_prior_scales = nn.ParameterList()

        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(n_nodes))

            if architecture in {"gat", "edge_gat", "edge_conf_gat"}:
                self.att_target_layers.append(nn.Linear(hidden_dim, 1, bias=False))
                self.att_source_layers.append(nn.Linear(hidden_dim, 1, bias=False))
                self.att_prior_scales.append(nn.Parameter(torch.tensor(1.0)))

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.att_activation = nn.LeakyReLU(0.2)
        self.last_attention: torch.Tensor | None = None

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_alpha(self) -> torch.Tensor:
        """返回相关性融合权重；edge模式下返回所有有效边的平均alpha。"""
        if self.use_edge_metric_weights:
            edge_alpha = self.get_edge_alpha()
            valid = (self.edge_mask > 0) & ~torch.eye(
                self.n_nodes, dtype=torch.bool, device=self.edge_mask.device
            )
            if valid.any():
                return edge_alpha[valid].mean(dim=0)
        return torch.softmax(self.beta, dim=0)

    def get_edge_alpha(self) -> torch.Tensor:
        """返回每条边自己的相关性指标softmax权重。"""
        return torch.softmax(self.edge_beta / self.edge_alpha_temperature, dim=-1)

    def compute_adjacency(self, row_normalize: bool = True) -> torch.Tensor:
        """根据当前alpha计算邻接先验矩阵。edge模式下每条边使用自己的alpha_ij。"""
        if self.use_edge_metric_weights:
            edge_alpha = self.get_edge_alpha()
            A_raw = torch.sum(self.metric_tensor * edge_alpha, dim=-1)
        else:
            alpha = self.get_alpha()
            A_raw = torch.einsum("ijk,k->ij", self.metric_tensor, alpha)
        A = A_raw * self.edge_mask
        A = A * (1.0 - torch.eye(self.n_nodes, device=A.device))
        if not row_normalize:
            return A
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return A / row_sum

    def _initial_hidden(self, node_values: torch.Tensor) -> torch.Tensor:
        if node_values.dim() == 2:
            node_values = node_values.unsqueeze(-1)

        B = node_values.shape[0]
        h_proj = self.input_proj(node_values)

        conf_emb = self.conf_embedding.unsqueeze(0).expand(B, -1, -1)
        idx = self.conf_indices.view(1, -1, 1).expand(B, -1, self.hidden_dim)
        return h_proj.scatter(1, idx, conf_emb)

    def _gat_messages(
        self,
        h: torch.Tensor,
        adjacency_prior: torch.Tensor,
        target_layer: nn.Linear,
        source_layer: nn.Linear,
        prior_scale: torch.Tensor,
    ) -> torch.Tensor:
        """在固定边掩码内学习邻居注意力并聚合消息。"""
        target_score = target_layer(h)              # (B, N, 1)
        source_score = source_layer(h).transpose(1, 2)  # (B, 1, N)

        # 方向约定：第i行聚合来自第j列的源节点。
        logits = target_score + source_score

        prior = adjacency_prior.clamp(min=1e-12)
        scale = F.softplus(prior_scale)
        logits = logits + scale * torch.log(prior).unsqueeze(0)
        logits = self.att_activation(logits)

        valid_edges = adjacency_prior > 0
        logits = logits.masked_fill(~valid_edges.unsqueeze(0), -1e9)

        attention = torch.softmax(logits, dim=-1)
        attention = attention * valid_edges.unsqueeze(0).float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attention = self.attention_dropout(attention)

        self.last_attention = attention.detach()
        return torch.bmm(attention, h)

    def forward(self, node_values: torch.Tensor) -> torch.Tensor:
        """
        参数:
            node_values: (B, N, input_dim) 或 (B, N)，Confidential位置已置零

        返回:
            (B, C) 对Confidential节点的预测值
        """
        h = self._initial_hidden(node_values)
        uses_gat = self.architecture in {"gat", "edge_gat", "edge_conf_gat"}
        A_prior = self.compute_adjacency(row_normalize=False)
        A_norm = self.compute_adjacency(row_normalize=True)

        for layer_idx, (self_layer, neigh_layer, bn) in enumerate(
            zip(self.self_layers, self.neigh_layers, self.batch_norms)
        ):
            if self.architecture in {"gat", "edge_gat"}:
                messages = self._gat_messages(
                    h,
                    A_prior,
                    self.att_target_layers[layer_idx],
                    self.att_source_layers[layer_idx],
                    self.att_prior_scales[layer_idx],
                )
            elif self.architecture == "edge_conf_gat":
                messages = torch.matmul(A_norm, h)
                gat_messages = self._gat_messages(
                    h,
                    A_prior,
                    self.att_target_layers[layer_idx],
                    self.att_source_layers[layer_idx],
                    self.att_prior_scales[layer_idx],
                )
                idx = self.conf_indices.view(1, -1, 1).expand(
                    h.shape[0], -1, self.hidden_dim
                )
                conf_messages = gat_messages[:, self.conf_indices, :]
                messages = messages.scatter(1, idx, conf_messages)
            else:
                messages = torch.matmul(A_norm, h)

            h_new = self_layer(h) + neigh_layer(messages)
            h_new = bn(h_new)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            h = h_new + h

        h_conf = h[:, self.conf_indices, :]
        return self.output_head(h_conf).squeeze(-1)

    def get_mean_attention(self, node_values: torch.Tensor) -> np.ndarray | None:
        """返回GAT最后一层在给定样本上的平均注意力矩阵。"""
        if self.architecture not in {"gat", "edge_gat", "edge_conf_gat"}:
            return None

        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self(node_values)
            if self.last_attention is None:
                return None
            attention = self.last_attention.mean(dim=0).cpu().numpy()
        if was_training:
            self.train()
        return attention


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
