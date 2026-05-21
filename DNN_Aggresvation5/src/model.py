"""
端到端推断驱动的GNN模型（DNN_Aggresvation5）。

子项目5实现 Route B：target-conditioned GAT。

核心变化：
- `gcn`: 子项目3/4基线，所有节点使用全局邻接矩阵均匀聚合。
- `gat`: 子项目4普通GAT，所有节点在固定边掩码内学习attention。
- `target_gat`: 子项目5方案B。General节点仍用稳定GCN更新；Confidential节点
  使用目标字段embedding条件化的attention读取邻居，让不同目标拥有不同边含义。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InferenceDrivenGNN(nn.Module):
    """推断驱动GNN，支持 `gcn`、`gat`、`target_gat`。"""

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
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if architecture not in {"gcn", "gat", "target_gat"}:
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
        self.attention_temperature = max(float(attention_temperature), 1e-3)

        self.beta = nn.Parameter(torch.zeros(self.n_metrics, dtype=torch.float32))

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
        self.att_condition_layers = nn.ModuleList()
        self.att_prior_scales = nn.ParameterList()

        needs_attention = architecture in {"gat", "target_gat"}
        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(n_nodes))

            if needs_attention:
                self.att_target_layers.append(nn.Linear(hidden_dim, 1, bias=False))
                self.att_source_layers.append(nn.Linear(hidden_dim, 1, bias=False))
                self.att_condition_layers.append(nn.Linear(hidden_dim, 1, bias=False))
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
        return torch.softmax(self.beta, dim=0)

    def compute_adjacency(self, row_normalize: bool = True) -> torch.Tensor:
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
        target_score = target_layer(h)
        source_score = source_layer(h).transpose(1, 2)
        logits = target_score + source_score
        prior = adjacency_prior.clamp(min=1e-12)
        logits = logits + F.softplus(prior_scale) * torch.log(prior).unsqueeze(0)
        logits = self.att_activation(logits) / self.attention_temperature

        valid_edges = adjacency_prior > 0
        logits = logits.masked_fill(~valid_edges.unsqueeze(0), -1e9)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * valid_edges.unsqueeze(0).float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attention = self.attention_dropout(attention)
        self.last_attention = attention.detach()
        return torch.bmm(attention, h)

    def _target_conditioned_messages(
        self,
        h: torch.Tensor,
        adjacency_prior: torch.Tensor,
        target_layer: nn.Linear,
        source_layer: nn.Linear,
        condition_layer: nn.Linear,
        prior_scale: torch.Tensor,
    ) -> torch.Tensor:
        """只为Confidential节点计算target-conditioned attention消息。"""
        B = h.shape[0]
        h_conf = h[:, self.conf_indices, :]               # (B, C, H)
        source_score = source_layer(h).transpose(1, 2)    # (B, 1, N)
        target_score = target_layer(h_conf)               # (B, C, 1)

        # 关键：目标字段身份embedding直接参与attention logits。
        cond_emb = self.conf_embedding.unsqueeze(0).expand(B, -1, -1)
        condition_score = condition_layer(cond_emb)       # (B, C, 1)
        logits = target_score + condition_score + source_score

        prior_raw = adjacency_prior[self.conf_indices]  # (C, N)
        valid_edges = prior_raw > 0
        prior = prior_raw.clamp(min=1e-12)
        logits = logits + F.softplus(prior_scale) * torch.log(prior).unsqueeze(0)
        logits = self.att_activation(logits) / self.attention_temperature

        logits = logits.masked_fill(~valid_edges.unsqueeze(0), -1e9)
        attention_conf = torch.softmax(logits, dim=-1)
        attention_conf = attention_conf * valid_edges.unsqueeze(0).float()
        attention_conf = attention_conf / attention_conf.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attention_conf = self.attention_dropout(attention_conf)

        conf_messages = torch.bmm(attention_conf, h)      # (B, C, H)

        # 保存成完整(N,N)注意力矩阵，便于沿用诊断代码。
        attention_full = torch.zeros(
            B, self.n_nodes, self.n_nodes, dtype=h.dtype, device=h.device
        )
        attention_full[:, self.conf_indices, :] = attention_conf
        self.last_attention = attention_full.detach()
        return conf_messages

    def forward(self, node_values: torch.Tensor) -> torch.Tensor:
        h = self._initial_hidden(node_values)
        A_norm = self.compute_adjacency(row_normalize=True)
        A_prior = self.compute_adjacency(row_normalize=False)

        for layer_idx, (self_layer, neigh_layer, bn) in enumerate(
            zip(self.self_layers, self.neigh_layers, self.batch_norms)
        ):
            if self.architecture == "gat":
                messages = self._gat_messages(
                    h,
                    A_prior,
                    self.att_target_layers[layer_idx],
                    self.att_source_layers[layer_idx],
                    self.att_prior_scales[layer_idx],
                )
            elif self.architecture == "target_gat":
                # General节点保持GCN稳定传播；Confidential节点用目标条件化attention替换消息。
                messages = torch.matmul(A_norm, h)
                conf_messages = self._target_conditioned_messages(
                    h,
                    A_prior,
                    self.att_target_layers[layer_idx],
                    self.att_source_layers[layer_idx],
                    self.att_condition_layers[layer_idx],
                    self.att_prior_scales[layer_idx],
                )
                idx = self.conf_indices.view(1, -1, 1).expand(
                    h.shape[0], -1, self.hidden_dim
                )
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
        if self.architecture not in {"gat", "target_gat"}:
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
