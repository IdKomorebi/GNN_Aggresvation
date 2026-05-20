from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def make_mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int = 1, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        h = int(hidden)
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CorrelationGateDNN(nn.Module):
    """DNN whose feature gates are generated from correlation metric vectors."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        correlation_vectors: np.ndarray,
    ) -> None:
        super().__init__()
        corr = torch.as_tensor(correlation_vectors, dtype=torch.float32)
        self.register_buffer("correlation_vectors", corr)
        self.beta = nn.Parameter(torch.zeros(corr.shape[1], dtype=torch.float32))
        self.gate_scale_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gate_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.net = make_mlp(in_dim, hidden_dims)

    def metric_weights(self) -> torch.Tensor:
        return torch.softmax(self.beta, dim=0)

    def get_gates(self) -> torch.Tensor:
        alpha = self.metric_weights()
        z = self.correlation_vectors @ alpha
        scale = torch.nn.functional.softplus(self.gate_scale_raw)
        return torch.sigmoid(scale * z + self.gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * self.get_gates())


class SingleFeatureDNN(nn.Module):
    def __init__(self, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        self.net = make_mlp(1, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedGraphRiskGNN(nn.Module):
    """Risk GNN using a fixed row-normalized adjacency matrix."""

    def __init__(
        self,
        in_dim: int,
        adjacency: np.ndarray,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adj)
        self.input = nn.Linear(in_dim, hidden_dim)
        self.self_layers = nn.ModuleList()
        self.neighbor_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.neighbor_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(node_features))
        for self_layer, neighbor_layer in zip(self.self_layers, self.neighbor_layers):
            messages = self.adjacency @ h
            h = torch.relu(self_layer(h) + neighbor_layer(messages))
            h = self.dropout(h)
        return torch.sigmoid(self.output(h)).squeeze(-1)

