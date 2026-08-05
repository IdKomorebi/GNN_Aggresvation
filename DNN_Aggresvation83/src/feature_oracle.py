"""带显式二阶算术字典的一次前向 oracle。"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import nn


class Poly2Oracle(nn.Module):
    """输入可见 raw、mask、平方和两两乘积；交互项只在父字段都可见时激活。"""

    def __init__(
        self,
        n_general: int,
        n_confidential: int,
        square_mean: np.ndarray,
        square_std: np.ndarray,
        product_mean: np.ndarray,
        product_std: np.ndarray,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        pairs = list(combinations(range(n_general), 2))
        self.register_buffer(
            "pair_left", torch.as_tensor([pair[0] for pair in pairs])
        )
        self.register_buffer(
            "pair_right", torch.as_tensor([pair[1] for pair in pairs])
        )
        self.register_buffer(
            "square_mean", torch.as_tensor(square_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "square_std", torch.as_tensor(square_std, dtype=torch.float32)
        )
        self.register_buffer(
            "product_mean", torch.as_tensor(product_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "product_std", torch.as_tensor(product_std, dtype=torch.float32)
        )
        input_dim = 3 * n_general + len(pairs)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_confidential),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        visible = x * m
        square = ((x.square() - self.square_mean) / self.square_std) * m
        pair_mask = m[:, self.pair_left] * m[:, self.pair_right]
        product = (
            (
                x[:, self.pair_left] * x[:, self.pair_right]
                - self.product_mean
            )
            / self.product_std
        ) * pair_mask
        return self.net(torch.cat([visible, m, square, product], dim=1))


def poly2_stats(x_train: np.ndarray) -> dict[str, np.ndarray]:
    n_general = x_train.shape[1]
    pairs = list(combinations(range(n_general), 2))
    square = x_train**2
    product = np.column_stack(
        [x_train[:, left] * x_train[:, right] for left, right in pairs]
    )
    square_std = square.std(axis=0)
    product_std = product.std(axis=0)
    square_std[square_std < 1e-6] = 1.0
    product_std[product_std < 1e-6] = 1.0
    return {
        "square_mean": square.mean(axis=0).astype(np.float32),
        "square_std": square_std.astype(np.float32),
        "product_mean": product.mean(axis=0).astype(np.float32),
        "product_std": product_std.astype(np.float32),
    }
