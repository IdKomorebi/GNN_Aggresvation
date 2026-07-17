"""子集条件泄露 oracle（DNN63，主线 Phase 1）。

一个 oracle 用随机子集失活训练：输入"可见 general 字段集合 S"（值 + 可见性 mask），
输出对 12 个 confidential 的预测。训练目标是随机掩码下的期望 MSE，其逐点最优解是
每个子集各自的条件期望 E[Y|X_S]——即一个模型摊销全部 2^44 个子集的攻击者最优响应。
之后查询 v̂(S)：把测试集按 S 掩码后前向、实测 12-conf 平均 R²，O(1) 一次前向。

两个 oracle 同一掩码分布、同一预算：
- MLPOracle：输入 [x⊙m, m]，无图先验（强 baseline）；
- GNNOracle：相关图先验 + 可见性感知注意力——被遮蔽字段既置 0 又从聚合的 softmax 中
  剔除（只在可见邻居上重归一化），这是"沿相关边替补被遮蔽字段"的结构化实现。
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLPOracle(nn.Module):
    """无图先验的子集条件 oracle。输入 [x⊙m, m]（2·nG 维）。"""

    def __init__(self, n_general: int, n_confidential: int,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_general, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_confidential),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x * m, m], dim=1))


class GNNOracle(nn.Module):
    """相关图先验 + 可见性感知注意力的子集条件 oracle。

    A_gg  : (nG,nG) general-general 相关性先验（非负，已按 edge_mask 稀疏化、对角 0）；
    prior_cg: (nC,nG) confidential<-general 相关性先验（非负，edge_mask 稀疏化）。
    可见性：被遮蔽 general 节点的值置 0，且在所有聚合的 softmax/行归一里被剔除，
    使 confidential 的信息只能来自可见邻居——遮蔽某字段时注意力自动滑向可见的冗余字段。
    """

    def __init__(self, A_gg: np.ndarray, prior_cg: np.ndarray, n_confidential: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.1,
                 prior_log_eps: float = 1e-4):
        super().__init__()
        self.register_buffer("A_gg", torch.as_tensor(A_gg, dtype=torch.float32))
        self.register_buffer("prior_cg", torch.as_tensor(prior_cg, dtype=torch.float32))
        self.nG = A_gg.shape[0]
        self.nC = n_confidential
        self.H = hidden
        self.prior_log_eps = prior_log_eps

        self.enc = nn.Sequential(nn.Linear(2, hidden), nn.ReLU())  # 每节点 [值, 可见位]
        self.gg_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.gg_neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.nG) for _ in range(n_layers)])

        self.conf_emb = nn.Parameter(torch.randn(n_confidential, hidden) * 0.01)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.val = nn.Linear(hidden, hidden, bias=False)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                          nn.Linear(hidden // 2, 1))
            for _ in range(n_confidential)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = torch.stack([x * m, m], dim=-1)          # (B,nG,2)
        h = self.enc(feat)                              # (B,nG,H)

        # ---- general-general 传播：只聚合可见邻居（按可见性重归一化）----
        A = self.A_gg.unsqueeze(0)                      # (1,nG,nG)：i<-j 相关先验
        for li in range(len(self.gg_self)):
            w = A * m.unsqueeze(1)                      # 邻居 j 不可见则权重 0
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
            msg = torch.bmm(w, h)                       # (B,nG,H)
            h_new = self.gg_self[li](h) + self.gg_neigh[li](msg)
            h_new = self.bn[li](h_new)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new

        # ---- general -> confidential 注意力：可见性感知 softmax ----
        q = self.q(self.conf_emb).unsqueeze(0).expand(B, -1, -1)   # (B,nC,H)
        k = self.k(h)                                              # (B,nG,H)
        v = self.val(h)                                            # (B,nG,H)
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.H)  # (B,nC,nG)
        prior_log = torch.log(self.prior_cg.clamp(min=self.prior_log_eps)).unsqueeze(0)
        score = score + F.softplus(self.prior_scale) * prior_log
        valid = (m.unsqueeze(1) > 0) & (self.prior_cg.unsqueeze(0) > 0)  # (B,nC,nG)
        score = score.masked_fill(~valid, -1e9)
        attn = torch.softmax(score, dim=-1)
        attn = attn * valid.float()
        attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-12)  # 无可见邻居→全 0→输出偏置
        attn = self.dropout(attn)
        ctx = torch.bmm(attn, v)                                   # (B,nC,H)
        outs = [self.heads[c](ctx[:, c, :]) for c in range(self.nC)]
        return torch.cat(outs, dim=1)                             # (B,nC)


def build_priors(metric_tensor: np.ndarray, edge_mask: np.ndarray, n_general: int):
    """从相关性张量与 edge_mask 构造 A_gg (nG,nG) 与 prior_cg (nC,nG)，均非负。"""
    corr = metric_tensor.mean(axis=2)                # (N,N) 多指标平均相关性
    np.fill_diagonal(corr, 0.0)
    corr = np.clip(corr, 0.0, None) * edge_mask       # 稀疏化 + 非负
    A_gg = corr[:n_general, :n_general].astype(np.float32)
    prior_cg = corr[n_general:, :n_general].astype(np.float32)
    return A_gg, prior_cg


def sample_mask(batch: int, n_general: int, rng: np.random.RandomState) -> np.ndarray:
    """逐样本掩码：先抽保留数 k~U{1..nG}，再均匀抽 k 个可见字段（覆盖各尺寸带均衡）。"""
    m = np.zeros((batch, n_general), dtype=np.float32)
    ks = rng.randint(1, n_general + 1, size=batch)
    for i, k in enumerate(ks):
        idx = rng.choice(n_general, size=int(k), replace=False)
        m[i, idx] = 1.0
    return m
