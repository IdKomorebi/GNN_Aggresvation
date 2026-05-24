"""
端到端推断驱动的混合GNN模型（DNN_Aggresvation27）。

子项目14沿用DNN13结构，用于分析少数低R2 target的调参敏感性：
- General-General: 使用一组全局alpha_G构造固定相关性边权，并做加权GCN聚合
- Confidential-Source: 每条有向入边(c<-j)学习一组alpha_{c,j}
- 动态项只使用target bilinear score，不再额外加入W_r r_{c,j}
- gate只使用q、k和log prior，避免原始关系向量绕过alpha_c
- 默认使用target-specific output heads
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class InferenceDrivenGNN(nn.Module):
    """DNN27混合聚合模型，Confidential-Source支持多头注意力。"""

    VALID_ARCHITECTURES = {
        "hybrid_bilinear_gated",
        "hybrid_bilinear_no_gate",
        "hybrid_bilinear_prior_only",
    }

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
        architecture: str = "hybrid_bilinear_gated",
        attention_dropout: float = 0.05,
        attention_dim: int | None = None,
        attention_heads: int = 1,
        attention_temperature: float = 1.0,
        edge_alpha_temperature: float = 1.0,
        alpha_init_std: float = 0.0,
        prior_scale_init: float = 1.0,
        gate_bias_init: float = -1.0,
        prior_log_eps: float = 1e-4,
        target_specific_heads: bool = True,
        input_encoder: str = "linear",
        input_encoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if architecture not in self.VALID_ARCHITECTURES:
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
        self.attention_dim = int(attention_dim or hidden_dim)
        self.attention_heads = max(int(attention_heads), 1)
        self.attention_temperature = max(float(attention_temperature), 1e-3)
        self.edge_alpha_temperature = max(float(edge_alpha_temperature), 1e-3)
        self.prior_log_eps = max(float(prior_log_eps), 1e-12)
        self.target_specific_heads = bool(target_specific_heads)
        self.input_encoder = str(input_encoder)

        # General-General共用一组相关性融合权重。
        self.beta_general = nn.Parameter(torch.zeros(self.n_metrics, dtype=torch.float32))
        # 每个Confidential target-source有向入边一组相关性融合权重。
        self.beta_confidential = nn.Parameter(
            torch.zeros(self.n_confidential, self.n_nodes, self.n_metrics, dtype=torch.float32)
        )
        if alpha_init_std > 0:
            nn.init.normal_(self.beta_general, mean=0.0, std=float(alpha_init_std))
            nn.init.normal_(self.beta_confidential, mean=0.0, std=float(alpha_init_std))

        if self.input_encoder == "mlp":
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(input_encoder_dropout)),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif self.input_encoder == "linear":
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown input_encoder: {self.input_encoder}")
        self.conf_embedding = nn.Parameter(
            torch.randn(self.n_confidential, hidden_dim) * 0.01
        )
        self.target_identity = nn.Parameter(
            torch.randn(self.n_confidential, hidden_dim) * 0.01
        )

        self.self_layers = nn.ModuleList()
        self.general_neigh_layers = nn.ModuleList()
        self.conf_neigh_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.query_hidden_layers = nn.ModuleList()
        self.query_identity_layers = nn.ModuleList()
        self.key_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        self.gate_q_layers = nn.ModuleList()
        self.gate_k_layers = nn.ModuleList()
        self.gate_p_layers = nn.ModuleList()
        self.attention_output_layers = nn.ModuleList()
        self.prior_scales = nn.ParameterList()

        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.general_neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.conf_neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(n_nodes))

            self.query_hidden_layers.append(
                nn.Linear(hidden_dim, self.attention_heads * self.attention_dim, bias=False)
            )
            self.query_identity_layers.append(
                nn.Linear(hidden_dim, self.attention_heads * self.attention_dim, bias=False)
            )
            self.key_layers.append(
                nn.Linear(hidden_dim, self.attention_heads * self.attention_dim, bias=False)
            )
            self.value_layers.append(
                nn.Linear(hidden_dim, self.attention_heads * hidden_dim, bias=False)
            )
            self.attention_output_layers.append(
                nn.Linear(self.attention_heads * hidden_dim, hidden_dim, bias=False)
            )
            # 等价于W_lambda[q || k || log(p)] + b，但避免构造(B,C,N,2H+1)大张量。
            self.gate_q_layers.append(nn.Linear(self.attention_dim, 1, bias=False))
            self.gate_k_layers.append(nn.Linear(self.attention_dim, 1, bias=False))
            self.gate_p_layers.append(nn.Linear(1, 1, bias=True))
            nn.init.constant_(self.gate_p_layers[-1].bias, float(gate_bias_init))
            self.prior_scales.append(nn.Parameter(torch.tensor(float(prior_scale_init))))

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.last_attention: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_prior: torch.Tensor | None = None

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.target_output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(self.n_confidential)
            ]
        )

    def get_alpha(self) -> torch.Tensor:
        """兼容训练日志：返回General-General的alpha_G。"""
        return self.get_alpha_general()

    def get_alpha_general(self) -> torch.Tensor:
        return torch.softmax(self.beta_general / self.edge_alpha_temperature, dim=0)

    def get_alpha_confidential(self) -> torch.Tensor:
        return torch.softmax(self.beta_confidential / self.edge_alpha_temperature, dim=-1)

    def _initial_hidden(self, node_values: torch.Tensor) -> torch.Tensor:
        if node_values.dim() == 2:
            node_values = node_values.unsqueeze(-1)

        B = node_values.shape[0]
        h_proj = self.input_proj(node_values)

        conf_emb = self.conf_embedding.unsqueeze(0).expand(B, -1, -1)
        idx = self.conf_indices.view(1, -1, 1).expand(B, -1, self.hidden_dim)
        return h_proj.scatter(1, idx, conf_emb)

    def compute_general_adjacency(self, row_normalize: bool = True) -> torch.Tensor:
        """General-General固定相关性边权。"""
        alpha_g = self.get_alpha_general()
        A_raw = torch.einsum("ijk,k->ij", self.metric_tensor, alpha_g)
        A = A_raw * self.edge_mask
        A = A * (1.0 - torch.eye(self.n_nodes, device=A.device))

        gg_mask = torch.zeros_like(A)
        gg_mask[: self.n_general, : self.n_general] = 1.0
        A = A * gg_mask
        if not row_normalize:
            return A
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return A / row_sum

    def compute_confidential_prior(self) -> torch.Tensor:
        """返回形状(C,N)的有向边关系先验p_{c,j}。"""
        alpha_c = self.get_alpha_confidential()
        rel = self.metric_tensor.index_select(0, self.conf_indices)
        prior = torch.sum(rel * alpha_c, dim=-1)
        mask = self.edge_mask.index_select(0, self.conf_indices)
        prior = prior * mask
        prior = prior.clone()
        for conf_pos, node_idx in enumerate(self.conf_indices.tolist()):
            prior[conf_pos, node_idx] = 0.0
        return prior

    def _confidential_messages(
        self,
        h: torch.Tensor,
        layer_idx: int,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        B = h.shape[0]
        h_conf = h.index_select(1, self.conf_indices)
        target_id = self.target_identity.unsqueeze(0).expand(B, -1, -1)

        q = (
            self.query_hidden_layers[layer_idx](h_conf)
            + self.query_identity_layers[layer_idx](target_id)
        )
        k = self.key_layers[layer_idx](h)
        v = self.value_layers[layer_idx](h)
        q = q.view(B, self.n_confidential, self.attention_heads, self.attention_dim)
        k = k.view(B, self.n_nodes, self.attention_heads, self.attention_dim)
        v = v.view(B, self.n_nodes, self.attention_heads, self.hidden_dim)

        dynamic_score = torch.einsum("bchd,bnhd->bchn", q, k) / math.sqrt(
            self.attention_dim
        )
        prior_log = torch.log(prior.clamp(min=self.prior_log_eps))
        prior_score = prior_log.unsqueeze(0).unsqueeze(2).expand(
            B, -1, self.attention_heads, -1
        )
        valid = prior > 0

        if self.architecture == "hybrid_bilinear_prior_only":
            score = prior_score
            gate = torch.zeros_like(score)
        elif self.architecture == "hybrid_bilinear_no_gate":
            scale = torch.nn.functional.softplus(self.prior_scales[layer_idx])
            score = dynamic_score + scale * prior_score
            gate = torch.ones_like(score)
        else:
            q_gate = self.gate_q_layers[layer_idx](q).squeeze(-1).unsqueeze(-1)
            k_gate = (
                self.gate_k_layers[layer_idx](k)
                .squeeze(-1)
                .permute(0, 2, 1)
                .unsqueeze(1)
            )
            gate_logits = (
                q_gate
                + k_gate
                + self.gate_p_layers[layer_idx](prior_score.unsqueeze(-1)).squeeze(-1)
            )
            gate = torch.sigmoid(gate_logits)
            scale = torch.nn.functional.softplus(self.prior_scales[layer_idx])
            score = gate * dynamic_score + (1.0 - gate) * scale * prior_score

        score = score / self.attention_temperature
        score = score.masked_fill(~valid.unsqueeze(0).unsqueeze(2), -1e9)
        attention = torch.softmax(score, dim=-1)
        attention = attention * valid.unsqueeze(0).unsqueeze(2).float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attention = self.attention_dropout(attention)
        attention_mean = attention.mean(dim=2)

        full_attention = torch.zeros(
            B, self.n_nodes, self.n_nodes, dtype=attention.dtype, device=attention.device
        )
        full_attention[:, self.conf_indices, :] = attention_mean
        self.last_attention = full_attention.detach()
        self.last_gate = gate.mean(dim=2).detach()
        self.last_prior = prior.detach()

        messages = torch.einsum("bchn,bnhd->bchd", attention, v)
        messages = messages.reshape(B, self.n_confidential, self.attention_heads * self.hidden_dim)
        return self.attention_output_layers[layer_idx](messages)

    def forward(self, node_values: torch.Tensor) -> torch.Tensor:
        h = self._initial_hidden(node_values)
        A_general = self.compute_general_adjacency(row_normalize=True)
        conf_prior = self.compute_confidential_prior()

        for layer_idx, (self_layer, gen_layer, conf_layer, bn) in enumerate(
            zip(
                self.self_layers,
                self.general_neigh_layers,
                self.conf_neigh_layers,
                self.batch_norms,
            )
        ):
            messages = torch.zeros_like(h)

            general_messages = torch.matmul(A_general[: self.n_general, : self.n_general], h[:, : self.n_general, :])
            messages[:, : self.n_general, :] = general_messages

            conf_messages = self._confidential_messages(h, layer_idx, conf_prior)
            messages[:, self.conf_indices, :] = conf_messages

            general_part = gen_layer(messages)
            conf_part = conf_layer(messages)
            transformed_messages = general_part
            transformed_messages = transformed_messages.clone()
            transformed_messages[:, self.conf_indices, :] = conf_part[:, self.conf_indices, :]

            h_new = self_layer(h) + transformed_messages
            h_new = bn(h_new)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            h = h_new + h

        h_conf = h.index_select(1, self.conf_indices)
        if self.target_specific_heads:
            outputs = [
                head(h_conf[:, target_pos, :])
                for target_pos, head in enumerate(self.target_output_heads)
            ]
            return torch.cat(outputs, dim=1)
        return self.output_head(h_conf).squeeze(-1)

    def get_mean_attention(self, node_values: torch.Tensor) -> np.ndarray | None:
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

    def get_mean_gate(self, node_values: torch.Tensor) -> np.ndarray | None:
        if self.architecture != "hybrid_bilinear_gated":
            return None
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self(node_values)
            if self.last_gate is None:
                return None
            gate = self.last_gate.mean(dim=0).cpu().numpy()
        if was_training:
            self.train()
        return gate

    def get_confidential_prior(self) -> np.ndarray:
        return self.compute_confidential_prior().detach().cpu().numpy()


def build_edge_mask(
    metric_tensor: np.ndarray,
    top_k: int = 10,
    threshold: float = 0.08,
    symmetrize: bool = True,
    selection: str = "threshold_or_topk",
    min_k: int | None = None,
    max_k: int | None = None,
) -> np.ndarray:
    """
    基于平均相关性构建二值边掩码。

    threshold_or_topk:
      对每个节点，保留其平均相关性最高的top_k个邻居，
      同时保留所有平均相关性超过threshold的边。

    adaptive_threshold_topk:
      对每个节点，先统计超过threshold的邻居数；若少于min_k则补足到min_k，
      若多于max_k则截断到max_k。这样密集节点可以超过固定top_k，稀疏节点
      也不会被强行补到过多邻居。
    """
    avg_corr = metric_tensor.mean(axis=2)
    np.fill_diagonal(avg_corr, 0.0)

    n = avg_corr.shape[0]
    selection = str(selection)

    if selection == "adaptive_threshold_topk":
        min_keep = top_k if min_k is None else int(min_k)
        max_keep = top_k if max_k is None else int(max_k)
        if min_keep < 0 or max_keep < 1 or min_keep > max_keep:
            raise ValueError(
                f"Invalid adaptive edge settings: min_k={min_keep}, max_k={max_keep}"
            )

        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            order = [j for j in np.argsort(avg_corr[i])[::-1] if j != i]
            above_threshold = [j for j in order if avg_corr[i, j] >= threshold]
            keep_count = min(max(len(above_threshold), min_keep), max_keep)
            for j in order[:keep_count]:
                mask[i, j] = True
    else:
        mask = avg_corr >= threshold

        for i in range(n):
            order = np.argsort(avg_corr[i])[::-1]
            neighbors = [j for j in order if j != i][:top_k]
            for j in neighbors:
                mask[i, j] = True

    if symmetrize:
        mask = mask | mask.T
        if selection == "adaptive_threshold_topk" and max_k is not None:
            # Symmetrization can add many reciprocal edges. Prune each row again so
            # every target keeps a comparable but still adaptive number of sources.
            pruned = np.zeros_like(mask, dtype=bool)
            min_keep = top_k if min_k is None else int(min_k)
            max_keep = int(max_k)
            for i in range(n):
                current = [j for j in np.argsort(avg_corr[i])[::-1] if j != i and mask[i, j]]
                if len(current) < min_keep:
                    order = [j for j in np.argsort(avg_corr[i])[::-1] if j != i]
                    current = order[:min_keep]
                else:
                    current = current[:max_keep]
                for j in current:
                    pruned[i, j] = True
            mask = pruned

    return mask.astype(np.float32)
