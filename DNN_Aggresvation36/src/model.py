"""
端到端推断驱动的混合GNN模型（DNN_Aggresvation36）。

在 DNN35 的基础上新增 **value 的低秩修正（LoRA）**：DNN35 的 FiLM 用对角
变换 diag(γ_c)·W_V，只能逐维缩放、不能跨维组合，被验证无效。LoRA 改用
    W_V^(c) = W_V + B_c·A_c   （ΔW_c=B_c·A_c 满矩阵，秩≤r，可跨维组合）
由 value_lora_rank=r 控制（0=关闭）。per-(layer,head) 一对 A:(H,C,r,d)、
B:(H,C,d,r)，B 零初始化 → 起步 ΔW=0（identity）。实现用恒等式
    m_c = (W_V + B_c·A_c)·h̄_c,  h̄_c = Σ_j α(c,j)·h_j
（h̄_c 为注意力加权的源隐藏态），避免把张量撑大到 (B,C,S,d)。

—— 以下沿用 DNN35 ——

在 DNN34（双重注意力，dynamic head attention 最优）基础上，新增
**target-specific value（FiLM 调制）**，攻击「共享 value/菜单」这个真瓶颈：

当前注意力里 value v_j = W_V·h_j 只依赖 source j、与 target c 无关，所有
confidential 字段共用同一组 {v_j}，差异只在注意力权重 α(c,j)。本项目让
每个 target c 用一组逐维缩放/平移 (γ_c, β_c) 调制 value：
    v(c,j) = γ_c ⊙ v_j + β_c
由于注意力权重与 head 权重都是凸组合（和为 1），γ_c/β_c 可提到求和外，
等价于在「汇聚后的消息」上做一次 FiLM：
    msg_c ← (1 + Γ_c) ⊙ msg_c + B_c
即用对角变换 diag(γ_c)·W_V 廉价逼近 single 才有的 per-target value 投影，
且仍是一次前向同时算完所有 target（不退化成 single 的成本）。

value_film 取值：
- "none"   ：不调制（= DNN34 dynamic_8head 行为）
- "static" ：Γ_c, B_c 为 per-(layer,target) 自由参数（查找表，对所有样本相同）
- "dynamic"：Γ_c, B_c 由 target 身份 e_c + 当前 conf 隐藏态 h_c 生成，随样本变化

—— 以下沿用 DNN34 ——
在 DNN32 bipartite 主干基础上引入真正的双重注意力机制：
- 每个 attention head 拥有独立的、完整维度的 q/k/v/gate 投影矩阵
  （不切维度，每个 head 在完整 attention_dim 上做完整 attention 计算）
- 第二级 head 聚合（head_aggregation 控制两种机制）：
  * "static_gate"（GLM 方案）：head_weight = softmax(W·target_identity/τ)，
    权重只取决于 target 身份，与输入数据无关（静态）。
  * "dynamic_attention"（用户思路）：对 head 维度再做一次注意力——
    query 来自 target 身份 + 当前 conf 隐藏态，key/value 来自各 head 的
    实际输出 msg_h，head_weight 随输入内容动态变化。直接针对 DNN33 报告
    自承认的「静态 gate 梯度信号弱、趋向均匀、不分化」弱点。
- 当 n_heads=1 时两种聚合等价，退化为 DNN32 单头行为
- bipartite 开关沿用 DNN29

与 DNN33 的区别：DNN33 切了维度（head_dim=attention_dim/n_heads），导致
单 head 表达能力不足。DNN34 每个 head 用完整 attention_dim，参数量是
n_heads 倍但表达能力不受损。
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class InferenceDrivenGNN(nn.Module):
    """DNN15混合聚合模型。"""

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
        attention_temperature: float = 1.0,
        edge_alpha_temperature: float = 1.0,
        alpha_init_std: float = 0.0,
        prior_scale_init: float = 1.0,
        gate_bias_init: float = -1.0,
        prior_log_eps: float = 1e-4,
        target_specific_heads: bool = True,
        input_encoder: str = "linear",
        input_encoder_dropout: float = 0.0,
        bipartite: bool = False,
        n_heads: int = 1,
        head_gate_temperature: float = 1.0,
        head_gate_bias_init: float = 0.0,
        head_aggregation: str = "static_gate",
        value_film: str = "none",
        value_lora_rank: int = 0,
    ) -> None:
        super().__init__()

        if architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")

        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")

        if head_aggregation not in {"static_gate", "dynamic_attention"}:
            raise ValueError(
                f"Unknown head_aggregation: {head_aggregation}"
            )

        if value_film not in {"none", "static", "dynamic"}:
            raise ValueError(f"Unknown value_film: {value_film}")

        if int(value_lora_rank) < 0:
            raise ValueError(f"value_lora_rank must be >= 0, got {value_lora_rank}")

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
        if bipartite:
            general_idx = torch.arange(n_general, dtype=torch.long)
            self.register_buffer("general_indices", general_idx)
        else:
            self.register_buffer("general_indices", torch.empty(0, dtype=torch.long))

        self.n_nodes = n_nodes
        self.n_general = n_general
        self.n_confidential = len(confidential_indices)
        self.n_metrics = metric_tensor.shape[2]
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.architecture = architecture
        self.attention_dim = int(attention_dim or hidden_dim)
        self.attention_temperature = max(float(attention_temperature), 1e-3)
        self.edge_alpha_temperature = max(float(edge_alpha_temperature), 1e-3)
        self.prior_log_eps = max(float(prior_log_eps), 1e-12)
        self.target_specific_heads = bool(target_specific_heads)
        self.input_encoder = str(input_encoder)
        self.bipartite = bool(bipartite)
        self.n_heads = int(n_heads)
        self.head_gate_temperature = max(float(head_gate_temperature), 1e-3)
        self.head_gate_bias_init = float(head_gate_bias_init)
        self.head_aggregation = str(head_aggregation)
        self.value_film = str(value_film)
        self.value_lora_rank = int(value_lora_rank)

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

        self.query_hidden_layers = nn.ModuleList()   # per layer: ModuleList of n_heads Linear
        self.query_identity_layers = nn.ModuleList()
        self.key_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        self.gate_q_layers = nn.ModuleList()
        self.gate_k_layers = nn.ModuleList()
        self.gate_p_layers = nn.ModuleList()
        self.prior_scales = nn.ParameterList()        # per layer: tensor of n_heads
        # 第二级 head 聚合：static_gate 用 head_gate_layers；
        # dynamic_attention 用 head_attn_{q_id,q_hidden,k}_layers。
        self.head_gate_layers = nn.ModuleList()       # per layer: Linear(hidden_dim, n_heads)
        self.head_attn_q_id_layers = nn.ModuleList()     # query from target identity
        self.head_attn_q_hidden_layers = nn.ModuleList() # query from conf hidden state
        self.head_attn_k_layers = nn.ModuleList()        # key from each head's output

        # target-specific value（FiLM）调制：static 用 film_gamma/film_beta 参数表；
        # dynamic 用 film_{g,b}_{id,h} 从 e_c + h_c 生成 Γ/B。均初始化为 identity。
        self.film_gamma = nn.ParameterList()
        self.film_beta = nn.ParameterList()
        self.film_g_id = nn.ModuleList()
        self.film_g_h = nn.ModuleList()
        self.film_b_id = nn.ModuleList()
        self.film_b_h = nn.ModuleList()

        # target-specific value 的低秩修正（LoRA）：per-(layer,head) 一对
        # A:(H,C,r,d)、B:(H,C,d,r)，得到 per-target 修正 ΔW_c=B_c·A_c（秩≤r，
        # 可跨维组合）。B 零初始化 → 起步 ΔW=0（identity）。
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()

        for _ in range(num_layers):
            self.self_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.general_neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.conf_neigh_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(n_nodes))

            # 每个 head 独立的完整 q/k/v/gate 投影（不切维度）
            self.query_hidden_layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, self.attention_dim, bias=False)
                for _ in range(self.n_heads)
            ]))
            self.query_identity_layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, self.attention_dim, bias=False)
                for _ in range(self.n_heads)
            ]))
            self.key_layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, self.attention_dim, bias=False)
                for _ in range(self.n_heads)
            ]))
            self.value_layers.append(nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim, bias=False)
                for _ in range(self.n_heads)
            ]))
            self.gate_q_layers.append(nn.ModuleList([
                nn.Linear(self.attention_dim, 1, bias=False)
                for _ in range(self.n_heads)
            ]))
            self.gate_k_layers.append(nn.ModuleList([
                nn.Linear(self.attention_dim, 1, bias=False)
                for _ in range(self.n_heads)
            ]))
            self.gate_p_layers.append(nn.ModuleList([
                nn.Linear(1, 1, bias=True)
                for _ in range(self.n_heads)
            ]))
            for gp in self.gate_p_layers[-1]:
                nn.init.constant_(gp.bias, float(gate_bias_init))
            self.prior_scales.append(
                nn.Parameter(torch.full((self.n_heads,), float(prior_scale_init)))
            )

            # 第二级 head 聚合层（按 head_aggregation 二选一构建）
            if self.head_aggregation == "static_gate":
                hg = nn.Linear(hidden_dim, self.n_heads, bias=True)
                nn.init.constant_(hg.bias, float(head_gate_bias_init))
                self.head_gate_layers.append(hg)
            else:  # dynamic_attention
                self.head_attn_q_id_layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                )
                self.head_attn_q_hidden_layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                )
                self.head_attn_k_layers.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                )

            # target-specific value（FiLM）调制层（按 value_film 构建，初始为 identity）
            if self.value_film == "static":
                self.film_gamma.append(
                    nn.Parameter(torch.zeros(self.n_confidential, hidden_dim))
                )
                self.film_beta.append(
                    nn.Parameter(torch.zeros(self.n_confidential, hidden_dim))
                )
            elif self.value_film == "dynamic":
                for mlist in (self.film_g_id, self.film_g_h,
                              self.film_b_id, self.film_b_h):
                    lin = nn.Linear(hidden_dim, hidden_dim, bias=True)
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
                    mlist.append(lin)

            # target-specific value 的低秩修正（LoRA），per-(layer,head)
            if self.value_lora_rank > 0:
                r = self.value_lora_rank
                A = nn.Parameter(
                    torch.empty(self.n_heads, self.n_confidential, r, hidden_dim)
                )
                nn.init.normal_(A, std=1.0 / math.sqrt(hidden_dim))
                B = nn.Parameter(
                    torch.zeros(self.n_heads, self.n_confidential, hidden_dim, r)
                )
                self.lora_A.append(A)
                self.lora_B.append(B)

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
        """返回形状(C,N)的有向边关系先验p_{c,j}。

        当 bipartite=True 时，只保留 General 列，Confidential 列被强制置 0，
        彻底切断 Confidential-Confidential 边。
        """
        alpha_c = self.get_alpha_confidential()
        rel = self.metric_tensor.index_select(0, self.conf_indices)
        prior = torch.sum(rel * alpha_c, dim=-1)
        mask = self.edge_mask.index_select(0, self.conf_indices)
        prior = prior * mask
        prior = prior.clone()
        for conf_pos, node_idx in enumerate(self.conf_indices.tolist()):
            prior[conf_pos, node_idx] = 0.0
        if self.bipartite:
            conf_mask = torch.ones(self.n_nodes, dtype=prior.dtype, device=prior.device)
            conf_mask[self.conf_indices] = 0.0
            prior = prior * conf_mask.unsqueeze(0)
        return prior

    def _confidential_messages(
        self,
        h: torch.Tensor,
        layer_idx: int,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """双重注意力：每个 head 独立完整 attention + target-conditioned head gate。

        每个 head 用完整的 attention_dim 做 q/k/v 投影和 attention 计算，
        然后用 softmax(head_gate(target_id) / τ) 加权合并各 head 的消息。
        n_heads=1 时退化为单头。
        """
        B = h.shape[0]
        h_conf = h.index_select(1, self.conf_indices)
        target_id = self.target_identity.unsqueeze(0).expand(B, -1, -1)

        if self.bipartite:
            h_src = h.index_select(1, self.general_indices)
        else:
            h_src = h

        if self.bipartite:
            prior = prior.index_select(1, self.general_indices)
        prior_log = torch.log(prior.clamp(min=self.prior_log_eps))
        prior_score = prior_log.unsqueeze(0).expand(B, -1, -1)
        valid = prior > 0
        valid_exp = valid.unsqueeze(0)  # (1, C, S)

        scales = torch.nn.functional.softplus(self.prior_scales[layer_idx])  # (H,)

        # ---- 逐 head 独立完整 attention ----
        head_msgs = []
        head_attns = []
        head_gates = []
        for head_idx in range(self.n_heads):
            q = (
                self.query_hidden_layers[layer_idx][head_idx](h_conf)
                + self.query_identity_layers[layer_idx][head_idx](target_id)
            )
            k = self.key_layers[layer_idx][head_idx](h_src)
            v = self.value_layers[layer_idx][head_idx](h_src)

            dynamic_score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.attention_dim)

            if self.architecture == "hybrid_bilinear_prior_only":
                score = prior_score
                gate = torch.zeros_like(score)
            elif self.architecture == "hybrid_bilinear_no_gate":
                s = scales[head_idx]
                score = dynamic_score + s * prior_score
                gate = torch.ones_like(score)
            else:
                gate_logits = (
                    self.gate_q_layers[layer_idx][head_idx](q)
                    + self.gate_k_layers[layer_idx][head_idx](k).transpose(1, 2)
                    + self.gate_p_layers[layer_idx][head_idx](prior_score.unsqueeze(-1)).squeeze(-1)
                )
                gate = torch.sigmoid(gate_logits)
                s = scales[head_idx]
                score = gate * dynamic_score + (1.0 - gate) * s * prior_score

            score = score / self.attention_temperature
            score = score.masked_fill(~valid_exp, -1e9)
            attention = torch.softmax(score, dim=-1)
            attention = attention * valid_exp.float()
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            attention = self.attention_dropout(attention)

            msg_h = torch.bmm(attention, v)  # (B, C, hidden_dim)
            # target-specific value 的低秩修正（LoRA）：
            # m_c = (W_V + B_c·A_c)·h̄_c，h̄_c 为注意力加权的源隐藏态。
            if self.value_lora_rank > 0:
                h_ctx = torch.bmm(attention, h_src)              # (B, C, hidden)
                A_h = self.lora_A[layer_idx][head_idx]           # (C, r, hidden)
                B_h = self.lora_B[layer_idx][head_idx]           # (C, hidden, r)
                z = torch.einsum("crd,bcd->bcr", A_h, h_ctx)     # (B, C, r)
                delta = torch.einsum("cdr,bcr->bcd", B_h, z)     # (B, C, hidden)
                msg_h = msg_h + delta
            head_msgs.append(msg_h)
            head_attns.append(attention)
            head_gates.append(gate)

        # ---- 第二级：head 聚合，得到 head_weight (B, C, H) ----
        msg_stack = torch.stack(head_msgs, dim=2)  # (B, C, H, hidden_dim)
        if self.head_aggregation == "static_gate":
            # GLM 方案：权重只取决于 target 身份（静态，与输入无关）
            head_logits = self.head_gate_layers[layer_idx](target_id)  # (B, C, H)
            head_weight = torch.softmax(
                head_logits / self.head_gate_temperature, dim=-1
            )
        else:
            # 用户思路：对 head 维度再做一次注意力。
            # query = target 身份 + 当前 conf 隐藏态；key = 各 head 的实际输出。
            # 权重随输入内容动态变化，梯度直连 head 输出。
            q_head = (
                self.head_attn_q_id_layers[layer_idx](target_id)
                + self.head_attn_q_hidden_layers[layer_idx](h_conf)
            )  # (B, C, hidden)
            k_head = self.head_attn_k_layers[layer_idx](msg_stack)  # (B, C, H, hidden)
            head_score = (q_head.unsqueeze(2) * k_head).sum(dim=-1)  # (B, C, H)
            head_score = head_score / math.sqrt(self.hidden_dim)
            head_weight = torch.softmax(
                head_score / self.head_gate_temperature, dim=-1
            )

        # 加权求和: (B, C, H, hidden) * (B, C, H, 1) -> (B, C, hidden)
        hw = head_weight.unsqueeze(-1)             # (B, C, H, 1)
        msg = (msg_stack * hw).sum(dim=2)          # (B, C, hidden_dim)

        # ---- target-specific value（FiLM）：msg_c ← (1+Γ_c) ⊙ msg_c + B_c ----
        # 因注意力权重与 head 权重均为凸组合，在汇聚后 msg 上做 FiLM
        # 等价于在 value v_j 上做 per-target 调制 v(c,j)=γ_c⊙v_j+β_c。
        if self.value_film == "static":
            gamma = 1.0 + self.film_gamma[layer_idx].unsqueeze(0)  # (1, C, hidden)
            beta = self.film_beta[layer_idx].unsqueeze(0)          # (1, C, hidden)
            msg = gamma * msg + beta
        elif self.value_film == "dynamic":
            raw_g = (self.film_g_id[layer_idx](target_id)
                     + self.film_g_h[layer_idx](h_conf))           # (B, C, hidden)
            raw_b = (self.film_b_id[layer_idx](target_id)
                     + self.film_b_h[layer_idx](h_conf))           # (B, C, hidden)
            msg = (1.0 + raw_g) * msg + raw_b

        # 诊断信息（用 head 0 的 attention 作代表）
        attn0 = head_attns[0]
        full_attention = torch.zeros(
            B, self.n_nodes, self.n_nodes, dtype=attn0.dtype, device=attn0.device
        )
        if self.bipartite:
            conf_rows = full_attention[:, self.conf_indices, :]
            conf_rows[:, :, self.general_indices] = attn0
            full_attention[:, self.conf_indices, :] = conf_rows
        else:
            full_attention[:, self.conf_indices, :] = attn0
        self.last_attention = full_attention.detach()
        self.last_gate = head_gates[0].detach()
        self.last_prior = prior.detach()
        self.last_head_weight = head_weight.detach()

        return msg

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
    n_general: int | None = None,
    bipartite: bool = False,
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

    bipartite=True 时（需提供 n_general）：
      在常规掩码构建完成后，强制把 Confidential-Confidential 块清零，
      只保留 General-General 与 Confidential-General 边，使图结构严格
      匹配攻击者威胁模型（攻击者只能读取 General 字段）。
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
                f"Invalid adaptive edge settings: min_k={min_k}, max_k={max_k}"
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

    if bipartite:
        if n_general is None or not (0 < n_general < n):
            raise ValueError("bipartite=True 需提供有效的 n_general")
        # Confidential 节点索引为 [n_general, n)
        mask[n_general:, n_general:] = False

    return mask.astype(np.float32)
