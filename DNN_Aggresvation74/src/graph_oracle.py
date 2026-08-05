# -*- coding: utf-8 -*-
"""DNN74：可配置的掩码图 oracle（MaskedGraphOracle）。

背景
----
63 号的 `DNN_Aggresvation69/src/oracle.py:GNNOracle` 是从零重写的，**没有继承
42–55 号十几个子项目调出来的图结构成果**：

  - 它的 G→C 聚合 = `q·k + 单一全局 prior_scale × log(先验)`，这正好等于 51/52 号
    实测的 `gat_static`（0.8523），而 51 §4.4 明说它是"唯一稳健最差"的两个之一；
    项目实测最优是 `gcn_dynamic`（0.8746，先验边 × 逐样本 sigmoid 门控、无 q·k）。
  - 相关性融合写死为 5 指标算术平均（`register_buffer`，不可学），而 45 号证明
    uniform_alpha 会掉 0.0125。
  - `train_oracle.py:73` 只读了 base.yaml 的 `graph.top_k/threshold`，整个 `model:` 段
    （num_layers 4 / edge_alpha_temperature 2.0 / attention_temperature 0.25 / hidden 64）
    和 `training:` 段（epochs 500 / patience 150）对 oracle 完全没生效。

本模块把 42–55 号的全部设计维度重新变成可配置开关，用于回答：
**"GNN 不如 MLP"是结构性结论，还是实现缺陷？**

设计原则
--------
1. **不修改 69 号的任何文件**（63–73 全部结论的依赖）。本文件是唯一的新模型代码。
2. 六种聚合的数学形式**照抄** `DNN_Aggresvation69/src/model.py:_unified_messages`
   （42–55 号验证过的实现），只加上 oracle 特有的**可见性掩码**。
3. 通过开关覆盖全部变体，避免复制粘贴分叉。
4. `gg_mode=gcn_static, gc_mode=gat_static, conf_per_layer=False, learnable_alpha=False,
   input_encoder="linear", readout="mean", attn_temperature=1.0, hidden=128, n_layers=3`
   这一组配置**在结构上等价于 63 号的 GNNOracle**，用作复现锚点（A0_repro）。

节点特征
--------
每个 general 节点的输入固定为 `[x·m, m]`（值 × 可见位, 可见位），与 GNNOracle 一致。
可见位是 oracle 的核心信息，不能去掉；各种 gate 的输入也因此隐式含可见性。
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AGG_MODES = ("gcn_noprior", "gcn_static", "gcn_dynamic",
             "gat_noprior", "gat_static", "gat_dynamic")


class MaskedGraphOracle(nn.Module):
    """可配置的掩码图 oracle。

    参数
    ----
    metric_tensor : (N, N, n_metrics) 相关性张量（原始 5 指标，不是平均后的）。
        可学习 alpha 需要原始张量；固定 alpha 时内部取算术平均（= build_priors 口径）。
    edge_mask : (N, N) 二值拓扑，来自 `build_edge_mask(..., bipartite=True)`。
    n_general, n_confidential : 字段数。约定 general 节点索引 [0, nG)，conf 在 [nG, N)。
    gg_mode / gc_mode : general→general / general→confidential 的聚合方式，见 AGG_MODES。
    learnable_alpha : True 时用可学习 beta（softmax over n_metrics）融合相关性指标（B1）。
    edge_alpha_temperature : beta 的 softmax 温度。43 号 multi 最优 2.0（防止塌缩到单一指标）。
    input_encoder : "linear"（Linear+ReLU，= 63 号现状）| "mlp"（2 层 MLP，= 42–55 号最优）。
    readout : "mean"（softmax / 行归一，= 现状）| "sum"（不归一化 + 把可见字段数拼进 head）。
        总发电、计量负荷这类 confidential 本质是可见字段的**和**，mean 聚合会丢总量信息。
    conf_per_layer : False 时 G→C 只在最后一层做一次（= 63 号）；True 时每层都做、
        conf 隐状态逐层精化（= 42–55 号 InferenceDrivenGNN 的做法）。
    attn_temperature : GAT 系 softmax 前的温度。42 号实测越 sharp 越好（0.25）；
        63 号隐含为 1.0。对 GCN 系无效（没有 softmax）。
    """

    def __init__(
        self,
        metric_tensor: np.ndarray,
        edge_mask: np.ndarray,
        n_general: int,
        n_confidential: int,
        hidden: int = 128,
        n_layers: int = 3,
        gg_mode: str = "gcn_static",
        gc_mode: str = "gat_static",
        learnable_alpha: bool = False,
        edge_alpha_temperature: float = 2.0,
        input_encoder: str = "linear",
        readout: str = "mean",
        conf_per_layer: bool = False,
        attn_temperature: float = 1.0,
        dropout: float = 0.1,
        prior_log_eps: float = 1e-4,
        mlp_bypass: bool = False,
    ) -> None:
        super().__init__()
        for name, mode in (("gg_mode", gg_mode), ("gc_mode", gc_mode)):
            if mode not in AGG_MODES:
                raise ValueError(f"未知 {name}: {mode}（可选 {AGG_MODES}）")
        if input_encoder not in ("linear", "mlp"):
            raise ValueError(f"未知 input_encoder: {input_encoder}")
        if readout not in ("mean", "sum"):
            raise ValueError(f"未知 readout: {readout}")

        self.nG, self.nC, self.H = n_general, n_confidential, hidden
        self.n_layers = n_layers
        self.gg_mode, self.gc_mode = gg_mode, gc_mode
        self.learnable_alpha = bool(learnable_alpha)
        self.edge_alpha_temperature = max(float(edge_alpha_temperature), 1e-3)
        self.readout = readout
        self.conf_per_layer = bool(conf_per_layer)
        self.attn_temperature = max(float(attn_temperature), 1e-3)
        self.prior_log_eps = float(prior_log_eps)
        self.mlp_bypass = bool(mlp_bypass)

        mt = torch.as_tensor(metric_tensor, dtype=torch.float32)
        em = torch.as_tensor(edge_mask, dtype=torch.float32)
        self.n_metrics = mt.shape[2]
        # 只保留 general 源列：G-G 块 与 C←G 块（严格匹配威胁模型：攻击者只能读 general）
        self.register_buffer("mt_gg", mt[:n_general, :n_general, :].contiguous())
        self.register_buffer("mt_cg", mt[n_general:, :n_general, :].contiguous())
        em_gg = em[:n_general, :n_general].clone()
        em_gg.fill_diagonal_(0.0)                       # 无自环，与 build_priors 一致
        self.register_buffer("em_gg", em_gg)
        self.register_buffer("em_cg", em[n_general:, :n_general].contiguous())

        if self.learnable_alpha:
            # 43/45 号：G-G 共享一组 beta；C←G 逐边一组 beta（自由度更高）
            self.beta_gg = nn.Parameter(torch.zeros(self.n_metrics))
            self.beta_cg = nn.Parameter(torch.zeros(n_confidential, n_general, self.n_metrics))

        # ---- 输入编码：每节点 [值·可见位, 可见位] ----
        if input_encoder == "mlp":      # 52 号最优（enc3）
            self.enc = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        else:                            # 63 号现状（enc2）
            self.enc = nn.Sequential(nn.Linear(2, hidden), nn.ReLU())

        # ---- general-general 传播 ----
        self.gg_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.gg_neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.gg_bn = nn.ModuleList([nn.BatchNorm1d(n_general) for _ in range(n_layers)])
        self._make_agg_params("gg", gg_mode, n_layers, hidden)

        # ---- general→confidential ----
        self.conf_emb = nn.Parameter(torch.randn(n_confidential, hidden) * 0.01)
        n_gc = n_layers if self.conf_per_layer else 1
        self.n_gc = n_gc
        self._make_agg_params("gc", gc_mode, n_gc, hidden)
        if self.conf_per_layer:
            self.gc_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_gc)])
            self.gc_neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_gc)])
            self.gc_bn = nn.ModuleList([nn.BatchNorm1d(n_confidential) for _ in range(n_gc)])

        # ---- 输出头（逐 confidential）----
        head_in = hidden + (1 if readout == "sum" else 0)   # sum 口径额外喂"可见字段数"
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, hidden // 2), nn.ReLU(),
                          nn.Linear(hidden // 2, 1))
            for _ in range(n_confidential)
        ])
        self.dropout = nn.Dropout(dropout)

        # ---- 诊断旁路（D_bypass）：仅用于读"图分支到底贡献了多少"，不是候选方案 ----
        if self.mlp_bypass:
            self.bypass = nn.Sequential(
                nn.Linear(2 * n_general, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, n_confidential),
            )

    # ------------------------------------------------------------------ #
    # 参数构造                                                            #
    # ------------------------------------------------------------------ #
    def _make_agg_params(self, tag: str, mode: str, n: int, hidden: int) -> None:
        """按聚合方式只创建它真正需要的参数（避免无用参数拉高参数量对比）。"""
        need_qk = mode.startswith("gat") or mode.endswith("dynamic")
        need_gate = mode.endswith("dynamic")
        need_scale = mode in ("gat_static", "gat_dynamic")
        need_value = mode.startswith("gat")

        if need_qk:
            setattr(self, f"{tag}_q", nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n)]))
            setattr(self, f"{tag}_k", nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n)]))
        if need_value:
            setattr(self, f"{tag}_v", nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n)]))
        if need_gate:
            # gate_logits = W_q q + W_k k + W_p log(P) + b（等价 W[q‖k‖logP]+b，避免大张量）
            setattr(self, f"{tag}_gq", nn.ModuleList(
                [nn.Linear(hidden, 1, bias=False) for _ in range(n)]))
            setattr(self, f"{tag}_gk", nn.ModuleList(
                [nn.Linear(hidden, 1, bias=False) for _ in range(n)]))
            gp = nn.ModuleList([nn.Linear(1, 1, bias=True) for _ in range(n)])
            for m in gp:
                nn.init.constant_(m.bias, -3.0)     # base.yaml 的 gate_bias_init
            setattr(self, f"{tag}_gp", gp)
        if need_scale:
            setattr(self, f"{tag}_scale", nn.ParameterList(
                [nn.Parameter(torch.tensor(1.0)) for _ in range(n)]))

    # ------------------------------------------------------------------ #
    # 先验                                                                #
    # ------------------------------------------------------------------ #
    def _prior_gg(self) -> torch.Tensor:
        """(nG,nG) general-general 相关性先验，非负、已稀疏化、无自环。"""
        if self.learnable_alpha:
            a = torch.softmax(self.beta_gg / self.edge_alpha_temperature, dim=0)
            corr = torch.einsum("ijk,k->ij", self.mt_gg, a)
        else:
            corr = self.mt_gg.mean(dim=2)           # = build_priors 口径
        return corr.clamp(min=0.0) * self.em_gg

    def _prior_cg(self) -> torch.Tensor:
        """(nC,nG) confidential←general 相关性先验。"""
        if self.learnable_alpha:
            a = torch.softmax(self.beta_cg / self.edge_alpha_temperature, dim=-1)
            corr = torch.sum(self.mt_cg * a, dim=-1)
        else:
            corr = self.mt_cg.mean(dim=2)
        return corr.clamp(min=0.0) * self.em_cg

    # ------------------------------------------------------------------ #
    # 通用聚合：六种模式 + 可见性                                          #
    # ------------------------------------------------------------------ #
    def _aggregate(self, mode: str, tag: str, li: int,
                   h_tgt: torch.Tensor, h_src: torch.Tensor,
                   P: torch.Tensor, topo: torch.Tensor,
                   m: torch.Tensor) -> torch.Tensor:
        """把 h_src（B,S,H）沿边聚合到 h_tgt（B,T,H）对应的目标上，返回 (B,T,H)。

        数学形式照抄 model.py:_unified_messages；新增的只有"可见性"：
        源节点 j 不可见（m_j=0）时，其权重被强制为 0，其余权重重新归一化
        （这正是 GNNOracle 的做法，也是"沿相关边替补被遮蔽字段"的实现）。

        P    : (T,S) 相关性先验
        topo : (T,S) 二值拓扑（= edge_mask 块）
        m    : (B,S)  源节点可见位
        """
        B = h_src.shape[0]
        mvis = m.unsqueeze(1)                                   # (B,1,S)
        valid = (topo.unsqueeze(0) > 0) & (mvis > 0)            # (B,T,S)

        # ---------------- GCN 系：无 q·k，加权邻接直接聚合 raw h ----------------
        if mode.startswith("gcn"):
            if mode == "gcn_noprior":
                A = topo.unsqueeze(0).expand(B, -1, -1)
            elif mode == "gcn_static":
                A = P.unsqueeze(0).expand(B, -1, -1)
            else:                                               # gcn_dynamic
                prior_log = torch.log(P.clamp(min=self.prior_log_eps)).unsqueeze(0)
                q = getattr(self, f"{tag}_q")[li](h_tgt)        # (B,T,H)
                k = getattr(self, f"{tag}_k")[li](h_src)        # (B,S,H)
                gate_logits = (getattr(self, f"{tag}_gq")[li](q)
                               + getattr(self, f"{tag}_gk")[li](k).transpose(1, 2)
                               + getattr(self, f"{tag}_gp")[li](
                                   prior_log.unsqueeze(-1)).squeeze(-1))
                A = torch.sigmoid(gate_logits) * P.unsqueeze(0)  # 动态调制每条先验边强度
            A = A * valid.float()
            A = A / A.sum(-1, keepdim=True).clamp(min=1e-12)
            if self.readout == "sum":
                A = A * self._count_scale(valid)
            return torch.bmm(A, h_src)

        # ---------------- GAT 系：q·k 注意力聚合 value(h) ----------------
        q = getattr(self, f"{tag}_q")[li](h_tgt)
        k = getattr(self, f"{tag}_k")[li](h_src)
        v = getattr(self, f"{tag}_v")[li](h_src)
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.H)

        if mode in ("gat_static", "gat_dynamic"):
            prior_log = torch.log(P.clamp(min=self.prior_log_eps)).unsqueeze(0)
            scale = F.softplus(getattr(self, f"{tag}_scale")[li])
            if mode == "gat_static":
                score = score + scale * prior_log
            else:
                gate_logits = (getattr(self, f"{tag}_gq")[li](q)
                               + getattr(self, f"{tag}_gk")[li](k).transpose(1, 2)
                               + getattr(self, f"{tag}_gp")[li](
                                   prior_log.unsqueeze(-1)).squeeze(-1))
                gate = torch.sigmoid(gate_logits)
                score = gate * score + (1.0 - gate) * scale * prior_log

        score = score / self.attn_temperature
        score = score.masked_fill(~valid, -1e9)
        attn = torch.softmax(score, dim=-1) * valid.float()
        attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-12)   # 无可见邻居→全 0
        if self.readout == "sum":
            attn = attn * self._count_scale(valid)
        attn = self.dropout(attn)
        return torch.bmm(attn, v)

    def _count_scale(self, valid: torch.Tensor) -> torch.Tensor:
        """sum 聚合的尺度因子：把"归一化均值"还原成"和"（差一个全局常数 1/nG）。

        mean 聚合会丢掉"有多少个可见邻居"这一信息；而 total_gen / metered_load_mw
        这类 confidential 本质上是可见字段的**和**（GIN 的核心论点：mean 无法区分
        {a} 与 {a,a}）。这里乘上可见邻居数、再除以 nG 保持有界（∈[0,1]），
        避免大集合下激活爆炸。forward 里还会把可见字段数作为标量特征拼进 head。
        """
        return valid.float().sum(-1, keepdim=True).clamp(min=1.0) / self.nG

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x: (B,nG) 标准化后的 general 值；m: (B,nG) 可见位 → (B,nC) 预测。"""
        B = x.shape[0]
        P_gg, P_cg = self._prior_gg(), self._prior_cg()

        h = self.enc(torch.stack([x * m, m], dim=-1))            # (B,nG,H)
        h_c = self.conf_emb.unsqueeze(0).expand(B, -1, -1)       # (B,nC,H)

        for li in range(self.n_layers):
            msg = self._aggregate(self.gg_mode, "gg", li, h, h, P_gg, self.em_gg, m)
            h_new = self.gg_self[li](h) + self.gg_neigh[li](msg)
            h_new = torch.relu(self.gg_bn[li](h_new))
            h = h + self.dropout(h_new)

            if self.conf_per_layer:
                ctx = self._aggregate(self.gc_mode, "gc", li, h_c, h, P_cg, self.em_cg, m)
                c_new = self.gc_self[li](h_c) + self.gc_neigh[li](ctx)
                c_new = torch.relu(self.gc_bn[li](c_new))
                h_c = h_c + self.dropout(c_new)

        if not self.conf_per_layer:
            h_c = self._aggregate(self.gc_mode, "gc", 0, h_c, h, P_cg, self.em_cg, m)

        if self.readout == "sum":
            nvis = m.sum(-1, keepdim=True).unsqueeze(1).expand(-1, self.nC, -1) / self.nG
            h_c = torch.cat([h_c, nvis], dim=-1)

        out = torch.cat([self.heads[c](h_c[:, c, :]) for c in range(self.nC)], dim=1)
        if self.mlp_bypass:
            out = out + self.bypass(torch.cat([x * m, m], dim=1))
        return out


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------- #
# 变体注册表                                                              #
# ---------------------------------------------------------------------- #
# Round A：只换聚合方式，其余严格对齐 63 号 GNNOracle（hidden128/3层/linear编码/
#          固定alpha/mean readout/attn_T=1.0/G→C只做一次）。
# Round B：在 Round A 胜出者上逐项叠加 42–55 号的已知增益。
_A_BASE = dict(hidden=128, n_layers=3, learnable_alpha=False, input_encoder="linear",
               readout="mean", conf_per_layer=False, attn_temperature=1.0, dropout=0.1)

VARIANTS: dict[str, dict] = {
    # ---------------- Round A ----------------
    # A0 在结构上等价于 63 号 GNNOracle，是新代码的复现锚点（V1 闸门）
    "A0_repro":       dict(_A_BASE, gg_mode="gcn_static",  gc_mode="gat_static"),
    "A4_gat_static":  dict(_A_BASE, gg_mode="gat_static",  gc_mode="gat_static"),
    "A1_gcn_dynamic": dict(_A_BASE, gg_mode="gcn_dynamic", gc_mode="gcn_dynamic"),
    "A2_gcn_noprior": dict(_A_BASE, gg_mode="gcn_noprior", gc_mode="gcn_noprior"),
    "A3_gat_dynamic": dict(_A_BASE, gg_mode="gat_dynamic", gc_mode="gat_dynamic"),
}

# Round B 的变体在 sched.py 里根据 Round A 的胜出者动态生成（见 build_round_b）。
B_DELTAS: dict[str, dict] = {
    "B1_alpha": dict(learnable_alpha=True, edge_alpha_temperature=2.0),   # 45/43 号
    "B2_enc":   dict(input_encoder="mlp"),                                # 52 号
    "B3_deep":  dict(n_layers=4, hidden=64),                              # 43/42 号（并把参数量降到 MLP 量级）
    "B4_train": dict(),          # 只改训练预算（EPOCHS/PATIENCE），在 train_variant.py 里处理
    "B5_sum":   dict(readout="sum"),                                      # 修 mean 聚合与"和"型 conf 的错配
    "B7_layered": dict(conf_per_layer=True),                              # 42–55 号 InferenceDrivenGNN 的做法
}
# 仅当 Round A 胜出者含 gat 模式时才有意义（GCN 系无 softmax，温度是 no-op）
B_DELTAS_GAT_ONLY: dict[str, dict] = {
    "B0_temp":  dict(attn_temperature=0.25),                              # 42 号最大的单项 config 增益
}


def build_round_b(winner: str) -> dict[str, dict]:
    """在 Round A 胜出者上生成 Round B 变体（逐项消融 + 全叠 + 诊断旁路）。"""
    base = dict(VARIANTS[winner])
    out: dict[str, dict] = {}
    deltas = dict(B_DELTAS)
    if "gat" in base["gg_mode"] or "gat" in base["gc_mode"]:
        deltas.update(B_DELTAS_GAT_ONLY)
    for name, d in deltas.items():
        out[name] = dict(base, **d)
    all_cfg = dict(base)
    for d in deltas.values():
        all_cfg.update(d)
    out["B6_all"] = all_cfg
    out["D_bypass"] = dict(all_cfg, mlp_bypass=True)     # 诊断用，不是候选方案
    return out
