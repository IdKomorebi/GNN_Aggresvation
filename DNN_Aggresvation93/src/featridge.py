# -*- coding: utf-8 -*-
"""91 号核心：**摊销表示 + 闭式读出**（Frozen-φ Ridge）。

诊断（本号的出发点）
--------------------
v_c(S) = 1 − min_g E[(Y_c − g(X_S))²]/Var(Y_c) 是一个**内层优化问题的最优值**。
当前 oracle 用一组共享权重 θ 去逼近全部 2^44 个 S 各自的最优解 g_S*，
属于**全摊销**，其 amortization gap 有一条已知性质：
**在"最优解离共享平均解最远"的实例上最大**——强协同集合正是这种实例
（要求精确配合的除法电路，训练掩码里出现率万分之几）。

两个推论，都与既有实证吻合：
1. 84 号测到集成方差 σ 在最强协同档是**平的**——因为所有集成成员共享同一结构性偏差，
   方差看不见它。⟹ 失明是**偏差**不是噪声。
2. syn = v(S) − max_T v(T) 是两个 0.8 量级的数相减得 0.1，
   误差 = gap_S − gap_T，而 gap 在强协同 S 上偏大、在其平庸父集 T 上偏小
   ⟹ **syn 被系统性压低**。

方案
----
把 oracle 从"值函数逼近器"重新定位为"**条件特征提取器**"：

    φ_θ(x⊙m_S, m_S)  ──摊销──►  条件特征（共享，一次前向）
    β_{S,c} = (ΦᵀΦ + λI)⁻¹ ΦᵀY_c  ──闭式──►  逐 (S,c) 独有，不共享，不受干涉

主干 θ 完全冻结（**零训练**）。每查询只需一次前向 + 一个 d×d solve。

与既有方法的统一：四种方法是同一框架沿"φ 的适应程度"排列的四个点——
poly2 闭式(φ=手工多项式基) / L0(φ=冻结 oracle 特征) / L1(φ=meta-训练) / 微调(φ,β 都动)。
**poly2 是 φ 被冻成手工字典的退化情形**，不是"另建的模型"。

工程保险
--------
`kind` 含 "+p2" 时把 poly2 基（raw+square+pairwise）拼进 φ，
使列空间**严格包含** 90 号的 poly2 字典 ⟹ 岭回归在包含关系下不会更差。
同时提供纯 φ 版本，用于归因"oracle 特征本身贡献多少"。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

# 与 85/88/90 号同一 alpha 网格，便于横向比较
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
VAL_SEED = 20260724      # 与 85/90 号一致：train 内部 fit/val 划分
SPLIT_SEED = 880725      # 与 88/90 号一致：held-out 再切 search/audit

# MLPOracle.net 的 Sequential 下标：Linear,ReLU,Dropout ×3, Linear
# ReLU 之后的激活位于 1 / 4 / 7
RELU_IDX = (1, 4, 7)


class FrozenPhi:
    """冻结的 oracle 主干，取 ReLU 激活作为条件特征 φ(x⊙m, m)。

    kind:
      last  — 只取最后一层 ReLU 激活（256 维）
      cat3  — 三层 ReLU 激活拼接（768 维）。这是 eNTK 的廉价代理：
              微调所有层的一阶效果等价于在 Jacobian 特征空间做线性回归，
              而多层激活拼接是它最便宜的近似。
    """

    def __init__(self, model: torch.nn.Module, kind: str = "cat3"):
        self.net = model.net
        self.model = model.eval()
        self.idx = RELU_IDX if kind == "cat3" else RELU_IDX[-1:]

    @torch.no_grad()
    def __call__(self, X: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """X:(n,nG)  masks:(B,nG) → (B,n,d)   Dropout 在 eval 下是恒等。"""
        B, n = masks.shape[0], X.shape[0]
        xm = X.unsqueeze(0) * masks.unsqueeze(1)                  # (B,n,nG)
        mm = masks.unsqueeze(1).expand(-1, n, -1)                 # (B,n,nG)
        h = torch.cat([xm, mm], dim=2).reshape(B * n, -1)
        outs = []
        for i, layer in enumerate(self.net):
            h = layer(h)
            if i in self.idx:
                outs.append(h)
            if i >= max(self.idx):
                break
        return torch.cat(outs, dim=1).reshape(B, n, -1)


def poly2_block(Z: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
    """(B,n,m+m+C(m,2))：S 内字段的 raw + square + 两两乘积（= 90 号 poly2 字典）。"""
    B, m = sets.shape
    raw = Z[:, sets.reshape(-1)].T.reshape(B, m, -1).transpose(1, 2)   # (B,n,m)
    cols = [raw, raw ** 2]
    if m > 1:
        cols.append(torch.stack([raw[:, :, a] * raw[:, :, b]
                                 for a, b in combinations(range(m), 2)], dim=2))
    return torch.cat(cols, dim=2)


def build_features(phi: FrozenPhi | None, X: torch.Tensor, Z: torch.Tensor,
                   sets: torch.Tensor, n_gen: int, kind: str) -> torch.Tensor:
    """按 kind 组装 (B,n,d) 特征。kind ∈ {last, cat3, poly2, last+p2, cat3+p2}。"""
    blocks = []
    if kind.startswith(("last", "cat3")):
        masks = torch.zeros(sets.shape[0], n_gen, device=X.device, dtype=X.dtype)
        masks.scatter_(1, sets, 1.0)
        blocks.append(phi(X, masks).to(Z.dtype))
    if kind == "poly2" or kind.endswith("+p2"):
        blocks.append(poly2_block(Z, sets))
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=2)


def ridge_r2(F_tr: torch.Tensor, Y_tr: torch.Tensor,
             F_ev: torch.Tensor, Y_ev: torch.Tensor,
             fit_idx: torch.Tensor, val_idx: torch.Tensor,
             ret_resid: bool = False):
    """批量闭式岭回归 → (B,nC) 的独立集 R²。

    口径与 85/88/90 号逐条对齐：train 内部 fit/val 选 alpha（逐 set×conf 独立选），
    再用全 train 拟合、在独立集上评估。**评估集从不参与 alpha 选择或拟合。**

    ret_resid=True 时额外返回 (B,n_ev,nC) 的逐样本残差平方，
    供"同一批样本上 S 与 T 的**配对**损失差"使用——这正是闭式解相对微调的
    结构性优势：没有优化噪声，S 与 T 的误差来自同一个 φ，差分大量抵消。
    """
    mu = F_tr.mean(1, keepdim=True)
    sd = F_tr.std(1, keepdim=True).clamp_min(1e-8)
    T, E = (F_tr - mu) / sd, (F_ev - mu) / sd
    B, _, p = T.shape
    nC = Y_tr.shape[1]
    eye = torch.eye(p, device=T.device, dtype=T.dtype).expand(B, p, p)

    Tf, Tv = T[:, fit_idx], T[:, val_idx]
    Yf, Yv = Y_tr[fit_idx], Y_tr[val_idx]
    ymf = Yf.mean(0)
    Gf = Tf.transpose(1, 2) @ Tf
    Hf = torch.einsum("bnp,nc->bpc", Tf, Yf - ymf)
    Gv = Tv.transpose(1, 2) @ Tv
    Hv = torch.einsum("bnp,nc->bpc", Tv, Yv - ymf)
    rtrv = ((Yv - ymf) ** 2).sum(0)

    best = torch.full((B, nC), float("inf"), device=T.device, dtype=T.dtype)
    best_a = torch.zeros(B, nC, dtype=torch.long, device=T.device)
    for ai, a in enumerate(ALPHAS):
        beta = torch.linalg.solve(Gf + a * eye, Hf)
        sse = rtrv - 2 * (beta * Hv).sum(1) + (beta * (Gv @ beta)).sum(1)
        upd = sse < best
        best = torch.where(upd, sse, best)
        best_a = torch.where(upd, ai, best_a)

    ym = Y_tr.mean(0)
    Ga = T.transpose(1, 2) @ T
    Ha = torch.einsum("bnp,nc->bpc", T, Y_tr - ym)
    sst = ((Y_ev - Y_ev.mean(0)) ** 2).sum(0) + 1e-12
    sse_ev = torch.zeros(B, nC, device=T.device, dtype=T.dtype)
    resid = torch.zeros(B, E.shape[1], nC, device=T.device, dtype=T.dtype) if ret_resid else None
    for ai, a in enumerate(ALPHAS):
        beta = torch.linalg.solve(Ga + a * eye, Ha)
        err = (Y_ev - ym).unsqueeze(0) - E @ beta                   # (B,n_ev,nC)
        sse_ev = torch.where(best_a == ai, (err ** 2).sum(1), sse_ev)
        if ret_resid:
            resid = torch.where((best_a == ai).unsqueeze(1), err ** 2, resid)
    r2 = (1 - sse_ev / sst).clamp_min(0.0)
    return (r2, resid, sst) if ret_resid else r2


def fit_val_idx(n_train: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """train 内部 fit/val 划分，与 85/90 号同一种子同一比例。"""
    p = np.random.RandomState(VAL_SEED).permutation(n_train)
    nv = round(n_train * 0.15)
    return (torch.as_tensor(p[nv:], device=device),
            torch.as_tensor(p[:nv], device=device))


def load_oracle(ckpt_path, n_general: int, n_conf: int, device):
    """加载 75 号训练好的 MLPOracle（本号完全冻结，零训练）。"""
    from src.oracle import MLPOracle
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPOracle(n_general, n_conf).to(device)
    model.load_state_dict(ck["state"] if "state" in ck else ck)
    return model.eval()
