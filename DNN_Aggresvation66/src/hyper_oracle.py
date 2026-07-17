"""掩码条件化参数生成 oracle（DNN66）。

65 号证明"共享初始化 + 微调"（优化式元学习）失败：子集任务太发散，不存在一个对
所有子集都 K 步可达的初始化。本模块换成"表示式元学习"：hypernetwork 以掩码 m 为
任务描述子，直接生成该子集的专属参数（FiLM 调制），把"一个点靠近所有最优解"的
不可能要求，替换成"学一个光滑映射 m -> θ*(m)"。

结构：
- backbone：与 63 号 MLPOracle 同容量（2nG -> 256 ×3 -> nC），保证公平；
- hypernet：m (nG) -> 128 -> 每个隐藏层的 (γ, β)，FiLM 施加在 ReLU 之后：h = γ⊙h + β；
  γ 初始化恒等（最后一层零初始化 + γ=1+δ），训练起点等价于普通 backbone。

查询时两种用法：
- K0：直接前向（γβ 来自 hypernet）；
- adapter 精修：把 hypernet(m) 生成的 γβ 拆成自由张量，冻结其余全部参数，只微调这
  ~1.5k 个 adapter 参数（参数高效 -> 可用大学习率而不崩，对照 64 号全参大 lr 崩溃）。
"""
from __future__ import annotations

import torch
from torch import nn


class HyperOracle(nn.Module):
    def __init__(self, n_general: int, n_confidential: int,
                 hidden: int = 256, hyper_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.nG, self.nC, self.H = n_general, n_confidential, hidden
        self.lin1 = nn.Linear(2 * n_general, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_confidential)
        self.drop = nn.Dropout(dropout)
        # hypernet: m -> (γ1,β1,γ2,β2,γ3,β3)，最后一层零初始化 => 起点为恒等 FiLM
        self.hyper = nn.Sequential(
            nn.Linear(n_general, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, 6 * hidden),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def film_params(self, m: torch.Tensor) -> list[torch.Tensor]:
        """m: (B,nG) -> [γ1,β1,γ2,β2,γ3,β3]，每个 (B,H)。γ = 1 + δ。"""
        raw = self.hyper(m).view(-1, 6, self.H)
        return [1.0 + raw[:, 0], raw[:, 1], 1.0 + raw[:, 2], raw[:, 3],
                1.0 + raw[:, 4], raw[:, 5]]

    def forward(self, x: torch.Tensor, m: torch.Tensor,
                film: list[torch.Tensor] | None = None) -> torch.Tensor:
        if film is None:
            film = self.film_params(m)
        g1, b1, g2, b2, g3, b3 = film
        h = torch.relu(self.lin1(torch.cat([x * m, m], dim=1)))
        h = self.drop(g1 * h + b1)
        h = torch.relu(self.lin2(h))
        h = self.drop(g2 * h + b2)
        h = torch.relu(self.lin3(h))
        h = self.drop(g3 * h + b3)
        return self.out(h)
