# -*- coding: utf-8 -*-
"""改造后的 GRPO 定级策略：逐字段共享的打分网络 + 组内相对优势 + KL 约束 + 熵正则。
动作：选一个字段，上调到“刚好能消除它所在的某个越线组合”的最低等级；直到一致为止。"""
import numpy as np, torch, torch.nn as nn
import env

NF = 17  # 每个字段的特征维数（含全局特征）


def features(ins, l):
    """返回 (特征矩阵 P×NF, 合法动作掩码 P, 各字段的目标等级 P)。"""
    P = ins.g["P"]; un = ins.unresolved(l); Au = ins.A[un]; glu = ins.gl[un]; exu = ins.ex[un]
    valid = ~ins.locked & (l < env.TOP) & Au.any(0)
    nun = max(1, int(un.sum()))
    n2 = (Au & (glu == 2)[:, None]).sum(0) / nun; n3 = (Au & (glu == 3)[:, None]).sum(0) / nun
    tgt = np.zeros(P, int); ratio = np.zeros(P); hit_now = np.zeros(P)
    for i in np.where(valid)[0]:
        Ls = glu[Au[:, i]]; tgt[i] = Ls[Ls > l[i]].min()
        hit_now[i] = (Au[:, i] & (glu <= tgt[i])).sum(); ratio[i] = hit_now[i] / (ins.w[i] * (tgt[i] - l[i]))
        for L in set(Ls[Ls > l[i]]):
            ratio[i] = max(ratio[i], (Au[:, i] & (glu <= L)).sum() / (ins.w[i] * (L - l[i])))
    M = ins.m_cur(l); M0 = ins.g["M0"]
    cs = np.unique(ins.tc[un]) if un.any() else np.arange(ins.g["C"])
    mk = M[:, cs].max(1); m0 = M0[:, cs].max(1)
    mex = np.array([exu[Au[:, i]].mean() if Au[:, i].any() else 0.0 for i in range(P)])
    # 在它所在的未消除组合里，它是否是“上调代价最小”的成员
    need = np.where(Au, (glu[:, None] - l[None, :]).clip(min=0) * ins.w[None, :], np.inf)
    need[:, ins.locked] = np.inf
    cheapest = np.array([(need[Au[:, i], i] <= need[Au[:, i]].min(1) + 1e-9).mean() if Au[:, i].any() else 0.0 for i in range(P)])
    size = np.array([Au[Au[:, i]].sum(1).mean() / 3 if Au[:, i].any() else 0.0 for i in range(P)])
    f = np.stack([ins.l0 / 3, l / 3, ins.w / 2, ins.locked.astype(float), valid.astype(float), n2, n3,
                  ratio / max(1e-9, ratio.max()), hit_now / nun, ins.w * (env.TOP - l) / 4, m0, mk, np.clip(mk - m0, 0, 1), mex, cheapest, size,
                  np.full(P, un.sum() / max(1, len(ins.gl)))], 1)
    return f.astype(np.float32), valid, tgt


class Policy(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(NF, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, f, valid):
        """f: (P, NF)，valid: (P,) bool → 合法动作上的 log 概率 (P,)。字段表示 + 合法字段的均值池化作上下文。"""
        z = self.phi(f); ctx = z[valid].mean(0, keepdim=True).expand_as(z)
        logit = self.rho(torch.cat([z, ctx], 1)).squeeze(1).masked_fill(~valid, -1e9)
        return torch.log_softmax(logit, 0)


def rollout(ins, pol, dev, greedy=False, rng=None, keep=False):
    """跑一条轨迹。返回 (最终等级, [(特征, 掩码, 动作)])。"""
    l = ins.l0.copy(); traj = []
    while ins.unresolved(l).any():
        f, valid, tgt = features(ins, l)
        ft = torch.from_numpy(f).to(dev); vt = torch.from_numpy(valid).to(dev)
        with torch.no_grad():
            lp = pol(ft, vt)
        a = int(lp.argmax()) if greedy else int(rng.choice(len(valid), p=np.exp(lp.double().cpu().numpy()) / np.exp(lp.double().cpu().numpy()).sum()))
        if keep:
            traj.append((ft, vt, a))
        l[a] = tgt[a]
    return l, traj


def best_of(ins, pol, dev, n, rng):
    """采样 n 条轨迹（含一条贪心解码），各自回退后取代价最小的。"""
    best = env.prune(ins, rollout(ins, pol, dev, greedy=True)[0])
    for _ in range(n - 1):
        l = env.prune(ins, rollout(ins, pol, dev, rng=rng)[0])
        if ins.cost(l) < ins.cost(best):
            best = l
    return best
