# -*- coding: utf-8 -*-
"""01 号公用：数据读取、实例生成、一致性检查、基线算法与精确最优。只读主目录数据。"""
import os, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

REPO = "/data1/duhaocun/projects/GNN_Aggresvation"
GROUPS = {  # 标签: (真值目录, 估计文件)
    "RTS-GMLC": ("DNN_Aggresvation111/outputs", "RTS-GMLC.npz"),
    "NEM": ("DNN_Aggresvation112/outputs", "NEM.npz"),
    "PJM-load": ("DNN_Aggresvation116/groups/pjm_load/outputs", "PJM-load.npz"),
    "PJM-gen/ic": ("DNN_Aggresvation116/groups/pjm_gen_ic/outputs", "PJM-gen_ic.npz"),
    "CAISO-load": ("DNN_Aggresvation116/groups/caiso_load/outputs", "CAISO-load.npz"),
}
EST_DIR = os.path.join(REPO, "DNN_Aggresvation125/outputs/final_est")
TOP = 3  # 目标最高等级；算法上调上限也是 3 级


def closure(V, keys):
    """子集最大值闭包（keys 含全部规模 ≤3 的集合）。"""
    V = np.clip(V.astype(np.float64), 0, 1).copy(); idx = {k: r for r, k in enumerate(keys)}
    for r in sorted(range(len(keys)), key=lambda r: len(keys[r])):
        k = keys[r]
        for j in range(len(k)):
            s = k[:j] + k[j + 1:]
            if s:
                V[r] = np.maximum(V[r], V[idx[s]])
    return V


def minimal_sets(V, keys, c, thr):
    """对目标 c 超过 thr 的最小集合（任何真子集都不超过）。返回 [(成员, 超出量)]。"""
    idx = {k: r for r, k in enumerate(keys)}; out = []
    for r, k in enumerate(keys):
        if V[r, c] <= thr:
            continue
        if all(V[idx[s], c] <= thr for n in range(1, len(k)) for s in itertools.combinations(k, n)):
            out.append((k, float(V[r, c] - thr)))
    return out


_CACHE = {}


def load_group(tag):
    if tag in _CACHE:
        return _CACHE[tag]
    rel, ef = GROUPS[tag]; O = os.path.join(REPO, rel)
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    z = np.load(os.path.join(O, "D.npz")); keys_all = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
    Vall = np.load(os.path.join(O, "V_official.npy"))
    s3 = np.array([len(k) <= 3 for k in keys_all]); keys = [k for k, t in zip(keys_all, s3) if t]
    e = np.load(os.path.join(EST_DIR, ef)); assert np.array_equal(e["sel"], s3), tag
    V = closure(Vall[s3], keys); Vh = closure(e["E"], keys)
    P = len(spec["cand"]); C = V.shape[1]; idx = {k: r for r, k in enumerate(keys)}
    # 边际表：背景为规模 ≤2 的集合（含空集），D[i, b, c] = V(T∪i) − V(T)
    bgs = [()] + [k for k in keys if len(k) <= 2]
    D = np.full((P, len(bgs), C), -np.inf); mem = np.zeros((len(bgs), P), bool)
    for b, T in enumerate(bgs):
        mem[b, list(T)] = True
        vT = V[idx[T]] if T else np.zeros(C)
        for i in range(P):
            if i not in T:
                D[i, b] = V[idx[tuple(sorted(T + (i,)))]] - vT
    g = dict(tag=tag, spec=spec, keys=keys, idx=idx, V=V, Vh=Vh, P=P, C=C, bgs=bgs, D=D, mem=mem,
             M0=np.stack([V[idx[(i,)]] for i in range(P)]))
    _CACHE[tag] = g
    return g


class Instance:
    """一个定级实例：规则等级 l0、代价 w、锁定标记、目标等级 tl、越线组合列表。"""

    def __init__(self, g, tau, rng, p1=0.7, target="all3", cost="uniform", lock=0.0, viol_sets=None):
        self.g, self.tau = g, tau; P, C = g["P"], g["C"]
        self.l0 = np.where(rng.random(P) < p1, 1, 2)
        self.w = np.ones(P) if cost == "uniform" else rng.uniform(0.5, 2.0, P)
        self.locked = np.zeros(P, bool)
        if lock > 0:
            one = np.where(self.l0 == 1)[0]; self.locked[rng.choice(one, max(1, int(round(lock * len(one)))), replace=False)] = True
        self.tl = np.full(C, TOP) if target == "all3" else rng.integers(2, TOP + 1, C)
        self.cfg = dict(p1=p1, target=target, cost=cost, lock=lock)
        self.set_viol(viol_sets if viol_sets is not None else [minimal_sets(g["V"], g["keys"], c, tau) for c in range(C)])

    def set_viol(self, sets_per_target):
        """sets_per_target[c] = [(成员, 超出量)]。只保留规则等级下确实越线的（成员全部低于目标等级）。"""
        A, gl, ex, tc = [], [], [], []; self.base_leak = 0
        for c, sets in enumerate(sets_per_target):
            for m, e in sets:
                if self.l0[list(m)].max() >= self.tl[c]:
                    continue
                if self.locked[list(m)].all():
                    self.base_leak += 1; continue
                row = np.zeros(self.g["P"], bool); row[list(m)] = True
                A.append(row); gl.append(self.tl[c]); ex.append(e); tc.append(c)
        self.A = np.array(A, bool).reshape(-1, self.g["P"]); self.gl = np.array(gl, int); self.ex = np.array(ex); self.tc = np.array(tc, int)

    # ---- 状态查询
    def unresolved(self, l):
        if len(self.gl) == 0:
            return np.zeros(0, bool)
        return np.where(self.A, l[None, :], 0).max(1) < self.gl

    def cost(self, l):
        return float((self.w * (l - self.l0)).sum())

    def m_cur(self, l):
        """当前等级下各字段的 M^(K)：对每个目标 c，背景池 = 等级低于目标等级的字段。返回 (P, C)。"""
        g = self.g; out = np.zeros((g["P"], g["C"]))
        for c in range(g["C"]):
            pool = l < self.tl[c]
            ok = ~(g["mem"] & ~pool[None, :]).any(1)
            out[:, c] = np.where(pool, g["D"][:, ok, c].max(1), 0.0)
        return np.maximum(out, 0)


# ---------------------------------------------------------------- 基线与最优
def all_raise(ins):
    l = ins.l0.copy()
    for a, gl in zip(ins.A, ins.gl):
        m = a & ~ins.locked
        l[m] = np.maximum(l[m], gl)
    return l


def _options(ins, l, un):
    """可选动作 (字段, 目标等级, 消除数, 代价)。"""
    out = []
    for i in np.where(~ins.locked & (l < TOP) & ins.A[un].any(0))[0]:
        for L in sorted(set(ins.gl[un & ins.A[:, i]])):
            if L > l[i]:
                hit = int((un & ins.A[:, i] & (ins.gl <= L)).sum())
                out.append((i, int(L), hit, ins.w[i] * (L - l[i])))
    return out


def greedy_cover(ins):
    l = ins.l0.copy()
    while True:
        un = ins.unresolved(l)
        if not un.any():
            return l
        i, L, _, _ = max(_options(ins, l, un), key=lambda o: (o[2] / o[3], -o[3]))
        l[i] = L


def greedy_M(ins):
    """方案里描述的做法：每步上调当前背景池下 M^(K) 最大的字段到所需等级，然后重算。"""
    l = ins.l0.copy()
    while True:
        un = ins.unresolved(l)
        if not un.any():
            return l
        M = ins.m_cur(l); best = None
        for i, L, _, _ in _options(ins, l, un):
            cs = np.unique(ins.tc[un & ins.A[:, i] & (ins.gl <= L)])
            s = M[i, cs].max()
            if best is None or s > best[0]:
                best = (s, i, L)
        l[best[1]] = best[2]


def prune(ins, l):
    l = l.copy()
    for i in sorted(np.where(l > ins.l0)[0], key=lambda i: -ins.w[i] * (l[i] - ins.l0[i])):
        while l[i] > ins.l0[i]:
            l[i] -= 1
            if ins.unresolved(l).any():
                l[i] += 1; break
    return l


def optimal(ins):
    """整数规划：x[i,L]=1 表示字段 i 的等级 ≥ L（L=2,3 且 L>l0_i）。"""
    P = ins.g["P"]; var = {}
    for i in range(P):
        if not ins.locked[i]:
            for L in range(ins.l0[i] + 1, TOP + 1):
                var[(i, L)] = len(var)
    if len(ins.gl) == 0 or not var:
        return ins.l0.copy()
    n = len(var); cst = np.zeros(n)
    for (i, L), j in var.items():
        cst[j] = ins.w[i]
    rows, lb = [], []
    for a, gl in zip(ins.A, ins.gl):
        r = np.zeros(n)
        for i in np.where(a)[0]:
            if (i, gl) in var:
                r[var[(i, gl)]] = 1
        rows.append(r); lb.append(1)
    for (i, L), j in var.items():           # 单调：≥3 蕴含 ≥2
        if (i, L - 1) in var:
            r = np.zeros(n); r[var[(i, L - 1)]] = 1; r[j] = -1; rows.append(r); lb.append(0)
    res = milp(c=cst, constraints=LinearConstraint(np.array(rows), lb=np.array(lb)), integrality=np.ones(n), bounds=Bounds(0, 1))
    assert res.success, res.message
    l = ins.l0.copy()
    for (i, L), j in var.items():
        if res.x[j] > 0.5:
            l[i] = max(l[i], L)
    return l


def sample_cfg(rng):
    return dict(p1=float(rng.choice([0.7, 0.5])), target=str(rng.choice(["all3", "mixed"])), cost=str(rng.choice(["uniform", "nonuniform"])))


SETTINGS = [dict(p1=p1, target=t, cost=c) for p1 in (0.7, 0.5) for t in ("all3", "mixed") for c in ("uniform", "nonuniform")]
