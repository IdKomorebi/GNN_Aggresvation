# -*- coding: utf-8 -*-
"""理论自查：在随机小型单调博弈（n=3/4）上暴力搜索命题的反例，并验证带基底版本的恒等式/不等式。

C1 "M_i^(n−1)=a_i（对所有 i）⇔ v 次模" 是否成立？——找"全部字段无增强，但 v 不次模"的单调博弈
C2 "k 阶协同强度 s ⇒ 全体成员 M^(k−1) ≥ s"：s 取 Möbius 红利 m(S) 时是否成立？取不可约增益 syn(S) 时是否成立？
C3 固定 τ 的最小余量 m_τ 是否可能严格小于 M（定理 1 只对"所有 τ"刻画 M）
C4 带基底 B：M^(0)=a^B、M^(1)=a^B+max(0,max_j I^B_ij)、预算 V(B∪A)−V(B) ≤ Σ M^(K|B)（|A|≤K+1）
C5 关键性 ⇔ 小最小不安全集合：非单调时是否失败（构造反例），单调时是否恒成立
"""
import itertools
import numpy as np

rng = np.random.RandomState(20260915)
out = []


def rand_monotone(n, integer=False):
    """随机单调博弈：先抽非负 Möbius 以外的任意值，再取子集最大闭包，v(∅)=0。"""
    V = rng.randint(0, 6, 2 ** n).astype(float) / 5 if integer else rng.rand(2 ** n)
    V[0] = 0
    for i in range(n):
        for S in range(2 ** n):
            if S >> i & 1:
                V[S] = max(V[S], V[S ^ (1 << i)])
    return V


def marg(V, i, T): return V[T | (1 << i)] - V[T]


def pc(x): return bin(x).count("1")


def M(V, n, i, K, base=0):
    return max(marg(V, i, T | base) for T in range(2 ** n) if not (T >> i & 1) and not (T & base) and pc(T) <= K)


def submodular(V, n):
    for i in range(n):
        for S in range(2 ** n):
            if S >> i & 1: continue
            for T in range(2 ** n):
                if (T & S) == S and not (T >> i & 1) and marg(V, i, T) > marg(V, i, S) + 1e-12:
                    return False
    return True


def mobius(V, n):
    m = V.copy()
    for i in range(n):
        for S in range(2 ** n):
            if S >> i & 1: m[S] -= m[S ^ (1 << i)]
    return m


# ---------------- C1
found = None
for trial in range(20000):
    n = 3; V = rand_monotone(n, integer=True)
    if all(abs(M(V, n, i, n - 1) - V[1 << i]) < 1e-12 for i in range(n)) and not submodular(V, n):
        found = V; break
out.append("C1 全部字段 M^(n−1)=a 但 v 非次模：" + ("找到反例 v=" + str(dict((format(S, "03b"), round(found[S], 2)) for S in range(8))) if found is not None else "未找到"))

# ---------------- C2
cm, cs = None, 0
for trial in range(20000):
    n = 3; V = rand_monotone(n); m = mobius(V, n); full = 2 ** n - 1
    minM = min(M(V, n, j, n - 1) for j in range(n))
    syn = V[full] - max(V[S] for S in range(full))
    if syn > minM + 1e-12: cs += 1
    if cm is None and m[full] > minM + 0.2: cm = (V.copy(), m[full], minM)
out.append(f"C2a 以不可约增益 syn(S) 为强度：20000 个随机单调 3 人博弈中违例 {cs} 个")
if cm is not None:
    V, mf, mm = cm
    out.append(f"C2b 以 Möbius 红利 m(S) 为强度：找到反例 m(123)={mf:.3f} > min_j M_j^(2)={mm:.3f}，v=" + str(dict((format(S, "03b"), round(V[S], 3)) for S in range(8))))
# 手工解析反例
V = np.array([0, .5, .5, .5, .5, .5, .5, 1.0]); m = mobius(V, 3)
out.append(f"C2c 手工反例：单字段均 0.5、两两 0.5（冗余）、三者 1.0 → m(123)={m[7]:.2f}，M_1^(2)={M(V,3,0,2):.2f}，syn(123)={V[7]-.5:.2f}")

# ---------------- C3
ex = None
for trial in range(5000):
    n = 3; V = rand_monotone(n); i = 0; K = 2; Mi = M(V, n, i, K)
    for tau in np.linspace(0.05, 0.95, 19):
        Ts = [T for T in range(2 ** n) if not (T >> i & 1)]
        cand = sorted(set([0.0] + [max(0.0, tau - V[T]) + 1e-12 for T in Ts]))
        mt = min(mm for mm in cand + [Mi] if all(V[T | 1 << i] <= tau + 1e-12 for T in Ts if V[T] <= tau - mm + 1e-15))
        if mt < Mi - 0.1:
            ex = (tau, mt, Mi); break
    if ex: break
out.append(f"C3a 随机搜索：τ={ex[0]:.2f} 时 m_τ={ex[1]:.3f} < M={ex[2]:.3f}（低阈值的平凡例）" if ex else "C3a 未找到")
# 手工非平凡例：v(i)=0.1, v(j)=0.5, v(ij)=0.9，K=1，τ=0.95
V = np.array([0, .1, .5, .9]); Mi = max(V[1] - V[0], V[3] - V[2])
ok = lambda mm, tau: all(V[T | 1] <= tau for T in [0, 2] if V[T] <= tau - mm)
mt = min(mm for mm in np.linspace(0, 1, 1001) if ok(mm, 0.95))
out.append(f"C3b 手工例 v(i)=0.1,v(j)=0.5,v(ij)=0.9,τ=0.95：M_i^(1)={Mi:.2f}，固定 τ 最小余量 m_τ={mt:.2f}；τ=0.7 时 m_τ={min(mm for mm in np.linspace(0,1,1001) if ok(mm,0.7)):.2f}")

# ---------------- C4
bad0 = bad1 = badb = 0
for trial in range(3000):
    n = 4; V = rand_monotone(n); B = 1 << 3; Hs = [0, 1, 2]
    for i in Hs:
        aB = V[B | 1 << i] - V[B]
        if abs(M(V, 3, i, 0, 0) - 0) > 1: pass
        M0 = max(marg(V, i, B | T) for T in [0]); bad0 += abs(M0 - aB) > 1e-12
        M1 = max(marg(V, i, B | T) for T in [0] + [1 << j for j in Hs if j != i])
        I = [V[B | 1 << i | 1 << j] - V[B | 1 << i] - V[B | 1 << j] + V[B] for j in Hs if j != i]
        bad1 += abs(M1 - (aB + max(0, max(I)))) > 1e-12
    for K in [0, 1, 2]:
        MK = {i: max(marg(V, i, B | T) for T in range(8) if not (T >> i & 1) and pc(T) <= K) for i in Hs}
        for A in range(1, 8):
            if pc(A) <= K + 1 and V[B | A] - V[B] > sum(MK[i] for i in Hs if A >> i & 1) + 1e-12: badb += 1
out.append(f"C4 带基底：M^(0)=a^B 违例 {bad0}；M^(1)=a^B+max(0,maxI^B) 违例 {bad1}；预算 V(B∪A)−V(B)≤ΣM^(K|B) 违例 {badb}（3000 个随机 4 人博弈，B=第 4 个字段）")

# ---------------- C5
def crit_vs_mus(V, n, tau, K):
    mis = 0
    for i in range(n):
        crit = any(V[T] <= tau < V[T | 1 << i] for T in range(2 ** n) if not T >> i & 1 and pc(T) <= K)
        mus = any(V[E] > tau and all(V[S] <= tau for S in range(2 ** n) if (S & E) == S and S != E)
                  for E in range(2 ** n) if E >> i & 1 and pc(E) <= K + 1)
        mis += crit != mus
    return mis
mono_bad = sum(crit_vs_mus(rand_monotone(4), 4, t, K) for _ in range(500) for t in [0.3, 0.6] for K in [1, 2])
# 非单调反例：位 0=字段1, 位 1=字段2, 位 2=字段3；v(1)=0.8 但 v(13)=0.2（加字段反而下降），v(123)=0.9
Vn = np.array([0, 0.8, 0.1, 0.9, 0.2, 0.2, 0.2, 0.9], dtype=float)
i = 1; T = 0b101
crit = Vn[T] <= 0.5 < Vn[T | 1 << i]
mus = [format(E, "03b") for E in range(8) if E >> i & 1 and Vn[E] > 0.5 and all(Vn[S] <= 0.5 for S in range(8) if (S & E) == S and S != E)]
out.append(f"C5 单调博弈上 关键性⇔小MUS 违例 {mono_bad}；非单调反例：字段2 在背景{{1,3}}上越过 τ=0.5（{crit}），但含字段2 的最小不安全集合 = {mus}（空）→ ⇒ 方向失败，单调性必要")

txt = "\n".join(out); print(txt)
open(__file__.replace("scripts/counterexamples.py", "outputs/counterexamples.txt"), "w").write(txt + "\n")
