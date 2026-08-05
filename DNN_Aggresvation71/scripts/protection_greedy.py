# -*- coding: utf-8 -*-
"""DNN71 Part A：最小防护集 = 协同图上的加权覆盖，贪心 + 次模性 + (1-1/e) 证书。

问题：数据持有方要共享一般字段，需删除一个防护集 P，使剩余可共享内容对所有机密字段
的推断泄露被控制。由 70 号"泄露排序低阶决定"，用现成的完备字段对真值把危险定义在
低阶结构上：
  - 危险单字段 i：max_c v_c({i}) > τ   （单独就泄露，强制删）
  - 危险字段对 (i,j)：max_c v_c({i,j}) > τ
删除字段 i 可"中和"所有以 i 为端点的危险单/对。这是协同图上的加权覆盖（顶点覆盖）问题。

覆盖收益 f(P) = Σ_{被 P 中和的危险单/对} w   （w = 泄露值）——单调次模。
贪心在预算 |P|≤k 下最大化 f(P) 有 (1-1/e) 保证。本脚本：贪心/基线/次模检验/小预算暴力最优。

全部用 69 号现成重训真值，不重训。
"""
import json, itertools, os
import numpy as np
import pandas as pd

SRC = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69/outputs"
OUT = "/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation71/outputs"

reg = json.load(open(f"{SRC}/subsets.json"))
tl = pd.read_csv(f"{SRC}/truth_long.csv")
truth = {(r.sid, r.conf): r.dnn for r in tl.itertuples()}
CONFS = sorted(tl.conf.unique())

# ---- 完备低阶真值：max over conf ----
single_lk, pair_lk = {}, {}
fields = set()
for sid, m in reg.items():
    if m["group"] == "single":
        f = m["fields"][0]; fields.add(f)
        single_lk[f] = max(truth.get((sid, c), 0.0) for c in CONFS)
    elif m["group"] == "pair":
        fs = tuple(sorted(m["fields"]))
        pair_lk[fs] = max(truth.get((sid, c), 0.0) for c in CONFS)
FIELDS = sorted(fields)
print(f"一般字段 {len(FIELDS)}，字段对 {len(pair_lk)}")


def build_instance(tau):
    """危险项：单字段(自覆盖)与字段对(端点覆盖)，权重=泄露值。"""
    dsingle = {f: v for f, v in single_lk.items() if v > tau}
    dpair = {fs: v for fs, v in pair_lk.items() if v > tau}
    return dsingle, dpair


def coverage(P, dsingle, dpair):
    """f(P)：被 P 中和的危险项总权重。"""
    P = set(P)
    s = sum(v for f, v in dsingle.items() if f in P)
    s += sum(v for (a, b), v in dpair.items() if a in P or b in P)
    return s


def total_weight(dsingle, dpair):
    return sum(dsingle.values()) + sum(dpair.values())


def worst_remaining_pair(P, dpair_all):
    """删除 P 后，剩余字段对的最大泄露（真值口径下的最坏对）。"""
    P = set(P)
    rem = [v for (a, b), v in dpair_all.items() if a not in P and b not in P]
    return max(rem) if rem else 0.0


def greedy(dsingle, dpair, budget):
    """贪心最大覆盖，返回每步选择与累计覆盖。"""
    P, curve = [], []
    remaining = set(FIELDS)
    fcur = 0.0
    for _ in range(budget):
        best_f, best_gain = None, -1
        for f in remaining:
            g = coverage(P + [f], dsingle, dpair) - fcur
            if g > best_gain:
                best_gain, best_f = g, f
        P.append(best_f); remaining.discard(best_f)
        fcur = coverage(P, dsingle, dpair)
        curve.append((best_f, fcur, best_gain))
    return P, curve


def baseline_degree(dsingle, dpair, budget):
    deg = {f: 0.0 for f in FIELDS}
    for f, v in dsingle.items(): deg[f] += v
    for (a, b), v in dpair.items(): deg[a] += v; deg[b] += v
    order = sorted(FIELDS, key=lambda f: -deg[f])
    return order[:budget]


def baseline_single_leak(budget):
    return sorted(FIELDS, key=lambda f: -single_lk[f])[:budget]


def baseline_random(budget, seed):
    rng = np.random.RandomState(seed)
    return list(rng.choice(FIELDS, budget, replace=False))


# ================= 主流程 =================
results = {}
for tau in [0.5, 0.6, 0.7]:
    dsingle, dpair = build_instance(tau)
    W = total_weight(dsingle, dpair)
    dpair_all = {fs: v for fs, v in pair_lk.items()}  # 用于最坏剩余对（全部对，不只危险的）
    maxk = 44
    # 贪心全程
    Pg, curve = greedy(dsingle, dpair, maxk)
    # 各预算下覆盖率 + 最坏剩余对 + 基线
    rows = []
    for k in range(0, maxk + 1):
        Pk = Pg[:k]
        fg = coverage(Pk, dsingle, dpair)
        # 基线覆盖
        fdeg = coverage(baseline_degree(dsingle, dpair, k), dsingle, dpair)
        fsl = coverage(baseline_single_leak(k), dsingle, dpair)
        frand = np.mean([coverage(baseline_random(k, s), dsingle, dpair) for s in range(20)]) if k > 0 else 0.0
        rows.append(dict(tau=tau, k=k,
                         greedy_cov=fg / W if W else 0, degree_cov=fdeg / W if W else 0,
                         single_cov=fsl / W if W else 0, random_cov=frand / W if W else 0,
                         greedy_worst_pair=worst_remaining_pair(Pk, dpair_all),
                         n_dpair_remaining=sum(1 for (a, b), v in dpair.items() if a not in set(Pk) and b not in set(Pk))))
    rows = pd.DataFrame(rows)
    results[tau] = rows
    # k* = 覆盖所有危险项所需最小预算
    kstar = int(rows[rows.greedy_cov >= 0.999].k.min()) if (rows.greedy_cov >= 0.999).any() else maxk
    print(f"\n=== τ={tau}: 危险单字段 {len(dsingle)}, 危险对 {len(dpair)}, 总权重 {W:.1f} ===")
    print(f"贪心覆盖全部危险项所需 k* = {kstar} / {len(FIELDS)}（即最小防护集大小）")
    print(f"贪心前 {kstar} 步选择: {Pg[:kstar]}")

pd.concat(results.values()).to_csv(f"{OUT}/protection_curves.csv", index=False)

# ---- 次模性经验检验：边际收益随 |P| 递减 ----
dsingle, dpair = build_instance(0.5)
Pg, _ = greedy(dsingle, dpair, 44)
marg = []
rng = np.random.RandomState(0)
for trial in range(200):
    k = rng.randint(0, 20)
    P = list(rng.choice(FIELDS, k, replace=False))
    rest = [f for f in FIELDS if f not in P]
    f0 = coverage(P, dsingle, dpair)
    fadd = rng.choice(rest)
    gain = coverage(P + [fadd], dsingle, dpair) - f0
    marg.append((k, gain))
marg = pd.DataFrame(marg, columns=["Psize", "marginal_gain"])
marg.to_csv(f"{OUT}/submodularity_check.csv", index=False)
corr = marg.Psize.corr(marg.marginal_gain)
print(f"\n=== 次模性检验（τ=0.5）：|P| 与边际收益相关 = {corr:.3f}（应显著负，即递减收益）===")
print(marg.groupby(pd.cut(marg.Psize, [-1, 4, 9, 14, 19])).marginal_gain.mean())

# ---- 小预算暴力最优 vs 贪心（验证 (1-1/e) 且实测更紧）----
print("\n=== 贪心 vs 暴力最优覆盖（τ=0.5，验证近最优）===")
rows = []
for k in [1, 2, 3, 4]:
    best = 0.0
    for combo in itertools.combinations(FIELDS, k):
        f = coverage(combo, dsingle, dpair)
        if f > best: best = f
    gk = coverage(Pg[:k], dsingle, dpair)
    rows.append(dict(k=k, greedy=gk, optimal=best, ratio=gk / best, bound_1_1e=0.632))
    print(f"  k={k}: 贪心 {gk:.2f}, 最优 {best:.2f}, 比值 {gk/best:.4f} (>= 0.632 保证)")
pd.DataFrame(rows).to_csv(f"{OUT}/greedy_vs_optimal.csv", index=False)
print("\nPart A 完成。")
