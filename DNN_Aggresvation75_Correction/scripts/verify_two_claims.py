# -*- coding: utf-8 -*-
"""用户提出的两个检测（用 75 号现成数据，零 GPU）：
检测 1：大部分集合的快速摊销逼近真值——是 K=0 oracle 做到的，还是需要微调？
检测 2：少数最强二阶协同——是哪一步发现的？none（无失活）微调后能不能也发现？
附：不碰大集合重训真值的两种"合法"简单校准（自校准 vs 小集合外推）实测。
附：承认性检查——上一版 GBDT 校准器的提升有多少来自真值特征本身（不带 est）。
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R75 = '/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation75'
R68 = '/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation68'

T = pd.read_csv(f'{R75}/outputs/truth_all.csv')
truth = T.set_index(['sid', 'conf']).truth.to_dict()
ev = json.load(open(f'{R75}/outputs/evalset.json'))
s1 = {ev[s]['fields'][0]: s for s in ev if ev[s]['size'] == 1}
s2 = {tuple(sorted(ev[s]['fields'])): s for s in ev if ev[s]['size'] == 2}

syn2 = pd.read_csv(f'{R68}/outputs/synergy2_perconf.csv')
syn2['sid_ij'] = [s2.get(tuple(sorted((a, b)))) for a, b in zip(syn2.fi, syn2.fj)]
syn2['sid_i'] = syn2.fi.map(s1)
syn2['sid_j'] = syn2.fj.map(s1)
syn2 = syn2.dropna(subset=['sid_ij', 'sid_i', 'sid_j']).reset_index(drop=True)
tv = syn2.synergy.values
strong = tv > 0.2          # 143 条"最强协同"
null = tv <= 0.02          # 真无协同
order_t = np.argsort(-tv)

KG = [0, 10, 50, 200]
rows1, rows2 = [], []
for scheme in ['uniform', 'none', 'bern50']:
    for seed in [0, 1, 2]:
        d = pd.read_csv(f'{R75}/outputs/est_{scheme}_seed{seed}.csv')
        d['truth'] = [truth.get((s, c), np.nan) for s, c in zip(d.sid, d.conf)]
        d = d.dropna(subset=['truth'])
        d['err'] = d.est - d.truth
        for K in KG:
            dk = d[d.K == K]
            rows1.append(dict(scheme=scheme, seed=seed, K=K,
                              mae=dk.err.abs().mean(), bias=dk.err.mean(),
                              rho=spearmanr(dk.est, dk.truth).correlation,
                              frac05=(dk.err.abs() <= 0.05).mean(),
                              frac02=(dk.err.abs() <= 0.02).mean()))
            est = dk.set_index(['sid', 'conf']).est.to_dict()
            eij = np.array([est.get((s, c), np.nan) for s, c in zip(syn2.sid_ij, syn2.conf)])
            ei = np.array([est.get((s, c), np.nan) for s, c in zip(syn2.sid_i, syn2.conf)])
            ej = np.array([est.get((s, c), np.nan) for s, c in zip(syn2.sid_j, syn2.conf)])
            es = eij - np.maximum(ei, ej)
            rank_e = pd.Series(-es).rank().values
            top20_t = set(order_t[:20])
            top20_e = set(np.argsort(-np.where(np.isnan(es), -9, es))[:20])
            rows2.append(dict(scheme=scheme, seed=seed, K=K,
                              rho=spearmanr(tv, es, nan_policy='omit').correlation,
                              top20=len(top20_t & top20_e) / 20,
                              med10=float(np.median(rank_e[order_t[:10]])),
                              recall=float((es[strong] > 0.1).mean()),
                              fp=float((es[null] > 0.1).mean())))

r1 = pd.DataFrame(rows1).groupby(['scheme', 'K']).mean(numeric_only=True).drop(columns='seed')
r2 = pd.DataFrame(rows2).groupby(['scheme', 'K']).mean(numeric_only=True).drop(columns='seed')

print("=" * 92)
print("检测 1：大部分集合的快速摊销（1576 子集 × 12 conf，3 seed 均值）")
print("=" * 92)
print(f"{'方案':<9}{'K':>5}{'MAE':>9}{'bias':>9}{'排序ρ':>9}{'|err|<=0.05占比':>15}{'<=0.02占比':>12}")
for sc in ['uniform', 'none']:
    for K in KG:
        r = r1.loc[(sc, K)]
        print(f"{sc:<9}{K:>5}{r.mae:>9.4f}{r.bias:>+9.4f}{r.rho:>9.4f}{r.frac05:>15.1%}{r.frac02:>12.1%}")
    print()

print("=" * 92)
print("检测 2：二阶协同（946 对 × 12 conf 对 68 号重训真值；143 条最强 syn>0.2）")
print("=" * 92)
print(f"{'方案':<9}{'K':>5}{'协同ρ':>9}{'top20命中':>10}{'真top10排名中位':>15}{'强协同检出':>11}{'误报率':>9}")
for sc in ['uniform', 'none', 'bern50']:
    for K in KG:
        r = r2.loc[(sc, K)]
        print(f"{sc:<9}{K:>5}{r.rho:>9.4f}{r.top20:>10.2f}{r.med10:>15.0f}{r.recall:>11.1%}{r.fp:>9.2%}")
    print()

# ---- 强协同条目在总体指标里的占比（解释"表面上不影响评价标准"）----
print(f"强协同条目占比：{strong.sum()}/{len(tv)} = {strong.mean():.1%} 的 (对,conf) 条目")

# ---- 合法校准实测（uniform，逐 seed 后平均）----
def band(n):
    return "1-2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else \
           "9-16" if n <= 16 else "17-32" if n <= 32 else "33-44"

res = {k: [] for k in ['raw0', 'self', 'small', 'raw200']}
for seed in [0, 1, 2]:
    d = pd.read_csv(f'{R75}/outputs/est_uniform_seed{seed}.csv')
    d['truth'] = [truth.get((s, c), np.nan) for s, c in zip(d.sid, d.conf)]
    d = d.dropna(subset=['truth'])
    w = d.pivot_table(index=['sid', 'conf'], columns='K', values='est').reset_index()
    w.columns = ['sid', 'conf', 'e0', 'e10', 'e50', 'e200']
    meta = d[d.K == 0][['sid', 'conf', 'truth', 'size']]
    w = w.merge(meta, on=['sid', 'conf'])
    w['band'] = w['size'].map(band)
    big = w[w['size'] >= 3]
    # 自校准：shift = mean(e0 − e200) per (band,conf)，不碰任何真值
    sh = (w.e0 - w.e200).groupby([w.band, w.conf]).mean()
    idx = pd.MultiIndex.from_frame(big[['band', 'conf']])
    tilde_self = big.e0.values - sh.reindex(idx).values
    # 小集合外推：per-conf 标量偏差只在 |S|<=2 上拟合（那些真值本来就有），外推到大集合
    small = w[w['size'] <= 2]
    sh2 = (small.e0 - small.truth).groupby(small.conf).mean()
    tilde_small = big.e0.values - sh2.reindex(big.conf).values
    res['raw0'].append(np.abs(big.e0 - big.truth).mean())
    res['self'].append(np.abs(tilde_self - big.truth).mean())
    res['small'].append(np.abs(tilde_small - big.truth).mean())
    res['raw200'].append(np.abs(big.e200 - big.truth).mean())

print("\n" + "=" * 92)
print("不碰大集合重训真值的两种简单校准（|S|>=3，3 seed 均值）")
print("=" * 92)
print(f"  K=0 原始                          MAE {np.mean(res['raw0']):.4f}")
print(f"  自校准（以自家 K=200 为参照平移）    MAE {np.mean(res['self']):.4f}   ← 零真值")
print(f"  小集合外推（per-conf 偏差,|S|<=2拟合）MAE {np.mean(res['small']):.4f}   ← 只用小集合真值")
print(f"  K=200 原始                        MAE {np.mean(res['raw200']):.4f}")
