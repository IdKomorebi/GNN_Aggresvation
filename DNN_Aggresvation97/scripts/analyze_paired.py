# -*- coding: utf-8 -*-
"""配对认证协议的判读。判据在跑之前就写死在这里，避免看到结果再挑标准。

闸门（k=5）：新协议必须复现 93 号认证为真的 10 个协同。
  通过条件（预注册）：配对口径下 syn>0.10 的个数 >= 8（允许边缘 2 个掉出），
  且与旧口径的 Spearman >= 0.8（排序未被改坏）。

判决（k=6）：top 组配对认证显著高于对照组。
  通过条件（预注册）：Mann-Whitney 单侧 p<0.05 且 AUC>0.7。
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.subtract.outer(pos, neg) > 0).mean()
                 + 0.5 * (np.subtract.outer(pos, neg) == 0).mean())


def load(tag):
    fs = sorted(glob.glob(f"outputs/paired_o*{tag}_s*of*.csv"))
    assert fs, f"没有 {tag} 的结果"
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def gate():
    d = load("_gate")
    print(f"=== k=5 闸门（{len(d)} 个集合，R={int(d.n_seed.iloc[0])} 种子）===")
    n_old = int((d.syn_old_protocol > 0.10).sum())
    n_new = int((d.syn_paired > 0.10).sum())
    n_naive = int((d.syn_naive > 0.10).sum())
    rho = spearmanr(d.syn_old_protocol, d.syn_paired).statistic
    print(f"  旧协议(93 号原值)  >0.10: {n_old}/{len(d)}   均值 {d.syn_old_protocol.mean():.4f}")
    print(f"  本次旧口径复算      >0.10: {n_naive}/{len(d)}   均值 {d.syn_naive.mean():.4f}"
          f"   种子间 sd 均值 {d.syn_naive_sd.mean():.4f}")
    print(f"  ★配对口径          >0.10: {n_new}/{len(d)}   均值 {d.syn_paired.mean():.4f}"
          f"   种子间 sd 均值 {d.syn_paired_sd.mean():.4f}")
    print(f"  排序一致性 ρ(旧, 配对) = {rho:.3f}")
    ok = n_new >= 8 and rho >= 0.8
    print("\n★闸门：", "通过——可用于 k=6 判决" if ok
          else "未通过（预注册要求 >0.10 至少 8 个且 ρ>=0.8）⟹ 停下查错，不得用于 k=6")
    return ok


def verdict(tag="_k6old"):
    d = load(tag)
    s, c = d[d.group == "strong"], d[d.group != "strong"]
    print(f"\n=== k=6 判决（top {len(s)} / 对照 {len(c)}，R={int(d.n_seed.iloc[0])} 种子）===")
    print(f"{'口径':<12}{'top均值':>10}{'对照均值':>10}{'top最强':>10}"
          f"{'>0.10':>8}{'MW p':>9}{'AUC':>7}")
    for name, col in [("旧口径", "syn_naive"), ("★配对", "syn_paired")]:
        p = mannwhitneyu(s[col], c[col], alternative="greater").pvalue
        a = auc(s[col].to_numpy(), c[col].to_numpy())
        print(f"{name:<12}{s[col].mean():>10.4f}{c[col].mean():>10.4f}"
              f"{s[col].max():>10.4f}{int((s[col] > 0.10).sum()):>8}{p:>9.4f}{a:>7.3f}")
    print(f"\n噪声底（对照组）：旧口径均值 {c.syn_naive.mean():.4f} → "
          f"配对 {c.syn_paired.mean():.4f}"
          f"（下降 {(1 - c.syn_paired.mean() / max(c.syn_naive.mean(), 1e-9)) * 100:.0f}%）")
    p = mannwhitneyu(s.syn_paired, c.syn_paired, alternative="greater").pvalue
    a = auc(s.syn_paired.to_numpy(), c.syn_paired.to_numpy())
    ok = p < 0.05 and a > 0.7
    print("\n★判决：", "k*=6 成立" if ok else
          "仍不可认证 ⟹ k*=5 定稿（降噪声底也救不回来，说明六阶确实没有可分辨的强协同）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        gate()
    else:
        gate()
        verdict()
