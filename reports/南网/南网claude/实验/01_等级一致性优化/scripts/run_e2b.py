# -*- coding: utf-8 -*-
"""E2b（补充）：验证闭环。E2 显示只靠估计值或一次性认证都有残余越线，这里把验证做成闭环：
  ① 表 X = 已验证集合用真值、未验证集合用估计值 + δ；按 X 求最优等级；
  ② 验证（=重训，这里查真值表）当前各目标背景池内所有未验证的组合；
  ③ 验证当前越线清单里未验证的组合（去掉估计造成的误报）；
  有新验证就回到 ①。结束时：池内组合全部验证过、清单全部为真 ⇒ 零残余，且代价等于真值最优。
指标：需要验证的集合占全部（≤3 字段）集合的比例、轮数。
另算一个不靠估计值的对照：从“什么都不知道”出发做同样的闭环（X 的未验证部分取 0）。"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(HERE, "src")); import env  # noqa: E402
OUT = os.path.join(HERE, "outputs")


def loop(g, tau, seed, cfg, prior):
    keys, V = g["keys"], g["V"]; C = g["C"]; n = len(keys)
    mem = np.zeros((n, g["P"]), bool)
    for r, k in enumerate(keys):
        mem[r, list(k)] = True
    known = np.zeros((n, C), bool); rounds = 0
    while True:
        rounds += 1
        X = env.closure(np.where(known, V, prior), keys)
        X = np.where(known, V, X)                                   # 已验证的不被闭包抬高
        sets = [env.minimal_sets(X, keys, c, tau) for c in range(C)]
        ins = env.Instance(g, tau, np.random.default_rng(seed), viol_sets=sets, **cfg)
        l = env.optimal(ins); new = np.zeros((n, C), bool)
        for c in range(C):
            pool = l < ins.tl[c]
            new[:, c] |= ~(mem & ~pool[None, :]).any(1) & ~known[:, c]          # 池内未验证的组合
            for m, _ in sets[c]:                                                 # 清单里未验证的组合
                r = g["idx"][m]
                if not known[r, c] and ins.l0[list(m)].max() < ins.tl[c]:
                    new[r, c] = True
        if not new.any():
            return ins, l, known, rounds
        known |= new


rows = []; t0 = time.time()
for tag in env.GROUPS:
    g = env.load_group(tag); n = len(g["keys"]) * g["C"]
    for tau in (0.5, 0.7):
        true_sets = [env.minimal_sets(g["V"], g["keys"], c, tau) for c in range(g["C"])]
        for si, cfg in enumerate(env.SETTINGS):
            for r in range(3):
                seed = 100000 * si + 1000 * int(tau * 10) + r
                it = env.Instance(g, tau, np.random.default_rng(seed), viol_sets=true_sets, **cfg)
                if len(it.gl) == 0:
                    continue
                co = it.cost(env.optimal(it))
                for name, prior in (("估计值+0.05", g["Vh"] + 0.05), ("估计值+0", g["Vh"]), ("不用估计值", np.zeros_like(g["V"]))):
                    ins, l, known, rounds = loop(g, tau, seed, cfg, prior)
                    assert not it.unresolved(l).any() and abs(it.cost(l) - co) < 1e-6, (tag, name)
                    rows.append(dict(组=tag, τ=tau, **cfg, 实例=r, 先验=name, 验证集合比例=float(known.sum() / n), 轮数=rounds,
                                     上调字段数=int((l > it.l0).sum()), 字段数=g["P"], 残余=0.0, 与真值最优之比=it.cost(l) / co))
        print("完成", tag, tau, "%.0fs" % (time.time() - t0), flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e2b_instances.csv"), index=False)
s = df.groupby("先验")[["验证集合比例", "轮数", "与真值最优之比"]].mean().round(4); s.to_csv(os.path.join(OUT, "e2b_summary.csv")); print(s.to_string())
g2 = df.groupby(["组", "τ", "先验"])[["验证集合比例", "轮数"]].mean().round(4); g2.to_csv(os.path.join(OUT, "e2b_summary_by_group.csv")); print(g2.to_string())
