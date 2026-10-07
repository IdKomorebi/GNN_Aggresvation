# -*- coding: utf-8 -*-
"""E1：全部上调 / 贪心 / 精确最优的代价对比。E2：用估计值代替真值优化后的残余越线，以及扫描—认证。
用法：python scripts/run_e1_e2.py   （纯 CPU，几分钟）"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(HERE, "src")); import env  # noqa: E402
OUT = os.path.join(HERE, "outputs"); R = 20; TAUS = (0.5, 0.7)
LOCK_SETTINGS = [dict(p1=0.7, target="all3", cost="uniform", lock=0.10), dict(p1=0.7, target="all3", cost="nonuniform", lock=0.10)]


def overgrade(ins, l, lopt):
    up = l > ins.l0
    return float((up & ~(lopt > ins.l0)).sum() / max(1, up.sum()))


def e1():
    rows = []
    for tag in env.GROUPS:
        g = env.load_group(tag)
        for tau in TAUS:
            base = [env.minimal_sets(g["V"], g["keys"], c, tau) for c in range(g["C"])]
            for si, cfg in enumerate(env.SETTINGS + LOCK_SETTINGS):
                for r in range(R):
                    rng = np.random.default_rng(100000 * si + 1000 * int(tau * 10) + r)
                    ins = env.Instance(g, tau, rng, viol_sets=base, **cfg)
                    if len(ins.gl) == 0:
                        continue
                    lo = env.optimal(ins); co = ins.cost(lo)
                    sols = dict(all_raise=env.all_raise(ins), greedy_M=env.greedy_M(ins), greedy_cover=env.greedy_cover(ins))
                    sols["greedy_M_prune"] = env.prune(ins, sols["greedy_M"]); sols["greedy_cover_prune"] = env.prune(ins, sols["greedy_cover"])
                    sols["optimal"] = lo
                    for m, l in sols.items():
                        assert not ins.unresolved(l).any(), (tag, m)
                        rows.append(dict(组=tag, τ=tau, **{k: v for k, v in ins.cfg.items()}, 实例=r, 方法=m, 字段数=g["P"], 越线组合数=len(ins.gl),
                                         基底已泄露=ins.base_leak, 代价=ins.cost(l), 上调字段数=int((l > ins.l0).sum()),
                                         与最优之比=ins.cost(l) / co if co > 0 else 1.0, 过度定级比例=overgrade(ins, l, lo)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e1_instances.csv"), index=False)
    main = df[df.lock == 0]
    ar = main[main.方法 == "all_raise"].set_index(["组", "τ", "p1", "target", "cost", "实例"]).代价
    main = main.join(ar.rename("全部上调代价"), on=["组", "τ", "p1", "target", "cost", "实例"]); main["相对全部上调节省"] = 1 - main.代价 / main.全部上调代价
    s1 = main.groupby("方法")[["代价", "上调字段数", "与最优之比", "过度定级比例", "相对全部上调节省"]].mean().round(4)
    s1["达到最优的实例比例"] = main.groupby("方法").与最优之比.apply(lambda x: float((x < 1 + 1e-9).mean())).round(4)
    s1.to_csv(os.path.join(OUT, "e1_summary_by_method.csv"))
    main.groupby(["组", "τ", "方法"])[["越线组合数", "代价", "上调字段数", "与最优之比", "相对全部上调节省"]].mean().round(4).to_csv(os.path.join(OUT, "e1_summary_by_group.csv"))
    main.groupby(["cost", "target", "方法"])[["与最优之比", "相对全部上调节省"]].mean().round(4).to_csv(os.path.join(OUT, "e1_summary_by_setting.csv"))
    lk = df[df.lock > 0]
    if len(lk):
        lk.groupby("方法")[["代价", "与最优之比", "基底已泄露"]].mean().round(4).to_csv(os.path.join(OUT, "e1_summary_lock10.csv"))
    print(s1.to_string())


def e2():
    rows = []
    for tag in env.GROUPS:
        g = env.load_group(tag); keys, V, Vh = g["keys"], g["V"], g["Vh"]; nset = len(keys)
        for tau in TAUS:
            true_sets = [env.minimal_sets(V, keys, c, tau) for c in range(g["C"])]
            for delta in (0.0, 0.05, 0.10):
                est_sets = [env.minimal_sets(Vh, keys, c, tau - delta) for c in range(g["C"])]
                # 扫描—认证：估计值超过 τ−δ 的集合全部“重训”（查真值），用真值重新求最小越线组合
                cert_sets, ncert = [], 0
                for c in range(g["C"]):
                    cand = Vh[:, c] > tau - delta; ncert += int(cand.sum())
                    Vc = np.where(cand, V[:, c], 0.0)[:, None]       # 未认证的集合视为未越线
                    cert_sets.append([(m, e) for m, e in env.minimal_sets(Vc, keys, 0, tau)])
                for si, cfg in enumerate(env.SETTINGS):
                    for r in range(10):
                        seed = 100000 * si + 1000 * int(tau * 10) + r
                        it = env.Instance(g, tau, np.random.default_rng(seed), viol_sets=true_sets, **cfg)
                        if len(it.gl) == 0:
                            continue
                        co = it.cost(env.optimal(it))
                        for name, sets in (("只用估计值", est_sets), ("扫描—认证", cert_sets)):
                            ie = env.Instance(g, tau, np.random.default_rng(seed), viol_sets=sets, **cfg)
                            assert np.array_equal(ie.l0, it.l0) and np.array_equal(ie.tl, it.tl)
                            for meth, f in (("optimal", env.optimal), ("greedy_cover_prune", lambda x: env.prune(x, env.greedy_cover(x)))):
                                l = f(ie); un = it.unresolved(l)
                                rows.append(dict(组=tag, τ=tau, δ=delta, 口径=name, 求解=meth, **cfg, 实例=r, 真越线组合数=len(it.gl),
                                                 残余比例=float(un.mean()), 有残余=bool(un.any()), 代价=it.cost(l), 与真值最优之比=it.cost(l) / co if co > 0 else 1.0,
                                                 认证集合数=ncert if name == "扫描—认证" else 0, 认证占全部集合比例=ncert / (nset * g["C"]) if name == "扫描—认证" else 0.0))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e2_instances.csv"), index=False)
    s = df.groupby(["口径", "δ", "求解"])[["残余比例", "有残余", "与真值最优之比", "认证占全部集合比例"]].mean().round(4)
    s.to_csv(os.path.join(OUT, "e2_summary.csv")); print(s.to_string())
    df.groupby(["组", "τ", "口径", "δ"])[["真越线组合数", "残余比例", "与真值最优之比", "认证占全部集合比例"]].mean().round(4).to_csv(os.path.join(OUT, "e2_summary_by_group.csv"))


if __name__ == "__main__":
    t = time.time(); e1(); print("E1 用时 %.1fs" % (time.time() - t)); t = time.time(); e2(); print("E2 用时 %.1fs" % (time.time() - t))
