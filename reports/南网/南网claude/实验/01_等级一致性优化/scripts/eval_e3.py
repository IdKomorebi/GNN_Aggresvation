# -*- coding: utf-8 -*-
"""E3：GRPO 与贪心在留出实例上的对比。用法：CUDA_VISIBLE_DEVICES=1 python scripts/eval_e3.py"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, torch
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(HERE, "src")); import env, grpo  # noqa: E402
OUT = os.path.join(HERE, "outputs"); torch.set_num_threads(1); dev = "cpu"   # 评测用 CPU 多进程（策略网络很小），每组一个进程
SEEN = ["RTS-GMLC", "PJM-load", "PJM-gen/ic"]; R = 10; NS = 16


def noisy_greedy(ins, rng, n):
    """同样采样预算的对照：贪心比值加随机扰动，采样 n 次（含一次原始贪心），各自回退后取最优。"""
    best = env.prune(ins, env.greedy_cover(ins))
    for _ in range(n - 1):
        l = ins.l0.copy()
        while True:
            un = ins.unresolved(l)
            if not un.any():
                break
            ops = env._options(ins, l, un); sc = np.array([o[2] / o[3] for o in ops]) * np.exp(rng.normal(0, 0.3, len(ops)))
            i, L, _, _ = ops[int(sc.argmax())]; l[i] = L
        l = env.prune(ins, l)
        if ins.cost(l) < ins.cost(best):
            best = l
    return best


def load_pols():
    pols = {}
    for s in (0, 1):
        f = os.path.join(OUT, "grpo", f"policy_seed{s}.pt")
        if os.path.exists(f):
            p = grpo.Policy().to(dev); p.load_state_dict(torch.load(f, map_location=dev)); p.eval(); pols[s] = p
    return pols


def eval_group(tag):
    torch.set_num_threads(1); pols = load_pols(); rows = []; g = env.load_group(tag)
    for tau in (0.5, 0.7):
        base = [env.minimal_sets(g["V"], g["keys"], c, tau) for c in range(g["C"])]
        for si, cfg in enumerate(env.SETTINGS):
            for r in range(R):
                seed = 7_000_000 + 100000 * si + 1000 * int(tau * 10) + r
                ins = env.Instance(g, tau, np.random.default_rng(seed), viol_sets=base, **cfg)
                if len(ins.gl) == 0:
                    continue
                co = ins.cost(env.optimal(ins)); rng = np.random.default_rng(seed + 1)
                sols = dict(all_raise=env.all_raise(ins), greedy_M_prune=env.prune(ins, env.greedy_M(ins)),
                            greedy_cover_prune=env.prune(ins, env.greedy_cover(ins)), noisy_greedy_x16=noisy_greedy(ins, rng, NS))
                for s, p in pols.items():
                    lg = grpo.rollout(ins, p, dev, greedy=True)[0]
                    sols[f"grpo_s{s}_raw"] = lg; sols[f"grpo_s{s}_prune"] = env.prune(ins, lg); sols[f"grpo_s{s}_x16"] = grpo.best_of(ins, p, dev, NS, rng)
                for m, l in sols.items():
                    assert not ins.unresolved(l).any()
                    rows.append(dict(组=tag, 训练时见过=tag in SEEN, τ=tau, **cfg, 实例=r, 方法=m, 代价=ins.cost(l), 与最优之比=ins.cost(l) / co if co > 0 else 1.0))
    print("完成", tag, flush=True)
    return rows


if __name__ == "__main__":
    dev = "cpu"   # 评测用 CPU 多进程（策略网络很小），每组一个进程
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(5) as pool:
        rows = sum(pool.map(eval_group, list(env.GROUPS)), [])
    pols = load_pols()
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e3_instances.csv"), index=False)
    df["方法族"] = df.方法.str.replace(r"_s[01]", "", regex=True)
    s = df.groupby(["训练时见过", "方法族"]).与最优之比.agg(["mean", "max"]).round(4)
    s["达到最优比例"] = df.groupby(["训练时见过", "方法族"]).与最优之比.apply(lambda x: float((x < 1 + 1e-9).mean())).round(4)
    s.to_csv(os.path.join(OUT, "e3_summary.csv")); print(s.to_string())
    df.groupby(["cost", "target", "方法族"]).与最优之比.mean().round(4).unstack().to_csv(os.path.join(OUT, "e3_summary_by_setting.csv"))
    df.groupby(["组", "方法"]).与最优之比.mean().round(4).unstack().to_csv(os.path.join(OUT, "e3_summary_by_group_seed.csv"))
    # 事先写定的判据
    key = ["组", "τ", "p1", "target", "cost", "实例"]; piv = df.pivot_table(index=key + ["训练时见过"], columns="方法", values="代价").reset_index()
    verdict = {}
    for s_ in pols:
        d = piv[f"grpo_s{s_}_prune"] / piv["greedy_cover_prune"]
        verdict[f"seed{s_}"] = dict(全部平均代价比=float(d.mean()), 见过的组=float(d[piv.训练时见过].mean()), 没见过的组=float(d[~piv.训练时见过].mean()),
                                  胜=float((d < 1 - 1e-9).mean()), 平=float((abs(d - 1) <= 1e-9).mean()), 负=float((d > 1 + 1e-9).mean()),
                                  x16对比同预算贪心=float((piv[f"grpo_s{s_}_x16"] / piv["noisy_greedy_x16"]).mean()))
        verdict[f"seed{s_}"]["不劣于贪心"] = bool(d.mean() <= 1.0 and d[~piv.训练时见过].mean() <= 1.01)
    json.dump(verdict, open(os.path.join(OUT, "e3_verdict.json"), "w"), ensure_ascii=False, indent=1); print(json.dumps(verdict, ensure_ascii=False, indent=1))
