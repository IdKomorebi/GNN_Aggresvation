# -*- coding: utf-8 -*-
"""D：直接重训 + 采样 Shapley 的预算曲线 vs 通用模型精确 Shapley。

在精确真值表 T 上模拟：若每个联盟都要重训一次，置换采样 / 成对 KernelSHAP 用 m 次重训能把 Shapley 估到多准。
横轴 = 不同联盟数（=重训次数），纵轴 = 与精确真值 Shapley 的 MAE（20 次重复均值）。
再列出各通用模型"精确枚举 2^p 次查询"得到的 Shapley MAE，读出盈亏平衡的重训次数。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import games as G
from analyze import load_truth, load_est, MODELS, OUT
rows = []
for game in ["gameB", "gameA"]:
    Tr = load_truth(game); T = (Tr[(0, "clean")] + Tr[(1, "clean")]) / 2; phiT = G.shapley(T)
    rng = np.random.RandomState(0)
    for npm in [1, 2, 4, 8, 16, 32, 64]:
        errs, ns = [], []
        for _ in range(20):
            e, n = G.perm_shapley(T, npm, rng); errs.append(np.abs(e - phiT).mean()); ns.append(n)
        rows.append(dict(game=game, method="perm_retrain", budget=np.mean(ns), phi_MAE=np.mean(errs)))
    for m in [32, 64, 128, 256, 512, 1024, 2048]:
        errs, ns = [], []
        for _ in range(20):
            e, n = G.kernel_shapley(T, m, rng); errs.append(np.abs(e - phiT).mean()); ns.append(n)
        rows.append(dict(game=game, method="kernel_retrain", budget=np.mean(ns), phi_MAE=np.mean(errs)))
    for mdl in dict.fromkeys(MODELS + ["rffx"]):
        E = load_est(game, mdl)
        if E is None: continue
        rows.append(dict(game=game, method=f"exact_{mdl}", budget=0, phi_MAE=np.abs(G.shapley(E) - phiT).mean()))
        # 误差分解：规模偏差部分 vs 字段特异部分；更紧的边际误差界
        e = E - T; p = int(np.log2(len(T))); sz = G.popcount(np.arange(2 ** p))
        esz = np.stack([e[sz == s].mean(0) for s in range(p + 1)])[sz]
        phi_e = G.shapley(e); phi_size = G.shapley(esz)
        b = np.arange(2 ** p); w = np.array([__import__("math").factorial(s) * __import__("math").factorial(p - s - 1) / __import__("math").factorial(p) for s in range(p)])
        epsD = np.stack([(w[sz[S]][:, None] * np.abs(e[S | (1 << i)] - e[S])).sum(0) for i in range(p) for S in [b[(b >> i) & 1 == 0]]])
        rows[-1].update(size_part_MAE=np.abs(phi_size).mean(), field_part_MAE=np.abs(phi_e - phi_size).mean(),
                        bound_2eps_mean=2 * np.abs(e).max(0).mean(), bound_margerr_mean=epsD.mean(), bound_margerr_max=epsD.max(0).mean(),
                        bound_holds_all=bool((np.abs(phi_e) <= epsD + 1e-9).all()),
                        actual_max=np.abs(phi_e).max(0).mean(), full_set_err=np.abs(e[-1]).mean())
df = pd.DataFrame(rows); df.to_csv(OUT / "D_budget_curve.csv", index=False)
print(df.to_markdown(index=False, floatfmt=".4f"))
