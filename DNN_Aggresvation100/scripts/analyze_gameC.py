# -*- coding: utf-8 -*-
"""CAISO 副本/备用通道对照（gameC，14 玩家 2^14 联盟精确枚举）。
对照：总体 Shapley φ（真值上）、全量模型 SAGE φ（3 种子）、全集删除边际 LOO、单字段 a、M^(1)、M^(2)。
只回答两件事：model reliance ≠ potential inference capability；fair allocation ≠ disclosure hazard。
"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT.parent / "DNN_Aggresvation99/src"))
import mk as MK
from common100 import load

meta = json.load(open(ROOT / "outputs/sets/caiso_gameC_meta.json")); P = meta["players"]; p = len(P)
D = load("caiso"); conf = D["conf"]
fs = sorted(glob.glob(str(ROOT / "outputs/truth/caiso_gameC_seed0_s*of3.npz")))
V = np.zeros((2 ** p, 12), np.float32)
for f in fs:
    z = np.load(f); V[z["idx"]] = z["clean"]
V = np.clip(V, 0, 1); V[0] = 0; Vb = MK.envelope(V)
phi = MK.shapley(Vb); M, _ = MK.mk_all(Vb); full = 2 ** p - 1
loo = np.stack([Vb[full] - Vb[full ^ (1 << i)] for i in range(p)])
sz = np.load(A / "sage_caiso_gameC.npz"); sage = [MK.shapley(np.clip(sz[f"V_seed{s}"], 0, 1)) for s in range(3)]
short = lambda s: (s.replace("dam_load_forecast__mw__", "负荷预测_").replace("dam_schedule__mw__load__", "负荷计划_").replace("dam_lmp__", "日前")
                   .replace("rt15_lmp_hourly_mean__", "实时").replace("_usd_per_mwh__th_", "_").replace("_gen_apnd", "")
                   .replace("actual_renewable_generation__mw__", "光伏实际_").replace("dam_renewable_forecast__mw__", "光伏预测_")
                   .replace("dam_as_requirement__mw__as_caiso__", "需求_").replace("__minimum", ""))
rows = []
for c in range(12):
    for i in range(p):
        rows.append(dict(conf=conf[c], field=short(P[i]), a=M[i, 0, c], M1=M[i, 1, c], M2=M[i, 2, c], shapley=phi[i, c], loo=loo[i, c],
                         sage_s0=sage[0][i, c], sage_s1=sage[1][i, c], sage_s2=sage[2][i, c]))
R = pd.DataFrame(rows); R.to_csv(A / "caiso_gameC_fields.csv", index=False)
summ = []
for c in range(12):
    d = R[R.conf == conf[c]]
    if Vb[full, c] < 0.1: continue
    summ.append(dict(conf=conf[c][:40], V_full=Vb[full, c], Kendall_Shapley_vs_M1=kendalltau(d.shapley, d.M1)[0],
                     Kendall_SAGE_vs_Shapley=np.mean([kendalltau(d[f"sage_s{s}"], d.shapley)[0] for s in range(3)]),
                     Kendall_SAGE_vs_M1=np.mean([kendalltau(d[f"sage_s{s}"], d.M1)[0] for s in range(3)]),
                     SAGE种子间Kendall=np.mean([kendalltau(d.sage_s0, d.sage_s1)[0], kendalltau(d.sage_s0, d.sage_s2)[0]]),
                     强字段Shapley除以a=float((d.shapley / d.a)[d.a > 0.5].mean()) if (d.a > 0.5).any() else np.nan,
                     强字段LOO均值=float(d.loo[d.a > 0.5].mean()) if (d.a > 0.5).any() else np.nan))
S = pd.DataFrame(summ); S.to_csv(A / "caiso_gameC_summary.csv", index=False)
out = ["# CAISO gameC 对照（自动生成）\n", S.to_markdown(index=False, floatfmt=".3f")]
for tgt in ["actual_load__mw__ca_iso_tac", "dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd", "dam_schedule__mw__generation__caiso_totals"]:
    d = R[R.conf == tgt].sort_values("M1", ascending=False)
    d = d.assign(SAGE均值=d[["sage_s0", "sage_s1", "sage_s2"]].mean(1))
    out.append(f"\n## {tgt}\n\n" + d[["field", "a", "M1", "M2", "shapley", "loo", "SAGE均值"]].to_markdown(index=False, floatfmt=".3f"))
(A / "report_gameC.md").write_text("\n".join(out), encoding="utf-8"); print("\n".join(out))
