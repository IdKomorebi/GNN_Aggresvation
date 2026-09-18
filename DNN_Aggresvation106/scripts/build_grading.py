# -*- coding: utf-8 -*-
"""P0-7 / RQ-B5：字段风险分级表与发布审查案例。
输入：104 号 M 全表（正式真值口径）+ 103 号正式真值 V̄。
输出：
 1) 分级表（每数据集选 1 个代表目标）：字段、M0/M1/M2、风险档位、见证、认证边际、是否 τ-critical；
 2) 发布审查案例：给定待发布集合 A，比较三种审查方式
    (a) 单字段相关性阈值放行、(b) 单字段推断能力放行、(c) M^(K) 逐阶预算放行，
    再用真值 V̄(A) 检查实际是否越过 τ。
"""
import sys, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100, R103, R104, R105 = (REPO / f"DNN_Aggresvation{n}" for n in (100, 103, 104, 105))
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import index_of
from runlog import log
A = ROOT / "outputs/analysis"; TAU = 0.7; GR = [0.05, 0.2, 0.5]; TIER = ["可忽略", "低", "中", "高"]
T = pd.read_csv(R104 / "outputs/analysis/M_table_official.csv")
CASE = {"pjm": "total_gen", "caiso": "actual_load__mw__ca_iso_tac"}
grade_rows, audit_rows = [], []
for ds, tgt in CASE.items():
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys, act, gen, conf = meta["keys_k3"], meta["active"], meta["general"], meta["conf"]
    Vt = np.load(R103 / f"outputs/analysis/{ds}_truth_official.npz")["V"]; idx = index_of(keys); c = conf.index(tgt)
    B = pd.read_csv(R105 / f"outputs/analysis/{ds}_field_baselines.csv"); B = B[B.目标 == tgt].set_index("字段")
    d = T[(T.数据集 == ds) & (T.目标 == tgt)].pivot_table(index="字段", columns="K",
        values=["exact_M", "est_M_集成", "exact_witness", "exact_V_T", "exact_V_Ti", "critical_τ0.7", "cert_L_集成"], aggfunc="first")
    for f in d.index:
        r = dict(数据集=ds, 目标=tgt, 字段=f,
                 相关性=float(B.loc[f, "pearson"]), M0=float(d[("exact_M", 0)][f]), M1=float(d[("exact_M", 1)][f]), M2=float(d[("exact_M", 2)][f]),
                 估计M2=float(d[("est_M_集成", 2)][f]), 认证下界=float(d[("cert_L_集成", 2)][f]),
                 风险档位=TIER[int(np.digitize(d[("exact_M", 2)][f], GR))], 见证背景=str(d[("exact_witness", 2)][f]),
                 见证下V=float(d[("exact_V_T", 2)][f]), 加入后V=float(d[("exact_V_Ti", 2)][f]),
                 critical=bool(d[("critical_τ0.7", 2)][f]), 单字段critical=bool(d[("critical_τ0.7", 0)][f]))
        grade_rows.append(r)
    # ---------- 发布审查案例：随机抽取待发布集合 A（规模 3），比较三种审查方式
    G = pd.DataFrame([g for g in grade_rows if g["数据集"] == ds and g["目标"] == tgt]).set_index("字段")
    import glob
    Ehat = np.zeros((len(keys), 12), np.float32)
    for fpath in sorted(glob.glob(str(R103 / f"outputs/est/{ds}_k3_L0ensx_s*of3.npz"))):
        z = np.load(fpath); Ehat[z["idx"]] = z["v"]
    Ehat = np.clip(Ehat, 0, 1)
    rng = np.random.RandomState(106); cands = [a for a in act]
    for trial in range(2000):
        A_idx = sorted(rng.choice(len(cands), 3, replace=False)); A_f = [gen[cands[j]] for j in A_idx]
        key = tuple(sorted(cands[j] for j in A_idx)); vA = float(Vt[idx[key], c])
        rule_corr = all(G.loc[f, "相关性"] <= 0.5 for f in A_f)                    # (a) 相关性阈值
        rule_single = all(G.loc[f, "M0"] <= TAU for f in A_f)                        # (b) 单字段推断能力
        rule_budget = sum(G.loc[f, "M2"] for f in A_f) <= TAU                        # (c) M^(2) 预算和（定理：|A|≤K+1 时为上界）
        rule_budget_est = sum(G.loc[f, "估计M2"] for f in A_f) <= TAU                # (c') 用估计 M
        # (d) 逐阶预算 U^(2)（99 号定理 1b）：第 t 个加入的字段只用 M^(t−1)，取最优顺序
        Ms = {f: [G.loc[f, "M0"], G.loc[f, "M1"], G.loc[f, "M2"]] for f in A_f}
        best = min(Ms[f1][0] + Ms[f2][1] + Ms[f3][2] for f1 in A_f for f2 in A_f for f3 in A_f if len({f1, f2, f3}) == 3)
        rule_stage = best <= TAU
        # (e) 通用模型直接估计集合风险
        rule_direct = float(Ehat[idx[key], c]) <= TAU
        audit_rows.append(dict(数据集=ds, 目标=tgt, 集合="+".join(x[:18] for x in A_f), 真值V=vA, 不安全=vA > TAU,
                               相关性放行=rule_corr, 单字段放行=rule_single, 预算放行=rule_budget, 预算放行_估计=rule_budget_est,
                               逐阶预算放行=rule_stage, 直接估计放行=rule_direct))
GR_df = pd.DataFrame(grade_rows); GR_df.to_csv(A / "grading_table.csv", index=False)
AU = pd.DataFrame(audit_rows); AU.to_csv(A / "release_audit.csv", index=False)
rows = []
for (ds, tgt), d in AU.groupby(["数据集", "目标"]):
    for rule in ["相关性放行", "单字段放行", "预算放行", "预算放行_估计", "逐阶预算放行", "直接估计放行"]:
        ok = d[rule]; uns = d.不安全
        rows.append(dict(数据集=ds, 目标=tgt, 审查规则=rule, 放行率=ok.mean(), 危险放行率=float((ok & uns).sum() / max(uns.sum(), 1)),
                         放行集合中不安全占比=float((ok & uns).sum() / max(ok.sum(), 1)), 误拒率=float((~ok & ~uns).sum() / max((~uns).sum(), 1)),
                         不安全集合数=int(uns.sum()), 集合总数=len(d)))
S = pd.DataFrame(rows); S.to_csv(A / "release_audit_summary.csv", index=False)
txt = "# 106 号：字段分级表与发布审查（RQ-B5）\n\n## 发布审查规则对比（随机 2,000 个规模 3 的待发布集合，τ=0.7）\n\n" + S.to_markdown(index=False, floatfmt=".4f")
(A / "report106.md").write_text(txt, encoding="utf-8"); log("GRADE", "DONE", "分级表与发布审查完成"); print(txt)
