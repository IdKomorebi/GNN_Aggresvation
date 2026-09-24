# -*- coding: utf-8 -*-
"""113 号：PJM / CAISO 敏感目标按《电力市场信息披露基本规则》披露时点重新定义，两种候选口径对照。

目标（运行日次日才披露的实际运行信息，附表 6.57/6.61/6.65）：
  PJM   metered_load_mw（实际负荷）、total_gen（发电总出力）、net_actual_interchange_mw（联络线实际交换）
  CAISO actual_load__mw__ca_iso_tac（实际负荷）
口径 A（全候选）：41 个候选全部保留；仅推断 total_gen 时剔除 19 个分燃料实际发电（其和即总出力，属恒等式）。
口径 B（按披露时点）：剔除同样"次日才披露"的字段——实际分燃料发电、实际新能源出力、其他平衡区实际负荷、
  实际/无意交换、事后结算补偿；只保留出清前后及时披露的预测、价格、辅助服务需求与出清、计划交换。

不重训：两种口径下规模 ≤3 的集合都是 41 字段枚举的子集，直接取 103 号正式三攻击器真值与 L0ensx 估计。
"""
import os, sys, json, pickle, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src"))
import pipe  # noqa: E402

A = os.path.join(ROOT, "outputs", "analysis"); os.makedirs(A, exist_ok=True)
FUEL = dict(coal="煤", gas="气", hydro="水", multiple_fuels="多燃料", nuclear="核", oil="油",
            other_renewables="其他可再生", solar="光", storage="储能", wind="风")
PJM_LAB = {"forecast_load_mw_latest_available": "最新负荷预测", "forecast_load_mw_day_ahead": "日前负荷预测",
           "rmccp": "调频容量价", "rmpcp": "调频性能价", "total_pjm_loc_credit": "调频机会成本补偿",
           "total_pjm_reg_purchases": "调频购买量", "total_pjm_self_sched_reg": "自调度调频量",
           "total_pjm_assigned_reg": "指派调频量", "total_pjm_rmccp_cr": "调频容量补偿", "total_pjm_rmpcp_cr": "调频性能补偿",
           "da_as_mcp_primary_reserve": "日前一级备用价", "da_as_mcp_synchronized_reserve": "日前同步备用价",
           "da_as_as_req_mw_primary_reserve": "一级备用需求", "da_as_as_req_mw_thirty_minutes_reserve": "30分钟备用需求",
           "da_as_ss_mw_primary_reserve": "一级备用自供量", "da_as_nsr_mw_primary_reserve": "一级非同步备用量",
           "system_energy_price_da": "日前系统电能价", "total_lmp_rt": "实时LMP", "marginal_loss_price_rt": "实时网损价",
           "net_inadv_interchange_mw": "净无意交换", "gross_sched_interchange_mw": "计划交换", "gross_inadv_interchange_mw": "总无意交换",
           "metered_load_mw": "实际计量负荷", "total_gen": "发电总出力", "net_actual_interchange_mw": "联络线净实际交换"}


def lab(ds, f):
    if ds == "pjm":
        if f.startswith("gen_fuel_"):
            body = f[len("gen_fuel_"):]; kind = "实际发电" if body.endswith("_mw") else "发电占比"
            return f"{kind}_{FUEL[body.rsplit('_', 1)[0]]}"
        return PJM_LAB.get(f, f)
    rep = [("dam_load_forecast__mw__", "日前负荷预测_"), ("actual_load__mw__", "实际负荷_"),
           ("actual_renewable_generation__mw__", "实际新能源_"), ("dam_renewable_forecast__mw__", "日前新能源预测_"),
           ("dam_lmp__energy_usd_per_mwh__th_", "日前电能价_"), ("dam_lmp__lmp_usd_per_mwh__th_", "日前LMP_"),
           ("rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_", "实时LMP_"), ("rt15_lmp_hourly_mean__loss_usd_per_mwh__th_", "实时网损价_"),
           ("dam_schedule__mw__load__tac_", "日前负荷计划_"), ("rtpd_eim_transfer__mw__", "EIM交换_"),
           ("dam_as_requirement__mw__as_caiso_exp__regulation_mileage_", "调频里程需求_"),
           ("dam_as_requirement__mw__as_caiso__", "辅助服务需求_")]
    s = f
    for a, b in rep:
        s = s.replace(a, b)
    return (s.replace("_gen_apnd", "").replace("__minimum", "").replace("ca_iso", "CAISO")
            .replace("__", "_").replace("_tac", ""))


TARGETS = {"pjm": ["metered_load_mw", "total_gen", "net_actual_interchange_mw"], "caiso": ["actual_load__mw__ca_iso_tac"]}
TARGET_RULE = {"metered_load_mw": "附表 6.61 实际负荷", "total_gen": "附表 6.57 发电总出力",
               "net_actual_interchange_mw": "附表 6.65 省间联络线输电情况", "actual_load__mw__ca_iso_tac": "附表 6.61 实际负荷"}


def next_day_fields(ds, names):
    """口径 B 要剔除的"次日才披露"字段。"""
    out = []
    for f in names:
        if ds == "pjm" and (f.startswith("gen_fuel_") or "inadv" in f or f in
                            ("total_pjm_loc_credit", "total_pjm_rmccp_cr", "total_pjm_rmpcp_cr")):
            out.append(f)
        if ds == "caiso" and (f.startswith("actual_load__") or f.startswith("actual_renewable_generation")
                              or f.startswith("rtpd_eim_transfer")):
            out.append(f)
    return out


def shards(pat, n):
    fs = sorted(glob.glob(pat)); out = np.zeros((n, 12), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z["v"]
    return out


rows, prof_rows, excl_rows = [], [], []
cache = {}
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(os.path.join(REPO, f"DNN_Aggresvation100/outputs/sets/{ds}_meta.pkl"), "rb"))
    keys, act, gen, conf = meta["keys_k3"], meta["active"], meta["general"], meta["conf"]
    Vall = np.load(os.path.join(REPO, f"DNN_Aggresvation103/outputs/analysis/{ds}_truth_official.npz"))["V"]
    Eall = pipe.closure_max(shards(os.path.join(REPO, f"DNN_Aggresvation103/outputs/est/{ds}_k3_L0ensx_s*of3.npz"), len(keys)), keys)
    names = [gen[i] for i in act]
    nd = set(next_day_fields(ds, names))
    for f in names:
        excl_rows.append(dict(数据集=ds.upper(), 字段=f, 中文名=lab(ds, f), 口径B是否剔除=("剔除（次日披露）" if f in nd else "保留")))
    # 单字段线性 r²（用 100 号口径的 train 数据）
    sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation100", "src"))
    from common100 import load  # noqa: E402
    Dd = load(ds); Xtr, Ytr = Dd["Xtr"], Dd["Ytr"]
    for t in TARGETS[ds]:
        c = conf.index(t)
        V, E = Vall[:, [c]], Eall[:, [c]]
        for var in ["A", "B"]:
            if var == "A":
                pool = [i for i in act if not (t == "total_gen" and gen[i].startswith("gen_fuel_"))]
            else:
                pool = [i for i in act if gen[i] not in nd]
            M, W, Dt, bks, bsz = pipe.m_table(V, keys, pool, 2)
            _, _, De, _, _ = pipe.m_table(E, keys, pool, 2)
            r2 = np.array([np.corrcoef(Xtr[:, i], Ytr[:, c])[0, 1] ** 2 for i in pool])
            rec = dict(数据集=ds.upper(), 目标=TARGET_RULE[t].split(" ", 1)[1] + f"（{t}）", 规则依据=TARGET_RULE[t],
                       口径=var, 候选数=len(pool), M0均值=float(M[:, 0, 0].mean()), M2均值=float(M[:, 2, 0].mean()),
                       最大M0=float(M[:, 0, 0].max()), 最大M2=float(M[:, 2, 0].max()),
                       背景升级均值=float((M[:, 2, 0] - M[:, 0, 0]).mean()),
                       K1饱和_M1除M2=float(M[:, 1, 0].sum() / max(M[:, 2, 0].sum(), 1e-9)))
            for tau in [0.5, 0.7, 0.9]:
                crit0 = np.array([V[keys.index((i,)), 0] > tau for i in pool])
                mus = pipe.mus_list(V, keys, pool, 0, tau, 3)
                crit2 = np.array([any(i in m for m in mus) for i in pool])
                single = {m[0] for m in mus if len(m) == 1}
                left = [m for m in mus if not set(m) & single]
                if mus:
                    Am = np.zeros((len(mus), len(pool))); pos = {i: k for k, i in enumerate(pool)}
                    for r_, m in enumerate(mus):
                        for i in m:
                            Am[r_, pos[i]] = 1
                    res = milp(c=np.ones(len(pool)), constraints=LinearConstraint(Am, lb=1),
                               integrality=np.ones(len(pool)), bounds=Bounds(0, 1))
                    hs = int(round(res.fun))
                else:
                    hs = 0
                rec[f"τ{tau}_单字段即危险"] = int(crit0.sum()); rec[f"τ{tau}_K2关键"] = int(crit2.sum())
                rec[f"τ{tau}_单看安全组合危险"] = int((crit2 & ~crit0).sum())
                rec[f"τ{tau}_危险小组合"] = len(mus); rec[f"τ{tau}_单字段定级后仍暴露"] = len(left)
                rec[f"τ{tau}_最少扣留"] = hs
            # 估计器保真（K=2）
            Mt, Me, L1, L3 = pipe.certify(Dt[:, :, 0], De[:, :, 0], bsz, 2)
            rec.update(估计M2_MAE=float(np.abs(Me - Mt).mean()), 认证下界比_top1=float(L1.sum() / Mt.sum()),
                       认证下界比_top3=float(L3.sum() / Mt.sum()),
                       排序Spearman=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")))
            rows.append(rec)
            for k, i in enumerate(pool):
                prof_rows.append(dict(数据集=ds.upper(), 目标=t, 口径=var, 字段=gen[i], 中文名=lab(ds, gen[i]),
                                      r2=float(r2[k]), M0=float(M[k, 0, 0]), M1=float(M[k, 1, 0]), M2=float(M[k, 2, 0]),
                                      估计M2=float(Me[k]), 见证=" + ".join(lab(ds, gen[j]) for j in W[k][2][0]) or "∅"))
            cache[(ds, t, var)] = dict(pool=pool, V=V, keys=keys, names=[lab(ds, gen[i]) for i in pool])
            print(f"{ds} {t} 口径{var}: 候选 {len(pool)}，M0 {rec['M0均值']:.3f} → M2 {rec['M2均值']:.3f}，"
                  f"τ0.7 单看安全组合危险 {rec['τ0.7_单看安全组合危险']}，最少扣留 {rec['τ0.7_最少扣留']}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(A, "113_summary.csv"), index=False)
pd.DataFrame(prof_rows).to_csv(os.path.join(A, "113_field_profile.csv"), index=False)
pd.DataFrame(excl_rows).to_csv(os.path.join(A, "113_candidate_roles.csv"), index=False)
# 增益矩阵（口径 B，按 M2 取前 12 个字段）
P = pd.DataFrame(prof_rows)
for (ds, t, var), cc in cache.items():
    if var != "B":
        continue
    top = P[(P.数据集 == ds.upper()) & (P.目标 == t) & (P.口径 == "B")].nlargest(12, "M2").字段.tolist()
    meta = pickle.load(open(os.path.join(REPO, f"DNN_Aggresvation100/outputs/sets/{ds}_meta.pkl"), "rb"))
    idxs = [meta["general"].index(f) for f in top]
    G = pipe.gain_matrix(cc["V"], cc["keys"], idxs)[:, :, 0]
    pd.DataFrame(G, index=[lab(ds, f) for f in top], columns=[lab(ds, f) for f in top]).to_csv(
        os.path.join(A, f"113_gain_matrix_{ds}_{t}_B.csv"))
print("完成")
