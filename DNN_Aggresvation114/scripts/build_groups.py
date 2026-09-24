# -*- coding: utf-8 -*-
"""114 号 步骤 1：PJM / CAISO 现实口径——按披露时点收窄候选池、剔除目标副本，分组生成 D.npz 与 fields.json。

规则（《电力市场信息披露基本规则》国能发监管〔2024〕9号）：
  敏感目标 = 运行日次日才披露的实际运行信息（附表 6.57 发电总出力、6.61 实际负荷、6.65 联络线输电情况）；
  候选 = 在此之前已披露的信息（预测、出清电价及分量、辅助服务需求与出清、计划交换）；
  剔除 1：同样次日才披露的字段（实际分燃料发电及占比、其他平衡区实际负荷、实际新能源出力、实际/无意交换、事后结算补偿）；
  剔除 2：目标副本——与目标是同一物理量的事前估计（PJM 两个系统负荷预测之于实际负荷；CAISO 总负荷预测之于 CAISO 实际负荷），
          与 FINAL_PROTOCOL §2 删除目标副本同一原则。
数据划分沿用冻结协议（100 号 common100.load：seed 42 的 70/30、train 内 15% val、仅 train 统计量标准化），与 103 号可比。
"""
import os, sys, json
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation100", "src"))
from common100 import load  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation113", "scripts"))
KMAX = 4


def next_day(ds, f):
    if ds == "pjm":
        return f.startswith("gen_fuel_") or "inadv" in f or f in ("total_pjm_loc_credit", "total_pjm_rmccp_cr", "total_pjm_rmpcp_cr")
    return f.startswith("actual_load__") or f.startswith("actual_renewable_generation") or f.startswith("rtpd_eim_transfer")


RULE_T = {"metered_load_mw": "附表 6.61 实际负荷（次日披露）", "total_gen": "附表 6.57 发电总出力（次日披露）",
          "net_actual_interchange_mw": "附表 6.65 联络线输电情况（次日披露）",
          "actual_load__mw__ca_iso_tac": "附表 6.61 实际负荷（次日披露）"}
GROUPS = {
    "pjm_load": dict(ds="pjm", targ=["metered_load_mw"],
                     copies=["forecast_load_mw_latest_available", "forecast_load_mw_day_ahead"]),
    "pjm_gen_ic": dict(ds="pjm", targ=["total_gen", "net_actual_interchange_mw"], copies=[]),
    "caiso_load": dict(ds="caiso", targ=["actual_load__mw__ca_iso_tac"], copies=["dam_load_forecast__mw__ca_iso_tac"]),
}
cache = {}
for g, cfg in GROUPS.items():
    ds = cfg["ds"]
    if ds not in cache:
        cache[ds] = load(ds)
    D0 = cache[ds]; gen, conf = D0["general"], D0["conf"]
    pool = [i for i in D0["active"] if not next_day(ds, gen[i]) and gen[i] not in cfg["copies"]]
    cand = [gen[i] for i in pool]; tc = [conf.index(t) for t in cfg["targ"]]
    D = dict(Xtr=D0["Xtr"][:, pool].astype(np.float32), Ytr=D0["Ytr"][:, tc].astype(np.float32),
             Xte=D0["Xte"][:, pool].astype(np.float32), Yte=D0["Yte"][:, tc].astype(np.float32),
             fit_idx=np.asarray(D0["fit_idx"]), val_idx=np.asarray(D0["val_idx"]))
    masks, keys = pipe.enumerate_sets(len(pool), KMAX)
    G = os.path.join(ROOT, "groups", g)
    for sub in ("outputs", "logs", "figures"):
        os.makedirs(os.path.join(G, sub), exist_ok=True)
    np.savez(os.path.join(G, "outputs", "D.npz"), **D, masks=masks, keys=np.array(["|".join(map(str, k)) for k in keys]))
    rule = {c: "及时披露（预测 / 出清电价及分量 / 辅助服务需求与出清 / 计划交换）" for c in cand}
    rule.update({f"Y_{t}": RULE_T[t] for t in cfg["targ"]})
    spec = dict(dataset=f"{ds.upper()} 现实口径｜{g}", cand=cand, targ=[f"Y_{t}" for t in cfg["targ"]], kmax=KMAX,
                taus=[0.5, 0.7, 0.9], rule=rule, removed_copies=cfg["copies"],
                removed_next_day=[gen[i] for i in D0["active"] if next_day(ds, gen[i])])
    json.dump(spec, open(os.path.join(G, "outputs", "fields.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{g}: 候选 {len(cand)}，目标 {cfg['targ']}，集合 {len(keys)}，train {len(D['Xtr'])} / test {len(D['Xte'])}")
