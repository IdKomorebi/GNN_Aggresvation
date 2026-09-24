# -*- coding: utf-8 -*-
"""116 号 步骤 1：PJM / CAISO 现实口径（补全候选池）——生成三组 D.npz 与 fields.json。

与 114 号的唯一区别：100 号协议把 12 列当作"机密列"，它们从未进入输入。114 号只取其中 3 个次日披露的实际量作目标，
其余机密列既没当目标、也没进候选。按《电力市场信息披露基本规则》（国能发监管〔2024〕9号），其中及时披露的列本该是候选：
  PJM  ：日前阻塞价、日前网损价、日前节点电价总价（附表 6.52 出清电价及分量）、实时阻塞价（6.52）、
         三类备用日前出清总量（辅助服务出清结果）；
  CAISO：日前计划的发电 / 进口 / 出口总量（日前出清计划）、SP15 日前电价及阻塞、网损分量（6.52）、SP15 实时阻塞分量（6.52）、
         四类辅助服务日前采购总量（辅助服务出清结果）。
次日才披露的机密列（PJM 实际总交换、实际网损）不进候选。其余规则与 114 号完全相同：
  剔除次日披露字段；剔除目标副本（与目标是同一物理量的事前估计，FINAL_PROTOCOL §2）；完全重复的别名按 100 号协议已去除。
数据划分沿用冻结协议（100 号 common100.load：seed 42 的 70/30、train 内 15% val、仅 train 统计量标准化）。
"""
import os, sys, json
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation100", "src"))
from common100 import load  # noqa: E402
KMAX = 4


def next_day(ds, f):
    if ds == "pjm":
        return f.startswith("gen_fuel_") or "inadv" in f or f in ("total_pjm_loc_credit", "total_pjm_rmccp_cr", "total_pjm_rmpcp_cr")
    return f.startswith("actual_load__") or f.startswith("actual_renewable_generation") or f.startswith("rtpd_eim_transfer")


# 机密列中及时披露、应进候选的列（附规则类别）
TIMELY_CONF = {
    "pjm": {"congestion_price_da": "附表 6.52 出清电价分量（日前阻塞）", "marginal_loss_price_da": "附表 6.52 出清电价分量（日前网损）",
            "total_lmp_da": "附表 6.52 出清电价（日前）", "congestion_price_rt": "附表 6.52 出清电价分量（实时阻塞）",
            "da_as_total_mw_primary_reserve": "辅助服务出清结果（一级备用）",
            "da_as_total_mw_synchronized_reserve": "辅助服务出清结果（旋转备用）",
            "da_as_total_mw_thirty_minutes_reserve": "辅助服务出清结果（30 分钟备用）"},
    "caiso": {"dam_schedule__mw__generation__caiso_totals": "日前出清计划（发电总量）",
              "dam_schedule__mw__import__caiso_totals": "日前出清计划（进口总量）",
              "dam_schedule__mw__export__caiso_totals": "日前出清计划（出口总量）",
              "dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd": "附表 6.52 出清电价（日前 SP15）",
              "dam_lmp__congestion_usd_per_mwh__th_sp15_gen_apnd": "附表 6.52 出清电价分量（日前 SP15 阻塞）",
              "dam_lmp__loss_usd_per_mwh__th_sp15_gen_apnd": "附表 6.52 出清电价分量（日前 SP15 网损）",
              "rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd": "附表 6.52 出清电价分量（实时 SP15 阻塞）",
              "dam_as_total_procured__mw__ru__as_caiso": "辅助服务出清结果（上调频）",
              "dam_as_total_procured__mw__rd__as_caiso": "辅助服务出清结果（下调频）",
              "dam_as_total_procured__mw__sr__as_caiso": "辅助服务出清结果（旋转备用）",
              "dam_as_total_procured__mw__nr__as_caiso": "辅助服务出清结果（非旋转备用）"}}
NEXTDAY_CONF = {"pjm": ["gross_actual_interchange_mw", "total_losses"], "caiso": []}
RULE_T = {"metered_load_mw": "附表 6.61 实际负荷（次日披露）", "total_gen": "附表 6.57 发电总出力（次日披露）",
          "net_actual_interchange_mw": "附表 6.65 联络线输电情况（次日披露）",
          "actual_load__mw__ca_iso_tac": "附表 6.61 实际负荷（次日披露）"}
GROUPS = {
    "pjm_load": dict(ds="pjm", targ=["metered_load_mw"], copies=["forecast_load_mw_latest_available", "forecast_load_mw_day_ahead"]),
    "pjm_gen_ic": dict(ds="pjm", targ=["total_gen", "net_actual_interchange_mw"], copies=[]),
    "caiso_load": dict(ds="caiso", targ=["actual_load__mw__ca_iso_tac"], copies=["dam_load_forecast__mw__ca_iso_tac"]),
}
if __name__ == "__main__":
    cache = {}
    for g, cfg in GROUPS.items():
        ds = cfg["ds"]
        if ds not in cache:
            cache[ds] = load(ds)
        D0 = cache[ds]; gen, conf = D0["general"], D0["conf"]
        pool = [i for i in D0["active"] if not next_day(ds, gen[i]) and gen[i] not in cfg["copies"]]
        extra = [conf.index(c) for c in TIMELY_CONF[ds]]
        cand = [gen[i] for i in pool] + [conf[j] for j in extra]; tc = [conf.index(t) for t in cfg["targ"]]
        cat = lambda X, Y: np.concatenate([X[:, pool], Y[:, extra]], 1).astype(np.float32)
        D = dict(Xtr=cat(D0["Xtr"], D0["Ytr"]), Ytr=D0["Ytr"][:, tc].astype(np.float32),
                 Xte=cat(D0["Xte"], D0["Yte"]), Yte=D0["Yte"][:, tc].astype(np.float32),
                 fit_idx=np.asarray(D0["fit_idx"]), val_idx=np.asarray(D0["val_idx"]))
        masks, keys = pipe.enumerate_sets(len(cand), KMAX)
        G = os.path.join(ROOT, "groups", g)
        for sub in ("outputs", "logs", "figures"):
            os.makedirs(os.path.join(G, sub), exist_ok=True)
        np.savez(os.path.join(G, "outputs", "D.npz"), **D, masks=masks, keys=np.array(["|".join(map(str, k)) for k in keys]))
        rule = {c: "及时披露（预测 / 出清电价及分量 / 辅助服务需求与出清 / 计划交换）" for c in cand}
        rule.update(TIMELY_CONF[ds]); rule.update({f"Y_{t}": RULE_T[t] for t in cfg["targ"]})
        spec = dict(dataset=f"{ds.upper()} 现实口径（补全候选）｜{g}", cand=cand, targ=[f"Y_{t}" for t in cfg["targ"]], kmax=KMAX,
                    taus=[0.5, 0.7, 0.9], rule=rule, removed_copies=cfg["copies"], added_timely=list(TIMELY_CONF[ds]),
                    removed_next_day=[gen[i] for i in D0["active"] if next_day(ds, gen[i])] + NEXTDAY_CONF[ds],
                    src_cols=dict(general=[gen[i] for i in pool], conf=[conf[j] for j in extra]))
        json.dump(spec, open(os.path.join(G, "outputs", "fields.json"), "w"), ensure_ascii=False, indent=1)
        print(f"{g}: 候选 {len(cand)}（其中新增 {len(extra)}），目标 {cfg['targ']}，集合 {len(keys)}")
