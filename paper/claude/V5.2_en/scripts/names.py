# -*- coding: utf-8 -*-
"""字段 / 目标的英文短名（论文图表用）。"""
import re

REG = {"新南威尔士": "NSW", "昆士兰": "QLD", "维多利亚": "VIC", "南澳": "SA", "塔斯马尼亚": "TAS"}
CN = [(r"^负荷预测_区(\d)$", r"Load fcst. area \1"), (r"^风电出力预测$", "Wind fcst."), (r"^光伏出力预测$", "PV fcst."),
      (r"^屋顶光伏预测$", "Rooftop PV fcst."), (r"^水电出力预测$", "Hydro fcst."), (r"^调节上调需求$", "Reg-up req."),
      (r"^调节下调需求$", "Reg-down req."), (r"^旋转备用需求$", "Spin. reserve req."), (r"^系统电能价格$", "System energy price"),
      (r"^节点电价_(\w+)$", r"LMP bus \1"), (r"^阻塞_(.+)$", r"Congestion \1"),
      (r"^机组出力_(.+)$", r"Unit \1"), (r"^线路潮流_(.+)$", r"Line \1 flow"),
      (r"^实际需求_(.+)$", r"Demand \1"), (r"^可用发电容量_(.+)$", r"Avail. cap. \1"), (r"^风电出力预测_(.+)$", r"Wind fcst. \1"),
      (r"^光伏出力预测_(.+)$", r"PV fcst. \1"), (r"^出清价_(.+)$", r"Price \1"), (r"^联络线潮流_(.+)$", r"Flow \1")]
EN = {"metered_load_mw": "PJM metered load", "total_gen": "PJM total generation", "net_actual_interchange_mw": "PJM net interchange",
      "actual_load__mw__ca_iso_tac": "CAISO load",
      "rmccp": "Reg. capability price", "rmpcp": "Reg. performance price", "total_pjm_reg_purchases": "Reg. purchases",
      "total_pjm_self_sched_reg": "Self-sched. reg.", "total_pjm_assigned_reg": "Assigned reg.",
      "da_as_mcp_primary_reserve": "DA primary res. price", "da_as_mcp_synchronized_reserve": "DA sync. res. price",
      "da_as_as_req_mw_primary_reserve": "DA primary res. req.", "da_as_as_req_mw_thirty_minutes_reserve": "DA 30-min res. req.",
      "da_as_ss_mw_primary_reserve": "DA primary res. self-sched.", "da_as_nsr_mw_primary_reserve": "DA non-sync. res.",
      "system_energy_price_da": "DA system energy price", "total_lmp_rt": "RT LMP", "marginal_loss_price_rt": "RT loss price",
      "gross_sched_interchange_mw": "Sched. interchange", "forecast_load_mw_latest_available": "Load fcst. (latest)",
      "forecast_load_mw_day_ahead": "Load fcst. (DA)", "congestion_price_da": "DA congestion price",
      "marginal_loss_price_da": "DA loss price", "total_lmp_da": "DA LMP", "congestion_price_rt": "RT congestion price",
      "da_as_total_mw_primary_reserve": "DA primary res. cleared", "da_as_total_mw_synchronized_reserve": "DA sync. res. cleared",
      "da_as_total_mw_thirty_minutes_reserve": "DA 30-min res. cleared"}


def field(n):
    n = n.replace("Y_", "")
    if n in EN:
        return EN[n]
    for k, v in REG.items():
        n = n.replace(k, v)
    for pat, rep in CN:
        if re.match(pat, n):
            return re.sub(pat, rep, n)
    # CAISO 长名
    m = re.match(r"dam_load_forecast__mw__(\w+)_tac", n)
    if m:
        return f"DA load fcst. {m.group(1).upper()}"
    m = re.match(r"dam_renewable_forecast__mw__(\w+)__(solar|wind)", n)
    if m:
        return f"DA {m.group(2)} fcst. {m.group(1).upper()}"
    m = re.match(r"(dam|rt15)_lmp(?:_hourly_mean)?__(\w+?)_usd_per_mwh__th_(\w+?)_gen_apnd", n)
    if m:
        comp = {"lmp": "LMP", "energy": "energy", "congestion": "congestion", "loss": "loss"}[m.group(2)]
        return f"{'DA' if m.group(1) == 'dam' else 'RT'} {comp} {m.group(3).upper()}"
    m = re.match(r"dam_schedule__mw__(\w+?)__(\w+)", n)
    if m:
        return f"DA sched. {m.group(1)} {m.group(2).replace('tac_', '').replace('caiso_totals', 'total')}"
    m = re.match(r"dam_as_requirement__mw__as_caiso(?:_exp)?__(\w+?)__minimum", n)
    if m:
        return f"AS req. {m.group(1).replace('_', ' ')}"
    m = re.match(r"dam_as_total_procured__mw__(\w+)__as_caiso", n)
    if m:
        return f"AS procured {m.group(1).upper()}"
    return n


TARGET_SHORT = {"机组出力_223_STEAM_3": "RTS unit 223", "机组出力_221_CC_1": "RTS unit 221", "线路潮流_C35": "RTS line C35",
                "机组出力_BW01": "NEM BW01", "机组出力_TUMUT3": "NEM TUMUT3", "机组出力_PPCCGT": "NEM PPCCGT",
                "metered_load_mw": "PJM load", "total_gen": "PJM total gen.", "net_actual_interchange_mw": "PJM net interch.",
                "actual_load__mw__ca_iso_tac": "CAISO load"}


def target(n):
    n = n.replace("Y_", "")
    return TARGET_SHORT.get(n, n)
