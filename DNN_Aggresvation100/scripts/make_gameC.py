# -*- coding: utf-8 -*-
"""CAISO 对照博弈 gameC（p=14，2^14 联盟全枚举）：检验"副本 / 备用通道"场景下 Shapley、全量模型 SAGE 与 M^(K) 的差异。
玩家取 CAISO 三簇真实冗余：分区负荷预测 ×4 + 分区负荷计划 ×3（与 CA ISO 实际负荷/发电计划近重复，r 至 0.986）；
节点电价 ×4（与 SP15 日前 LMP 替代通道，r 至 0.982）；SP15 光伏实际/日前预测（r=0.989）；调节下调需求（与 rd 采购 r=0.988）。"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from common100 import load
D = load("caiso"); g = D["general"]
names = ["dam_load_forecast__mw__pge_tac", "dam_load_forecast__mw__sce_tac", "dam_load_forecast__mw__sdge_tac", "dam_load_forecast__mw__ca_iso_tac",
         "dam_schedule__mw__load__tac_north", "dam_schedule__mw__load__tac_ecntr", "dam_schedule__mw__load__tac_south",
         "dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd", "dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd", "dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd",
         "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd", "actual_renewable_generation__mw__sp15__solar",
         "dam_renewable_forecast__mw__sp15__solar", "dam_as_requirement__mw__as_caiso__regulation_down__minimum"]
P = [g.index(n) for n in names]; p = len(P)
bits = (np.arange(2 ** p)[:, None] >> np.arange(p)[None, :]) & 1
M = np.zeros((2 ** p, len(g)), np.uint8); M[:, P] = bits
M[0, P[0]] = 1   # 空集占位（结果在分析中强制置 0）
np.save(ROOT / "outputs/sets/caiso_gameC.npy", M); json.dump(dict(players=names, idx=P), open(ROOT / "outputs/sets/caiso_gameC_meta.json", "w"))
print("gameC", M.shape)
