# -*- coding: utf-8 -*-
"""98 号公共件：数据口径、字段别名组、博弈玩家集。

数据口径与 69/75/91/93/97 完全一致（同一 prepare_data、shuffle seed=42、70/30），
这样历史 oracle 的训练集与本号测试集严格不相交，可直接复用其权重。
新增：train 内部再切 15% 作 val（真值早停与选参只看 val，测试集只做最终报告）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
if str(R69) not in sys.path:
    sys.path.insert(0, str(R69))

VAL_SEED = 20260915

# 精确重复 / 仿射重复（标准化后逐行相等）的字段组。组内任取一个即携带全部信息。
ALIAS_GROUPS_PJM = [
    ["da_as_ss_mw_primary_reserve", "da_as_ss_mw_synchronized_reserve",
     "da_as_ss_mw_thirty_minutes_reserve"],
    ["da_as_as_req_mw_primary_reserve", "da_as_as_req_mw_synchronized_reserve"],
]

# 博弈 A：14 个玩家，刻意放入已知结构——
#   风电出力/占比（total_gen 强协同）、两种负荷预测（近重复 r=0.9966）、
#   备用自调度两列（精确重复）、价格簇、燃料、交换功率。
GAME_A = [
    "gen_fuel_wind_mw", "gen_fuel_wind_pct",
    "forecast_load_mw_latest_available", "forecast_load_mw_day_ahead",
    "da_as_ss_mw_primary_reserve", "da_as_ss_mw_synchronized_reserve",
    "system_energy_price_da", "total_lmp_rt", "marginal_loss_price_rt",
    "gen_fuel_coal_mw", "gen_fuel_gas_pct",
    "gross_sched_interchange_mw", "gross_inadv_interchange_mw",
    "da_as_as_req_mw_primary_reserve",
]


def load_pjm():
    from src.data_processing import prepare_data  # 69 号
    cfg = yaml.safe_load((R69 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    tr, te = di["train_data"], di["test_data"]
    n = len(tr)
    perm = np.random.RandomState(VAL_SEED).permutation(n)
    nv = round(n * 0.15)
    return {
        "Xtr": tr[:, gi], "Ytr": tr[:, ci], "Xte": te[:, gi], "Yte": te[:, ci],
        "fit_idx": perm[nv:], "val_idx": perm[:nv],
        "general": list(di["general"]), "conf": list(di["confidential"]),
    }


def dedup_fields(general: list[str]) -> list[str]:
    """去掉别名组中除第一个外的成员，返回 41 个不同字段。"""
    drop = {f for g in ALIAS_GROUPS_PJM for f in g[1:]}
    return [f for f in general if f not in drop]


def game_b(general: list[str], p: int = 12, seed: int = 98) -> list[str]:
    pool = [f for f in dedup_fields(general) if f not in GAME_A]
    rng = np.random.RandomState(seed)
    return [pool[i] for i in sorted(rng.choice(len(pool), size=p, replace=False))]


def game_masks(players_idx: list[int], n_general: int) -> np.ndarray:
    """全部 2^p 个联盟的 44 维掩码，第 b 行对应位掩码 b（玩家 k ↔ 第 k 位）。"""
    p = len(players_idx)
    bits = (np.arange(2 ** p)[:, None] >> np.arange(p)[None, :]) & 1
    m = np.zeros((2 ** p, n_general), dtype=np.float32)
    m[:, players_idx] = bits
    return m
