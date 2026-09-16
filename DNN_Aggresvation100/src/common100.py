# -*- coding: utf-8 -*-
"""100 号公共件：两数据集的统一口径（去别名后的完整字段集）。

数据切分沿用 69/75/95 号（prepare_data：删常量列、删含 NaN 行、shuffle seed=42、70/30），
历史通用模型的训练集与本号测试集严格不相交；train 内部 15% 为 val（VAL_SEED=20260915，与 98 号相同）。
去别名规则：完全相同或确定性仿射相同的字段组只保留第一个；目标字段的可见副本删除；
近重复但真实存在的字段（r≈0.995）保留。被删字段在 44 维掩码中恒为 0，所以无需重训通用模型。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
if str(R69) not in sys.path:
    sys.path.insert(0, str(R69))
VAL_SEED = 20260915

CFG = {
    "pjm": dict(yaml=R69 / "base.yaml", csv=REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv",
                drop_alias=["da_as_ss_mw_synchronized_reserve", "da_as_ss_mw_thirty_minutes_reserve",   # ≡ ss_mw_primary
                            "da_as_as_req_mw_synchronized_reserve"]),                                  # = 0.667·primary + 63.3
    "caiso": dict(yaml=REPO / "DNN_Aggresvation95_caiso/base.yaml", csv=REPO / "data/Processed/caiso_2025_hourly_cleaned.csv",
                  drop_alias=["dam_as_requirement__mw__as_caiso__non_spinning_reserve__minimum",       # spinning = 4 × non_spinning
                              "dam_as_clearing_price__usd_per_mw__ru__as_caiso",                      # 8722 行仅 11 行非 0
                              "dam_as_clearing_price__usd_per_mw__rd__as_caiso"]),                    # 8722 行仅 25 行非 0
}
# PJM 公开基底场景：日前与最新负荷预测为 PJM 官方公开发布的字段
BASE_PJM = ["forecast_load_mw_latest_available", "forecast_load_mw_day_ahead"]


def load(ds: str, split: str = "main", half: int = 0, rep: int = 0):
    """split=main：69 号口径；split=half：101 号独立认证用（本文件只提供 main）。"""
    from src.data_processing import prepare_data
    c = CFG[ds]
    cfg = yaml.safe_load(Path(c["yaml"]).read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(c["csv"])
    di = prepare_data(cfg)
    gi, ci = np.asarray(di["general_indices"]), np.asarray(di["confidential_indices"])
    tr, te = di["train_data"], di["test_data"]
    perm = np.random.RandomState(VAL_SEED).permutation(len(tr)); nv = round(len(tr) * 0.15)
    general = list(di["general"])
    active = [k for k, f in enumerate(general) if f not in c["drop_alias"]]
    return {"Xtr": tr[:, gi], "Ytr": tr[:, ci], "Xte": te[:, gi], "Yte": te[:, ci],
            "fit_idx": perm[nv:], "val_idx": perm[:nv], "general": general, "conf": list(di["confidential"]),
            "active": active, "ds": ds}


def enumerate_sets(active: list[int], nG: int, kmax: int, base: list[int] | None = None):
    """在 active 字段上枚举规模 1..kmax 的全部集合（base 字段始终可见、不计规模，也不在候选内）。
    返回 (masks uint8 (N,nG), keys list[tuple])，keys 为候选字段的升序元组（不含 base）。"""
    base = base or []
    cand = [a for a in active if a not in base]
    keys = [()] if base else []
    for k in range(1, kmax + 1):
        keys += list(combinations(cand, k))
    M = np.zeros((len(keys), nG), np.uint8)
    for r, key in enumerate(keys):
        M[r, list(key) + base] = 1
    return M, keys


def load_half(ds: str, rep: int, half: int):
    """独立认证用：把全部完整行按 rep 随机对半分成 H0/H1，在 H_half 内再 shuffle 70/30 分 train/test，
    标准化只用该半的 train 统计量；train 内 15% 为 val。两半之间没有任何共享样本。"""
    from src.data_processing import prepare_data
    c = CFG[ds]
    cfg = yaml.safe_load(Path(c["yaml"]).read_text(encoding="utf-8")); cfg["dataset"]["csv_path"] = str(c["csv"])
    di = prepare_data(cfg); df = di["raw_df"]; cols = di["general"] + di["confidential"]
    perm = np.random.RandomState(9900 + rep).permutation(len(df)); h = perm[: len(df) // 2] if half == 0 else perm[len(df) // 2:]
    sub = df.iloc[h].reset_index(drop=True)
    p2 = np.random.RandomState(42 + rep).permutation(len(sub)); ntr = int(len(sub) * 0.7)
    tr_df, te_df = sub.iloc[p2[:ntr]], sub.iloc[p2[ntr:]]
    mu, sd = tr_df[cols].mean(), tr_df[cols].std().replace(0, 1)
    tr = ((tr_df[cols] - mu) / sd).values.astype(np.float32); te = ((te_df[cols] - mu) / sd).values.astype(np.float32)
    nG = len(di["general"]); pv = np.random.RandomState(VAL_SEED).permutation(len(tr)); nv = round(len(tr) * 0.15)
    general = list(di["general"])
    return {"Xtr": tr[:, :nG], "Ytr": tr[:, nG:], "Xte": te[:, :nG], "Yte": te[:, nG:], "fit_idx": pv[nv:], "val_idx": pv[:nv],
            "general": general, "conf": list(di["confidential"]), "active": [k for k, f in enumerate(general) if f not in c["drop_alias"]],
            "ds": ds, "rep": rep, "half": half}


def load_any(ds, split="main", rep=0, half=0):
    return load(ds) if split == "main" else load_half(ds, rep, half)
