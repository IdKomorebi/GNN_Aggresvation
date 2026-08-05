# -*- coding: utf-8 -*-
"""生成 CAISO 清洗版 CSV（对齐 1 号实验对 PJM 的一级清洗职责：只选列，不动行）。

输入 307 列原始表 → 输出 56 列（44 可见 + 12 机密，方案 A2，见 ../fields_design.md）。
dropna/常量列/切分/标准化都留给 69 号 prepare_data 运行时完成，与 PJM 管线同构。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RAW = (REPO / "data/caiso/caiso_extracted/caiso_2025_hourly_augmented/"
       "caiso_2025_hourly_aligned_augmented.csv.gz")
OUT = REPO / "data/Processed/caiso_2025_hourly_cleaned.csv"

VISIBLE = [
    # 负荷 7（内部 TAC 只给日前预测；实际负荷只给 WEIM 外部 BA，避开加和恒等式）
    "dam_load_forecast__mw__pge_tac", "dam_load_forecast__mw__sce_tac",
    "dam_load_forecast__mw__sdge_tac", "dam_load_forecast__mw__ca_iso_tac",
    "actual_load__mw__pace", "actual_load__mw__nevp", "actual_load__mw__azps",
    # 风光 12
    *[f"actual_renewable_generation__mw__{h}__{s}"
      for h in ("np15", "sp15", "zp26") for s in ("solar", "wind")],
    *[f"dam_renewable_forecast__mw__{h}__{s}"
      for h in ("np15", "sp15", "zp26") for s in ("solar", "wind")],
    # 价格 7（RT energy 不可见，镜像 PJM 丢 system_energy_price_rt）
    "dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd",
    "dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd",
    "dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd",
    "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd",
    "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd",
    "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_zp26_gen_apnd",
    "rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd",
    # 外部负荷补充 1
    "actual_load__mw__bpat",
    # 计划负荷分区 3（generation/import/export 子区不可见，避开机密 totals 的加和恒等式）
    "dam_schedule__mw__load__tac_north", "dam_schedule__mw__load__tac_ecntr",
    "dam_schedule__mw__load__tac_south",
    # AS 8（sr/nr 清算价整列为零→换成 regulation mileage 需求 minimum，见 fields_design.md）
    *[f"dam_as_clearing_price__usd_per_mw__{p}__as_caiso" for p in ("ru", "rd")],
    "dam_as_requirement__mw__as_caiso_exp__regulation_mileage_up__minimum",
    "dam_as_requirement__mw__as_caiso_exp__regulation_mileage_down__minimum",
    *[f"dam_as_requirement__mw__as_caiso__{p}__minimum"
      for p in ("regulation_up", "regulation_down", "spinning_reserve",
                "non_spinning_reserve")],
    # EIM 大 tie 6
    "rtpd_eim_transfer__mw__export__bcha__malin500",
    "rtpd_eim_transfer__mw__export__banc__ranchoseco",
    "rtpd_eim_transfer__mw__import__nevp__eldorado230",
    "rtpd_eim_transfer__mw__export__srp__pvwest",
    "rtpd_eim_transfer__mw__import__ladwp__sylmar",
    "rtpd_eim_transfer__mw__export__pge__malin500",
]

CONF = [
    "actual_load__mw__ca_iso_tac",
    "dam_schedule__mw__generation__caiso_totals",
    "dam_schedule__mw__import__caiso_totals",
    "dam_schedule__mw__export__caiso_totals",
    "dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd",
    "dam_lmp__congestion_usd_per_mwh__th_sp15_gen_apnd",
    "dam_lmp__loss_usd_per_mwh__th_sp15_gen_apnd",
    "rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd",
    "dam_as_total_procured__mw__ru__as_caiso",
    "dam_as_total_procured__mw__rd__as_caiso",
    "dam_as_total_procured__mw__sr__as_caiso",
    "dam_as_total_procured__mw__nr__as_caiso",
]


def main() -> None:
    assert len(VISIBLE) == 44 and len(set(VISIBLE)) == 44
    assert len(CONF) == 12 and not set(VISIBLE) & set(CONF)
    df = pd.read_csv(RAW)
    missing = [c for c in VISIBLE + CONF if c not in df.columns]
    assert not missing, missing
    out = df[VISIBLE + CONF].apply(pd.to_numeric, errors="coerce")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    meta = {
        "raw_file": RAW.name,
        "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "scheme": "A2",
        "n_rows": len(out),
        "n_visible": len(VISIBLE),
        "n_confidential": len(CONF),
        "n_rows_complete": int(out.dropna().shape[0]),
        "visible": VISIBLE,
        "confidential": CONF,
    }
    OUT.with_suffix("").with_suffix(".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{OUT.name}: {len(out)} 行 × {out.shape[1]} 列，完整行 {meta['n_rows_complete']}")


if __name__ == "__main__":
    main()
