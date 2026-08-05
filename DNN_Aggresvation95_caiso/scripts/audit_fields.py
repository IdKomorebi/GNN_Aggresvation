"""字段方案泄露审计：CAISO 候选方案 vs PJM 基准轮廓。

对每个机密字段计算"全部 44 可见字段 → 该机密"的线性 OLS R²（全数据，含截距）。
判定原则：CAISO 任一机密的线性 R² 不应显著超出 PJM 轮廓的最大值（≈精确重构红线 0.9995）。
同时检查可见集内部近重复对（|r|>0.995）与设计矩阵条件数。
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAISO_CSV = ROOT / "data/caiso/caiso_extracted/caiso_2025_hourly_augmented/caiso_2025_hourly_aligned_augmented.csv.gz"
PJM_CLEAN = ROOT / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"

# ---------------- PJM 基准（69 号运行时清洗协议） ----------------
PJM_CONF = [
    "net_actual_interchange_mw", "gross_actual_interchange_mw", "total_gen",
    "metered_load_mw", "total_losses", "congestion_price_da", "congestion_price_rt",
    "marginal_loss_price_da", "total_lmp_da", "da_as_total_mw_primary_reserve",
    "da_as_total_mw_synchronized_reserve", "da_as_total_mw_thirty_minutes_reserve",
]

# ---------------- CAISO 方案 ----------------
CAISO_CONF = [
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

_load = ["actual_load__mw__pge_tac", "actual_load__mw__sce_tac", "actual_load__mw__sdge_tac",
         "dam_load_forecast__mw__pge_tac", "dam_load_forecast__mw__sce_tac",
         "dam_load_forecast__mw__sdge_tac", "dam_load_forecast__mw__ca_iso_tac"]
_renew = [f"actual_renewable_generation__mw__{h}__{s}" for h in ("np15", "sp15", "zp26") for s in ("solar", "wind")] + \
         [f"dam_renewable_forecast__mw__{h}__{s}" for h in ("np15", "sp15", "zp26") for s in ("solar", "wind")]
_sched_load = ["dam_schedule__mw__load__tac_north", "dam_schedule__mw__load__tac_ecntr",
               "dam_schedule__mw__load__tac_south"]
_as8 = [f"dam_as_clearing_price__usd_per_mw__{p}__as_caiso" for p in ("ru", "rd", "sr", "nr")] + \
       [f"dam_as_requirement__mw__as_caiso__{p}__minimum"
        for p in ("regulation_up", "regulation_down", "spinning_reserve", "non_spinning_reserve")]
_eim6 = ["rtpd_eim_transfer__mw__export__bcha__malin500",
         "rtpd_eim_transfer__mw__export__banc__ranchoseco",
         "rtpd_eim_transfer__mw__import__nevp__eldorado230",
         "rtpd_eim_transfer__mw__export__srp__pvwest",
         "rtpd_eim_transfer__mw__import__ladwp__sylmar",
         "rtpd_eim_transfer__mw__export__pge__malin500"]

_price_A0 = ["dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd",
             "rt15_lmp_hourly_mean__energy_usd_per_mwh__th_sp15_gen_apnd",
             "dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd",
             "dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd",
             "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd",
             "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd",
             "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_zp26_gen_apnd",
             "rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd"]
# A1：去掉 RT energy（镜像 PJM 丢 system_energy_price_rt），换入外部大 BA 实际负荷
_price_A1 = [c for c in _price_A0 if "rt15_lmp_hourly_mean__energy" not in c]
_extra_A1 = ["actual_load__mw__bpat"]

# A2：在 A1 基础上把 3 个 CAISO 内部 TAC 实际负荷（参与加和恒等式，致机密
# ca_iso_tac R²=0.99998）换成 3 个 WEIM 外部 BA 实际负荷（不参与恒等式），
# 镜像 PJM"可见集只有负荷预测、无分区实际负荷"的结构
_load_A2 = ["dam_load_forecast__mw__pge_tac", "dam_load_forecast__mw__sce_tac",
            "dam_load_forecast__mw__sdge_tac", "dam_load_forecast__mw__ca_iso_tac",
            "actual_load__mw__pace", "actual_load__mw__nevp", "actual_load__mw__azps"]

# A3：A2 基础上把两列**整列为零**的 AS 清算价（sr/nr as_caiso，运行时会被当常量列
# 删掉导致 nG=42）换成 regulation mileage 需求 minimum（up/down，非零、零缺失）
_as8_A3 = [f"dam_as_clearing_price__usd_per_mw__{p}__as_caiso" for p in ("ru", "rd")] + \
          ["dam_as_requirement__mw__as_caiso_exp__regulation_mileage_up__minimum",
           "dam_as_requirement__mw__as_caiso_exp__regulation_mileage_down__minimum"] + \
          [f"dam_as_requirement__mw__as_caiso__{p}__minimum"
           for p in ("regulation_up", "regulation_down", "spinning_reserve",
                     "non_spinning_reserve")]

SCHEMES = {
    "A0": _load + _renew + _price_A0 + _sched_load + _as8 + _eim6,
    "A1": _load + _renew + _price_A1 + _extra_A1 + _sched_load + _as8 + _eim6,
    "A2": _load_A2 + _renew + _price_A1 + _extra_A1 + _sched_load + _as8 + _eim6,
    "A3": _load_A2 + _renew + _price_A1 + _extra_A1 + _sched_load + _as8_A3 + _eim6,
}


def linear_r2_profile(df, visible, conf):
    sub = df[visible + conf].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    X = sub[visible].to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    X = np.hstack([X, np.ones((n, 1))])
    out = {}
    for c in conf:
        y = sub[c].to_numpy(float)
        y = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[c] = 1.0 - float(((y - X @ beta) ** 2).mean())
    cond = float(np.linalg.cond(X[:, :-1]))
    corr = np.corrcoef(X[:, :-1].T)
    np.fill_diagonal(corr, 0)
    dup = [(visible[i], visible[j], round(float(corr[i, j]), 4))
           for i in range(len(visible)) for j in range(i + 1, len(visible))
           if abs(corr[i, j]) > 0.995]
    return n, out, cond, dup


print("=" * 90)
print("PJM 基准轮廓（44 可见 → 12 机密，线性 R²）")
pjm = pd.read_csv(PJM_CLEAN)
pjm = pjm.select_dtypes("number")
pjm = pjm.loc[:, pjm.std() > 1e-10]
pjm_vis = [c for c in pjm.columns if c not in PJM_CONF]
n, prof, cond, dup = linear_r2_profile(pjm, pjm_vis, PJM_CONF)
print(f"行数 {n}, 可见 {len(pjm_vis)}, 条件数 {cond:.1e}, 近重复对 {dup}")
for c, r2 in sorted(prof.items(), key=lambda kv: -kv[1]):
    print(f"  {r2:.6f}  {c}")
pjm_max = max(prof.values())

caiso = pd.read_csv(CAISO_CSV)
for name, vis in SCHEMES.items():
    print("=" * 90)
    missing = [c for c in vis + CAISO_CONF if c not in caiso.columns]
    assert not missing, f"{name} 列不存在: {missing}"
    assert len(vis) == 44 and len(set(vis)) == 44, f"{name} 可见列数 {len(vis)}"
    n, prof, cond, dup = linear_r2_profile(caiso, vis, CAISO_CONF)
    print(f"CAISO 方案 {name}: 行数 {n}, 条件数 {cond:.1e}")
    print(f"近重复对(|r|>0.995): {dup}")
    for c, r2 in sorted(prof.items(), key=lambda kv: -kv[1]):
        flag = " ★超PJM最大" if r2 > pjm_max else ""
        flag = " ★★近精确重构" if r2 > 0.9995 else flag
        print(f"  {r2:.6f}  {c}{flag}")
