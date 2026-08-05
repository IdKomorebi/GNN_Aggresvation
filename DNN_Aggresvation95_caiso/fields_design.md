# CAISO 第二数据集字段设计（方案 A3 定稿，2026-07-26）

原始数据：`data/caiso/caiso_extracted/caiso_2025_hourly_augmented/caiso_2025_hourly_aligned_augmented.csv.gz`
（307 列 × 8760 小时，UTC 对齐，163 原始 + 142 增补，全部官方 OASIS 可溯源）。

设计目标：与 PJM 的 44 可见 + 12 机密逐项对位，可比性优先；剔除 47 全零列、
10 近零列、1 列 98.9% 缺失、4 条 energy 跨 hub 纯副本后再做选择。

## 12 个机密字段（与 PJM 逐项对位）

| # | CAISO 列 | PJM 对应 | 备注 |
|---|---|---|---|
| 1 | actual_load__mw__ca_iso_tac | metered_load_mw | 实测系统负荷 |
| 2 | dam_schedule__mw__generation__caiso_totals | total_gen | ★日前计划口径（无 RT 实测，limitation） |
| 3 | dam_schedule__mw__import__caiso_totals | gross_actual_interchange_mw | 同上 |
| 4 | dam_schedule__mw__export__caiso_totals | net_actual_interchange_mw | 同上 |
| 5 | dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd | total_lmp_da | SP15 主 hub |
| 6 | dam_lmp__congestion_usd_per_mwh__th_sp15_gen_apnd | congestion_price_da | |
| 7 | dam_lmp__loss_usd_per_mwh__th_sp15_gen_apnd | marginal_loss_price_da | |
| 8 | rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd | congestion_price_rt | |
| 9-12 | dam_as_total_procured__mw__{ru,rd,sr,nr}__as_caiso | da_as_total_mw_×3 | AS 总采购（PJM 3 个，CAISO 4 个） |

PJM 的 total_losses（MW 损耗）在 CAISO 无载体，用第 4 个 AS 产品补足 12 个。

## 44 个可见字段

- 负荷 7：dam_load_forecast {pge_tac, sce_tac, sdge_tac, ca_iso_tac}；
  actual_load {pace, nevp, azps}（WEIM 外部 BA，不参与 CAISO 加和恒等式）
- 风光 12：actual_renewable_generation + dam_renewable_forecast × {np15,sp15,zp26} × {solar,wind}
- 价格 7：dam_lmp energy sp15（每市场唯一 energy）、dam_lmp lmp {np15,zp26}、
  rt15 lmp {np15,sp15,zp26}、rt15 loss np15。**RT energy 不可见**（镜像 PJM 丢
  system_energy_price_rt；见下方审计）
- 外部负荷补充 1：actual_load bpat
- 计划负荷分区 3：dam_schedule load {tac_north, tac_ecntr, tac_south}
- AS 8：dam_as_clearing_price {ru,rd} as_caiso + dam_as_requirement
  regulation_mileage {up,down} minimum (caiso_exp) + dam_as_requirement
  {regulation_up, regulation_down, spinning_reserve, non_spinning_reserve} minimum as_caiso
- EIM 大 tie 6：export bcha malin500 / export banc ranchoseco / import nevp eldorado230 /
  export srp pvwest / import ladwp sylmar / export pge malin500

精确列名清单 = `scripts/build_caiso_cleaned.py` 的 VISIBLE/CONF 常量（单一事实来源）。

## 泄露审计（scripts/audit_fields.py，全数据线性 OLS R²）

判定原则：任一机密的"44 可见 → 机密"线性 R² 不超过 PJM 轮廓最大值（0.9984）。

| 迭代 | 问题 | 修正 |
|---|---|---|
| A0（体检 agent 原案） | rt_cong_sp15 R²=0.9964：RT energy 可见时恒等式残差只剩 loss 项 | A1：移出 RT energy（镜像 PJM），换入 actual_load bpat → 降至 0.788 |
| A1 | actual_load ca_iso_tac R²=0.99998：三大 TAC 实际负荷可见 → 加和恒等式只差 vea+mwd 的 0.83% | A2：3 个 TAC 实际负荷换成 WEIM 外部 BA（pace/nevp/azps）→ 降至 0.943 |
| A2 | sr/nr AS 清算价**整列为零**（体检"7 列全零清算价"含这两列，选列时漏对），运行时被当常量列删掉 → nG=42 | A3：换成 regulation mileage {up,down} 需求 minimum（std 448/1198、零缺失；区域 ru/rd 清算价也近乎全零不可用）→ nG=44 ✅（冒烟训练实证） |

**A2 最终轮廓**（PJM 对照：0.998 total_gen / 0.995 metered_load / 0.986 lmp_da / … / 0.277 cong_da）：

    0.989 dam_lmp_sp15 · 0.982 as_rd · 0.943 load_ca_iso · 0.931 gen_totals ·
    0.841 as_ru · 0.814 dam_cong_sp15 · 0.787 rt_cong_sp15 · 0.668 export ·
    0.660 dam_loss_sp15 · 0.566 import · 0.499 as_nr · 0.240 as_sr

最大 0.989 < PJM 最大 0.998；跨度 0.24–0.99 与 PJM（0.28–0.998）形态一致。✅

已知结构（记录不修）：
- 可见集内 spinning/non_spinning requirement minimum 完全重复（r=1.0）——PJM 可见集
  同样有两组完全重复列（req primary=sync、ss 三胞胎），忠实镜像，保留；
- 机密 dam_lmp_sp15 = 可见 dam_energy + 机密 cong_sp15 + 机密 loss_sp15（恒等式两个
  未知数在机密侧，不可精确重构单个机密）；
- hub 间相关 sp15–zp26=0.982 等近似冗余无法消除，与 PJM zone/RTO 相关同性质。

## 与 PJM 的口径差异（论文 limitation 必写）

1. total_gen / import / export 是**日前计划值**（ENE_SLRS），非 RT 实测；MW 损耗无载体；
2. 机密里 AS 产品 4 个（PJM 3 个 + total_losses）；
3. 行数：dropna 后 8722（PJM 6556）；train/test = 6105/2617（PJM 4589/1967）。
