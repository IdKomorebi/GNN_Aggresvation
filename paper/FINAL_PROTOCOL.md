# FINAL_PROTOCOL：正式论文实验协议（冻结版）

> 冻结日期：2026-09-18。依据 `paper/20260918_论文主线与完整实验计划.md` 第 12、20 节。
> **本协议一经冻结，除发现 bug 外不得修改。** 所有正式实验（102 号起）必须声明遵循本协议；历史实验（≤101 号）作为背景证据保留，不覆盖。

## 1. 数据

| | PJM | CAISO |
| --- | --- | --- |
| 文件 | `data/Processed/pjm_rto_hourly_2025_cleaned.csv` | `data/Processed/caiso_2025_hourly_cleaned.csv` |
| SHA-256（前 16 位） | `861affda60a665d8` | `9addab5be13ede8b` |
| 字段角色来源 | `DNN_Aggresvation69/base.yaml` | `DNN_Aggresvation95_caiso/base.yaml` |
| 预处理 | `DNN_Aggresvation69/src/data_processing.py:prepare_data`：删除指定列 → 仅保留数值列 → 删除常量列（std<1e-10）→ 删除含 NaN 的行 | 同左 |
| 完整样本行数 | 6,556 | 8,722 |
| 划分 | shuffle，seed=42，train 70% / test 30% | 同左 |
| train / val / test | 4589（其中 val 688） / 1967 | 6105（其中 val 916） / 2617 |
| val 抽取 | train 内随机 15%，seed=20260915（`common100.VAL_SEED`） | 同左 |
| 标准化 | 仅用 train 的均值/标准差（std=0 置 1） | 同左 |
| 候选字段数 | 41 | 41 |
| 敏感目标数 | 12 | 12 |

## 2. 别名合并与目标副本删除

规则：**标准化后逐行相等（精确或确定性仿射）的字段只保留第一个；近重复（r<0.999 但很高）保留**，因为它们正是方法要处理的真实冗余。

- PJM 删除：`da_as_ss_mw_synchronized_reserve`, `da_as_ss_mw_thirty_minutes_reserve`, `da_as_as_req_mw_synchronized_reserve`
  （`ss_mw` 三列逐行相等；`as_req_mw_synchronized = 0.667×primary + 63.3`）
- CAISO 删除：`dam_as_requirement__mw__as_caiso__non_spinning_reserve__minimum`, `dam_as_clearing_price__usd_per_mw__ru__as_caiso`, `dam_as_clearing_price__usd_per_mw__rd__as_caiso`
  （spinning = 4 × non_spinning；两个清算价在 8,722 行中分别只有 11 / 25 行非零）
- **目标副本**：PJM 的 `da_as_as_mw_*`、`system_energy_price_rt`、`prelim_load_avg_hourly`、`total_pjm_rt_load_mwh`、`net_sched_interchange_mw`、`wind/solar_generation_mw` 等已在 69 号 `drop_columns` 中删除；CAISO 经检查无 |r|>0.999 的可见-目标对。

## 3. 候选字段清单

### PJM（41 个）

1. `gen_fuel_coal_mw`
2. `gen_fuel_gas_mw`
3. `gen_fuel_hydro_mw`
4. `gen_fuel_multiple_fuels_mw`
5. `gen_fuel_nuclear_mw`
6. `gen_fuel_oil_mw`
7. `gen_fuel_other_renewables_mw`
8. `gen_fuel_solar_mw`
9. `gen_fuel_storage_mw`
10. `gen_fuel_wind_mw`
11. `gen_fuel_coal_pct`
12. `gen_fuel_gas_pct`
13. `gen_fuel_hydro_pct`
14. `gen_fuel_multiple_fuels_pct`
15. `gen_fuel_nuclear_pct`
16. `gen_fuel_oil_pct`
17. `gen_fuel_other_renewables_pct`
18. `gen_fuel_solar_pct`
19. `gen_fuel_wind_pct`
20. `forecast_load_mw_latest_available`
21. `forecast_load_mw_day_ahead`
22. `rmccp`
23. `rmpcp`
24. `total_pjm_loc_credit`
25. `total_pjm_reg_purchases`
26. `total_pjm_self_sched_reg`
27. `total_pjm_assigned_reg`
28. `total_pjm_rmccp_cr`
29. `total_pjm_rmpcp_cr`
30. `da_as_mcp_primary_reserve`
31. `da_as_mcp_synchronized_reserve`
32. `da_as_as_req_mw_primary_reserve`
33. `da_as_as_req_mw_thirty_minutes_reserve`
34. `da_as_ss_mw_primary_reserve`
35. `da_as_nsr_mw_primary_reserve`
36. `system_energy_price_da`
37. `total_lmp_rt`
38. `marginal_loss_price_rt`
39. `net_inadv_interchange_mw`
40. `gross_sched_interchange_mw`
41. `gross_inadv_interchange_mw`

### CAISO（41 个）

1. `dam_load_forecast__mw__pge_tac`
2. `dam_load_forecast__mw__sce_tac`
3. `dam_load_forecast__mw__sdge_tac`
4. `dam_load_forecast__mw__ca_iso_tac`
5. `actual_load__mw__pace`
6. `actual_load__mw__nevp`
7. `actual_load__mw__azps`
8. `actual_renewable_generation__mw__np15__solar`
9. `actual_renewable_generation__mw__np15__wind`
10. `actual_renewable_generation__mw__sp15__solar`
11. `actual_renewable_generation__mw__sp15__wind`
12. `actual_renewable_generation__mw__zp26__solar`
13. `actual_renewable_generation__mw__zp26__wind`
14. `dam_renewable_forecast__mw__np15__solar`
15. `dam_renewable_forecast__mw__np15__wind`
16. `dam_renewable_forecast__mw__sp15__solar`
17. `dam_renewable_forecast__mw__sp15__wind`
18. `dam_renewable_forecast__mw__zp26__solar`
19. `dam_renewable_forecast__mw__zp26__wind`
20. `dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd`
21. `dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd`
22. `dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd`
23. `rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd`
24. `rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd`
25. `rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_zp26_gen_apnd`
26. `rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd`
27. `actual_load__mw__bpat`
28. `dam_schedule__mw__load__tac_north`
29. `dam_schedule__mw__load__tac_ecntr`
30. `dam_schedule__mw__load__tac_south`
31. `dam_as_requirement__mw__as_caiso_exp__regulation_mileage_up__minimum`
32. `dam_as_requirement__mw__as_caiso_exp__regulation_mileage_down__minimum`
33. `dam_as_requirement__mw__as_caiso__regulation_up__minimum`
34. `dam_as_requirement__mw__as_caiso__regulation_down__minimum`
35. `dam_as_requirement__mw__as_caiso__spinning_reserve__minimum`
36. `rtpd_eim_transfer__mw__export__bcha__malin500`
37. `rtpd_eim_transfer__mw__export__banc__ranchoseco`
38. `rtpd_eim_transfer__mw__import__nevp__eldorado230`
39. `rtpd_eim_transfer__mw__export__srp__pvwest`
40. `rtpd_eim_transfer__mw__import__ladwp__sylmar`
41. `rtpd_eim_transfer__mw__export__pge__malin500`

## 4. 敏感目标清单

### PJM

1. `net_actual_interchange_mw`
2. `gross_actual_interchange_mw`
3. `total_gen`
4. `metered_load_mw`
5. `total_losses`
6. `congestion_price_da`
7. `congestion_price_rt`
8. `marginal_loss_price_da`
9. `total_lmp_da`
10. `da_as_total_mw_primary_reserve`
11. `da_as_total_mw_synchronized_reserve`
12. `da_as_total_mw_thirty_minutes_reserve`

### CAISO

1. `actual_load__mw__ca_iso_tac`
2. `dam_schedule__mw__generation__caiso_totals`
3. `dam_schedule__mw__import__caiso_totals`
4. `dam_schedule__mw__export__caiso_totals`
5. `dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd`
6. `dam_lmp__congestion_usd_per_mwh__th_sp15_gen_apnd`
7. `dam_lmp__loss_usd_per_mwh__th_sp15_gen_apnd`
8. `rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd`
9. `dam_as_total_procured__mw__ru__as_caiso`
10. `dam_as_total_procured__mw__rd__as_caiso`
11. `dam_as_total_procured__mw__sr__as_caiso`
12. `dam_as_total_procured__mw__nr__as_caiso`

## 5. 威胁模型

- **Auditor**：数据发布者，持有完整历史数据（含敏感目标），离线审计。
- **Adversary**：可获得历史辅助标签，基于公开字段训练推断器；风险以独立 test 上的样本外 R² 衡量（supervised inferability，非 zero-shot）。
- **公开基底 B**：默认 ∅；现实场景实验取 PJM `B = {forecast_load_mw_latest_available, forecast_load_mw_day_ahead}`。
- **背景预算 K**：攻击者除 B 外还能掌握的候选字段数上限，正文 K∈{0,1,2}，K=3 作扩展。
- **阈值**：τ∈{0.5, 0.7, 0.9}，正文固定 0.7；风险档位 0.05 / 0.2 / 0.5 仅作工程展示，并做敏感性。

## 6. 数据使用边界（防泄露）

| 用途 | 允许使用 |
| --- | --- |
| backbone 预学习 | train（含其内部 val 用于早停） |
| 闭式读出 β 求解 | train |
| λ（ridge 正则）选择 | train 内部 fit/val 划分 |
| 攻击器轮次 / 类型选择 | val |
| 见证背景搜索 | discovery 侧数据（101/106 号）或 train+val（100 号同表口径，须标注） |
| **test** | **仅用于最终报告 R²，不参与上述任何选择** |

## 7. 经验攻击能力（真值）与攻击器族

对字段集合 S 与目标 y：

  V_retrain(S) = max_{h∈H} R²_test(h_S)，其中 h 的轮次/类型由 val 选择。

攻击器族 H（正式版）：
1. **单目标 DNN**：Linear(|S|,128)-ReLU-Dropout0.15-Linear(128,128)-ReLU-Dropout0.15-Linear(128,1)，Adam lr 1e-3、weight_decay 5e-4、batch 128、≤300 epoch，逐 epoch 在 val 上选最佳轮次；
2. **多目标 DNN**（历史口径，12 输出，同超参）；
3. **梯度提升树**：sklearn `HistGradientBoostingRegressor`，lr 0.05、max_iter 600、max_leaf_nodes 31、min_samples_leaf 20、l2 1.0，用 `staged_predict` 在 val 上选迭代数。

实现：GPU 并行专用重训（`DNN_Aggresvation98/src/batch_truth.py`，掩码输入置 0 与只用 S 列重训在函数上等价）；树模型 CPU 多进程。
**单调闭包**：V̄(S) = V(argmax_{T⊆S} V_val(T))，在 val 上选子集、test 上报告（修复有限样本非单调）。

## 8. 通用模型（正式口径）

- **Target-Specific Subset-Universal Inference Model**：对每个目标 y 单独随机掩码预学习一个 backbone φ_{θ_y}(x,m)，输入 [x⊙m, m]；
- 掩码分布：两段式 uniform（先抽保留数 k~U{1..p}，再均匀抽 k 个字段）；
- 训练：EPOCHS 400 / PATIENCE 60 / BATCH 256 / Adam 1e-3 / wd 5e-4 / train 内 15% val / 8 组固定验证掩码；
- 读出特征：Z = [φ, X_S, X_S²]；闭式 ridge，α 在 train 内部 fit/val 选（网格 1e-3…1e2）；
- 对照：多目标 backbone（历史 75/95 号）、共享输出头、raw ridge、raw+x²、随机特征、元训练特征（L1）。

## 9. 随机种子

| 用途 | 种子 |
| --- | --- |
| 数据划分 | 42 |
| val 抽取 | 20260915 |
| backbone 训练 | 0（主），1、2（稳定性） |
| 真值重训 | 0（主），1（噪声估计） |
| ridge 内部 fit/val | 20260724（沿用 91 号） |

## 10. 指标冻结

**通用模型（Part A）**：主指标 M-MAE、τ-critical recall / false-safe rate；次指标 Kendall(M)、风险档位一致率、marginal-MAE、V-MAE、查询时延。
**M 指标（Part B）**：主指标 τ-critical 字段 recall、PR-AUC、false-safe rate；次指标 K 升级曲线、witness 可解释性、分级一致率、相对认证 MCI 下界的饱和。

τ-critical 定义（真值口径）：存在 |T|≤K 使 V̄(B∪T) ≤ τ < V̄(B∪T∪{i})。

## 11. 正式实验编号

| 号 | 内容 |
| --- | --- |
| 102 | single-target vs multi-target backbone（P0-2）+ 真值攻击器族扩展诊断 |
| 103 | 通用模型基线与消融（RQ-A2/A3）、V/边际/M 保真（RQ-A1/A4）、效率（RQ-A5） |
| 104 | 正式 M 全表（estimated vs exact、top-3 witness、档位、critical 状态） |
| 105 | 领域 baseline 与 τ-critical 检测（RQ-B1）、K 升级（RQ-B2）、witness 案例（RQ-B3）、基底（RQ-B4） |
| 106 | 字段分级表与发布审查案例（P0-7 / RQ-B5） |
| 107 | 总结目录：实验成色、遗留问题、论文写作方案 |

历史实验 98–101 号的真值与结论在协议一致处可复用，复用时必须注明差异（例如 100 号真值用多目标 DNN + 树，无单目标 DNN）。

---

## 12. 修订记录

> 协议正文冻结，但当正式实验推翻某条口径时，在此处记录修订，并注明证据。

| 日期 | 修订 | 证据 |
| --- | --- | --- |
| 2026-09-18 | §8 通用模型正式口径由"对每个目标 y 单独预学习 backbone"改为 **"multi-target 预学习（一次训练给 12 个目标）+ single-target 作为消融"**。读出仍对 (集合, 目标) 特定。 | 102 号：single-target 在 PJM/CAISO 全部主指标上不优于 multi-target（V-MAE 0.0395/0.0216 vs 0.0316/0.0170；M-MAE K=2 0.0514/0.0211 vs 0.0307/0.0156；τ-critical 召回 0.626/0.912 vs 0.701/0.920），且查询成本 12 倍、预学习成本 12 倍 |
| 2026-09-18 | §7 攻击器族确认为 **单目标 DNN ∪ 多目标 DNN ∪ 梯度提升树** 三者取 max（val 选择）。 | 102 号：单目标 DNN 比多目标 DNN 一致略强（规模 3 上 +0.0085/+0.0055，更强占比 75%/82%），但规模 3 上树模型最强（0.3746/0.4247）⟹ 单一攻击器会低估经验攻击能力 |
| 2026-09-18 | **取消时间块 / 时序划分评估**（原 P0-6）。数据划分一律随机 70/30；论文不声称跨时段泛化，局限中写明未评估时间外推。 | 用户决定（2026-09-18） |
