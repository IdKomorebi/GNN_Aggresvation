# 注意力记录 — enc3_mlp / gat_noprior

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8905** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力弱区分（略偏离均匀）**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_synchronized_reserve | 15 | 0.8034 | 0.3994 | 0.0667 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_primary_reserve | 18 | 0.8404 | 0.3321 | 0.0556 | da_as_nsr_mw_primary_reserve |
| total_lmp_da | 29 | 0.8415 | 0.3143 | 0.0345 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8433 | 0.3095 | 0.0417 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.8987 | 0.1878 | 0.0526 | gross_inadv_interchange_mw |
| congestion_price_da | 20 | 0.9123 | 0.1462 | 0.0500 | gross_inadv_interchange_mw |
| total_losses | 22 | 0.9124 | 0.1476 | 0.0455 | gen_fuel_nuclear_pct |
| net_actual_interchange_mw | 19 | 0.9199 | 0.1802 | 0.0526 | gross_sched_interchange_mw |
| total_gen | 27 | 0.9231 | 0.1148 | 0.0370 | gen_fuel_nuclear_pct |
| marginal_loss_price_da | 23 | 0.9234 | 0.1385 | 0.0435 | gen_fuel_nuclear_pct |
| metered_load_mw | 30 | 0.9267 | 0.1087 | 0.0333 | gen_fuel_nuclear_pct |
| congestion_price_rt | 19 | 0.9410 | 0.1741 | 0.0526 | gen_fuel_nuclear_pct |