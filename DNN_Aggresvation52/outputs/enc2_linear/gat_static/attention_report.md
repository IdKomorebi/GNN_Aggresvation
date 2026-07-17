# 注意力记录 — enc2_linear / gat_static

- confidential 目标数: 12
- 平均归一化注意力熵: **0.7655** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.5984 | 0.4332 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.6535 | 0.3781 | 0.0667 | da_as_nsr_mw_primary_reserve |
| total_lmp_da | 29 | 0.6901 | 0.4385 | 0.0345 | system_energy_price_da |
| metered_load_mw | 30 | 0.6991 | 0.2452 | 0.0333 | forecast_load_mw_latest_available |
| total_gen | 27 | 0.7270 | 0.2118 | 0.0370 | forecast_load_mw_latest_available |
| gross_actual_interchange_mw | 19 | 0.7515 | 0.4163 | 0.0526 | gross_inadv_interchange_mw |
| total_losses | 22 | 0.8040 | 0.1398 | 0.0455 | forecast_load_mw_latest_available |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8151 | 0.1994 | 0.0417 | gen_fuel_nuclear_mw |
| marginal_loss_price_da | 23 | 0.8202 | 0.2656 | 0.0435 | system_energy_price_da |
| congestion_price_rt | 19 | 0.8207 | 0.2170 | 0.0526 | total_lmp_rt |
| net_actual_interchange_mw | 19 | 0.8695 | 0.2108 | 0.0526 | gross_sched_interchange_mw |
| congestion_price_da | 20 | 0.9373 | 0.1641 | 0.0500 | system_energy_price_da |