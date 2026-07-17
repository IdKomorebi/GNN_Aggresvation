# 注意力记录 — enc1_linear_noact / gat_static

- confidential 目标数: 12
- 平均归一化注意力熵: **0.7458** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.5916 | 0.3430 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.6307 | 0.2945 | 0.0667 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.6536 | 0.5206 | 0.0526 | gross_inadv_interchange_mw |
| total_lmp_da | 29 | 0.7072 | 0.3683 | 0.0345 | system_energy_price_da |
| metered_load_mw | 30 | 0.7189 | 0.2045 | 0.0333 | forecast_load_mw_latest_available |
| total_gen | 27 | 0.7328 | 0.1763 | 0.0370 | forecast_load_mw_latest_available |
| congestion_price_rt | 19 | 0.7479 | 0.2829 | 0.0526 | total_lmp_rt |
| net_actual_interchange_mw | 19 | 0.7832 | 0.3390 | 0.0526 | gross_sched_interchange_mw |
| marginal_loss_price_da | 23 | 0.7917 | 0.2555 | 0.0435 | system_energy_price_da |
| total_losses | 22 | 0.8111 | 0.1713 | 0.0455 | gen_fuel_coal_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8237 | 0.1773 | 0.0417 | gen_fuel_nuclear_mw |
| congestion_price_da | 20 | 0.9575 | 0.1496 | 0.0500 | system_energy_price_da |