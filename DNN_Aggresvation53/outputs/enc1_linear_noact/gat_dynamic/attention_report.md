# 注意力记录 — enc1_linear_noact / gat_dynamic

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8382** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_synchronized_reserve | 15 | 0.6922 | 0.2298 | 0.0667 | da_as_as_req_mw_synchronized_reserve |
| da_as_total_mw_primary_reserve | 18 | 0.6970 | 0.2571 | 0.0556 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.7785 | 0.4214 | 0.0526 | gross_inadv_interchange_mw |
| metered_load_mw | 30 | 0.7912 | 0.2010 | 0.0333 | forecast_load_mw_latest_available |
| total_gen | 27 | 0.8120 | 0.1768 | 0.0370 | forecast_load_mw_latest_available |
| total_lmp_da | 29 | 0.8266 | 0.2691 | 0.0345 | system_energy_price_da |
| total_losses | 22 | 0.8714 | 0.1587 | 0.0455 | gen_fuel_coal_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8931 | 0.1316 | 0.0417 | da_as_nsr_mw_primary_reserve |
| congestion_price_rt | 19 | 0.8959 | 0.1903 | 0.0526 | total_lmp_rt |
| marginal_loss_price_da | 23 | 0.9088 | 0.1723 | 0.0435 | system_energy_price_da |
| net_actual_interchange_mw | 19 | 0.9152 | 0.1787 | 0.0526 | gross_sched_interchange_mw |
| congestion_price_da | 20 | 0.9761 | 0.0844 | 0.0500 | system_energy_price_da |