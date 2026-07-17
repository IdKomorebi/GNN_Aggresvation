# 注意力记录 — enc2_linear / gat_dynamic

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8449** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.7366 | 0.2217 | 0.0556 | da_as_as_req_mw_synchronized_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.7384 | 0.2315 | 0.0667 | da_as_as_req_mw_synchronized_reserve |
| metered_load_mw | 30 | 0.7828 | 0.1949 | 0.0333 | forecast_load_mw_latest_available |
| gross_actual_interchange_mw | 19 | 0.7967 | 0.3809 | 0.0526 | gross_inadv_interchange_mw |
| total_gen | 27 | 0.8054 | 0.1687 | 0.0370 | forecast_load_mw_latest_available |
| total_lmp_da | 29 | 0.8259 | 0.2785 | 0.0345 | system_energy_price_da |
| congestion_price_rt | 19 | 0.8789 | 0.1620 | 0.0526 | total_lmp_rt |
| total_losses | 22 | 0.8873 | 0.1472 | 0.0455 | gen_fuel_coal_mw |
| net_actual_interchange_mw | 19 | 0.8889 | 0.1581 | 0.0526 | gross_sched_interchange_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.9026 | 0.1500 | 0.0417 | gen_fuel_nuclear_mw |
| marginal_loss_price_da | 23 | 0.9197 | 0.1320 | 0.0435 | system_energy_price_da |
| congestion_price_da | 20 | 0.9762 | 0.1024 | 0.0500 | gen_fuel_gas_mw |