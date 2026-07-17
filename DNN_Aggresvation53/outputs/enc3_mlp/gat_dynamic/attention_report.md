# 注意力记录 — enc3_mlp / gat_dynamic

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8306** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.7205 | 0.2384 | 0.0556 | da_as_as_req_mw_synchronized_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.7291 | 0.2534 | 0.0667 | da_as_as_req_mw_synchronized_reserve |
| gross_actual_interchange_mw | 19 | 0.7448 | 0.4593 | 0.0526 | gross_inadv_interchange_mw |
| metered_load_mw | 30 | 0.7712 | 0.2217 | 0.0333 | forecast_load_mw_latest_available |
| total_gen | 27 | 0.7945 | 0.1887 | 0.0370 | forecast_load_mw_latest_available |
| total_lmp_da | 29 | 0.8010 | 0.3111 | 0.0345 | system_energy_price_da |
| congestion_price_rt | 19 | 0.8557 | 0.1873 | 0.0526 | total_lmp_rt |
| net_actual_interchange_mw | 19 | 0.8854 | 0.1980 | 0.0526 | gross_sched_interchange_mw |
| total_losses | 22 | 0.8901 | 0.1428 | 0.0455 | gen_fuel_coal_mw |
| marginal_loss_price_da | 23 | 0.8931 | 0.1697 | 0.0435 | system_energy_price_da |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8991 | 0.1408 | 0.0417 | da_as_mcp_synchronized_reserve |
| congestion_price_da | 20 | 0.9833 | 0.0869 | 0.0500 | gen_fuel_gas_mw |