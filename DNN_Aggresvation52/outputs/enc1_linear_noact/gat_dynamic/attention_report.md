# 注意力记录 — enc1_linear_noact / gat_dynamic

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8408** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.7442 | 0.2559 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.7596 | 0.2138 | 0.0667 | da_as_nsr_mw_primary_reserve |
| metered_load_mw | 30 | 0.7828 | 0.2057 | 0.0333 | forecast_load_mw_latest_available |
| gross_actual_interchange_mw | 19 | 0.7903 | 0.3783 | 0.0526 | gross_inadv_interchange_mw |
| total_gen | 27 | 0.8080 | 0.1792 | 0.0370 | forecast_load_mw_latest_available |
| total_lmp_da | 29 | 0.8270 | 0.2779 | 0.0345 | system_energy_price_da |
| net_actual_interchange_mw | 19 | 0.8693 | 0.2532 | 0.0526 | gross_sched_interchange_mw |
| total_losses | 22 | 0.8696 | 0.1740 | 0.0455 | gen_fuel_coal_mw |
| congestion_price_rt | 19 | 0.8857 | 0.1886 | 0.0526 | marginal_loss_price_rt |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8971 | 0.1261 | 0.0417 | da_as_mcp_synchronized_reserve |
| marginal_loss_price_da | 23 | 0.8976 | 0.1511 | 0.0435 | system_energy_price_da |
| congestion_price_da | 20 | 0.9588 | 0.1248 | 0.0500 | gen_fuel_gas_mw |