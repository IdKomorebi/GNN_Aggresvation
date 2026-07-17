# 注意力记录 — enc3_mlp / gat_dynamic

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8638** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力弱区分（略偏离均匀）**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.7317 | 0.2273 | 0.0556 | da_as_as_req_mw_synchronized_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.7523 | 0.2347 | 0.0667 | da_as_as_req_mw_synchronized_reserve |
| metered_load_mw | 30 | 0.8055 | 0.1956 | 0.0333 | forecast_load_mw_latest_available |
| gross_actual_interchange_mw | 19 | 0.8275 | 0.3514 | 0.0526 | gross_inadv_interchange_mw |
| total_gen | 27 | 0.8353 | 0.1675 | 0.0370 | forecast_load_mw_latest_available |
| total_lmp_da | 29 | 0.8405 | 0.2588 | 0.0345 | system_energy_price_da |
| congestion_price_rt | 19 | 0.8916 | 0.1678 | 0.0526 | total_lmp_rt |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.9215 | 0.1266 | 0.0417 | da_as_mcp_synchronized_reserve |
| marginal_loss_price_da | 23 | 0.9222 | 0.1416 | 0.0435 | system_energy_price_da |
| total_losses | 22 | 0.9227 | 0.1133 | 0.0455 | gen_fuel_coal_mw |
| net_actual_interchange_mw | 19 | 0.9260 | 0.1655 | 0.0526 | gross_sched_interchange_mw |
| congestion_price_da | 20 | 0.9886 | 0.0804 | 0.0500 | gen_fuel_nuclear_mw |