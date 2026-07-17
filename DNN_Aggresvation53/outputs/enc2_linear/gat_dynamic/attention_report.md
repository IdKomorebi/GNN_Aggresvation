# 注意力记录 — enc2_linear / gat_dynamic

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8377** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_synchronized_reserve | 15 | 0.7059 | 0.2245 | 0.0667 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_primary_reserve | 18 | 0.7163 | 0.2266 | 0.0556 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.7761 | 0.4037 | 0.0526 | gross_inadv_interchange_mw |
| total_lmp_da | 29 | 0.8101 | 0.2870 | 0.0345 | system_energy_price_da |
| metered_load_mw | 30 | 0.8154 | 0.1993 | 0.0333 | forecast_load_mw_latest_available |
| total_gen | 27 | 0.8408 | 0.1707 | 0.0370 | forecast_load_mw_latest_available |
| congestion_price_rt | 19 | 0.8637 | 0.1666 | 0.0526 | total_lmp_rt |
| total_losses | 22 | 0.8769 | 0.1319 | 0.0455 | gen_fuel_coal_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8823 | 0.1655 | 0.0417 | gen_fuel_nuclear_mw |
| net_actual_interchange_mw | 19 | 0.8846 | 0.1663 | 0.0526 | gen_fuel_nuclear_mw |
| marginal_loss_price_da | 23 | 0.9034 | 0.1606 | 0.0435 | system_energy_price_da |
| congestion_price_da | 20 | 0.9766 | 0.1053 | 0.0500 | gen_fuel_nuclear_mw |