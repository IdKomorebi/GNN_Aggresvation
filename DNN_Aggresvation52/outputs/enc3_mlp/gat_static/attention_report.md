# 注意力记录 — enc3_mlp / gat_static

- confidential 目标数: 12
- 平均归一化注意力熵: **0.7490** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.5970 | 0.4263 | 0.0556 | da_as_nsr_mw_primary_reserve |
| total_lmp_da | 29 | 0.6389 | 0.4698 | 0.0345 | system_energy_price_da |
| da_as_total_mw_synchronized_reserve | 15 | 0.6464 | 0.3685 | 0.0667 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.6711 | 0.5204 | 0.0526 | gross_inadv_interchange_mw |
| metered_load_mw | 30 | 0.6907 | 0.2456 | 0.0333 | forecast_load_mw_latest_available |
| total_gen | 27 | 0.7234 | 0.2178 | 0.0370 | forecast_load_mw_latest_available |
| marginal_loss_price_da | 23 | 0.7862 | 0.3009 | 0.0435 | system_energy_price_da |
| congestion_price_rt | 19 | 0.8095 | 0.2628 | 0.0526 | total_lmp_rt |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8174 | 0.1919 | 0.0417 | gen_fuel_nuclear_mw |
| total_losses | 22 | 0.8246 | 0.1468 | 0.0455 | gen_fuel_coal_mw |
| net_actual_interchange_mw | 19 | 0.8515 | 0.2711 | 0.0526 | gross_sched_interchange_mw |
| congestion_price_da | 20 | 0.9309 | 0.1714 | 0.0500 | system_energy_price_da |