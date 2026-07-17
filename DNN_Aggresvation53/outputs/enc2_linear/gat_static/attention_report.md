# 注意力记录 — enc2_linear / gat_static

- confidential 目标数: 12
- 平均归一化注意力熵: **0.7563** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.5469 | 0.4270 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.6006 | 0.3119 | 0.0667 | da_as_nsr_mw_primary_reserve |
| total_lmp_da | 29 | 0.6851 | 0.4551 | 0.0345 | system_energy_price_da |
| metered_load_mw | 30 | 0.7186 | 0.3089 | 0.0333 | forecast_load_mw_latest_available |
| gross_actual_interchange_mw | 19 | 0.7217 | 0.4455 | 0.0526 | gross_inadv_interchange_mw |
| total_gen | 27 | 0.7559 | 0.2769 | 0.0370 | forecast_load_mw_latest_available |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8093 | 0.1883 | 0.0417 | gen_fuel_nuclear_mw |
| marginal_loss_price_da | 23 | 0.8106 | 0.2661 | 0.0435 | system_energy_price_da |
| total_losses | 22 | 0.8117 | 0.1450 | 0.0455 | gen_fuel_coal_mw |
| congestion_price_rt | 19 | 0.8225 | 0.2275 | 0.0526 | marginal_loss_price_rt |
| net_actual_interchange_mw | 19 | 0.8559 | 0.2651 | 0.0526 | gross_sched_interchange_mw |
| congestion_price_da | 20 | 0.9370 | 0.1476 | 0.0500 | system_energy_price_da |