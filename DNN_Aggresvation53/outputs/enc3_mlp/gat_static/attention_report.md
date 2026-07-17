# 注意力记录 — enc3_mlp / gat_static

- confidential 目标数: 12
- 平均归一化注意力熵: **0.7648** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **1/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.5442 | 0.4155 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.5892 | 0.3164 | 0.0667 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.6215 | 0.4698 | 0.0526 | gross_inadv_interchange_mw |
| total_lmp_da | 29 | 0.7772 | 0.3049 | 0.0345 | system_energy_price_da |
| metered_load_mw | 30 | 0.7788 | 0.1809 | 0.0333 | forecast_load_mw_latest_available |
| congestion_price_rt | 19 | 0.7924 | 0.2369 | 0.0526 | marginal_loss_price_rt |
| net_actual_interchange_mw | 19 | 0.7933 | 0.3246 | 0.0526 | gross_sched_interchange_mw |
| total_gen | 27 | 0.8067 | 0.1581 | 0.0370 | forecast_load_mw_day_ahead |
| total_losses | 22 | 0.8179 | 0.1514 | 0.0455 | marginal_loss_price_rt |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8344 | 0.1788 | 0.0417 | gen_fuel_nuclear_mw |
| marginal_loss_price_da | 23 | 0.8547 | 0.1812 | 0.0435 | marginal_loss_price_rt |
| congestion_price_da | 20 | 0.9668 | 0.1004 | 0.0500 | system_energy_price_da |