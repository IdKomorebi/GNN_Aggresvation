# 注意力记录 — enc1_linear_noact / gat_noprior

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8341** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| metered_load_mw | 30 | 0.7597 | 0.2608 | 0.0333 | gross_inadv_interchange_mw |
| total_lmp_da | 29 | 0.7657 | 0.2604 | 0.0345 | gross_inadv_interchange_mw |
| total_gen | 27 | 0.7771 | 0.2623 | 0.0370 | gross_inadv_interchange_mw |
| congestion_price_rt | 19 | 0.8033 | 0.2543 | 0.0526 | marginal_loss_price_rt |
| congestion_price_da | 20 | 0.8273 | 0.2750 | 0.0500 | gross_inadv_interchange_mw |
| marginal_loss_price_da | 23 | 0.8349 | 0.2754 | 0.0435 | gross_inadv_interchange_mw |
| da_as_total_mw_primary_reserve | 18 | 0.8551 | 0.2746 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.8566 | 0.2640 | 0.0667 | da_as_nsr_mw_primary_reserve |
| total_losses | 22 | 0.8626 | 0.2743 | 0.0455 | gen_fuel_wind_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.8718 | 0.2508 | 0.0417 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.8833 | 0.2664 | 0.0526 | gross_inadv_interchange_mw |
| net_actual_interchange_mw | 19 | 0.9118 | 0.2028 | 0.0526 | gross_sched_interchange_mw |