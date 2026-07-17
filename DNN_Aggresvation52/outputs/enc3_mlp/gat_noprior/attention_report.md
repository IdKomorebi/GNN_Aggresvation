# 注意力记录 — enc3_mlp / gat_noprior

- confidential 目标数: 12
- 平均归一化注意力熵: **0.9476** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **7/12**
- 判定: **注意力弱区分（略偏离均匀）**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| da_as_total_mw_primary_reserve | 18 | 0.8780 | 0.2285 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.8897 | 0.2723 | 0.0667 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.9098 | 0.1993 | 0.0526 | gross_inadv_interchange_mw |
| net_actual_interchange_mw | 19 | 0.9343 | 0.1690 | 0.0526 | gross_sched_interchange_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.9382 | 0.1599 | 0.0417 | da_as_nsr_mw_primary_reserve |
| total_lmp_da | 29 | 0.9605 | 0.1559 | 0.0345 | da_as_nsr_mw_primary_reserve |
| total_losses | 22 | 0.9651 | 0.1226 | 0.0455 | gen_fuel_wind_mw |
| congestion_price_rt | 19 | 0.9686 | 0.1312 | 0.0526 | total_lmp_rt |
| congestion_price_da | 20 | 0.9772 | 0.0824 | 0.0500 | total_lmp_rt |
| metered_load_mw | 30 | 0.9814 | 0.0949 | 0.0333 | gross_sched_interchange_mw |
| total_gen | 27 | 0.9827 | 0.0969 | 0.0370 | gross_sched_interchange_mw |
| marginal_loss_price_da | 23 | 0.9862 | 0.0787 | 0.0435 | total_lmp_rt |