# 注意力记录 — enc2_linear / gat_noprior

- confidential 目标数: 12
- 平均归一化注意力熵: **0.9346** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力弱区分（略偏离均匀）**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| total_lmp_da | 29 | 0.9159 | 0.1530 | 0.0345 | gross_inadv_interchange_mw |
| marginal_loss_price_da | 23 | 0.9165 | 0.2191 | 0.0435 | gross_inadv_interchange_mw |
| congestion_price_rt | 19 | 0.9244 | 0.2049 | 0.0526 | total_lmp_rt |
| gross_actual_interchange_mw | 19 | 0.9266 | 0.1873 | 0.0526 | gross_inadv_interchange_mw |
| da_as_total_mw_synchronized_reserve | 15 | 0.9282 | 0.2182 | 0.0667 | da_as_nsr_mw_primary_reserve |
| congestion_price_da | 20 | 0.9357 | 0.1858 | 0.0500 | gross_inadv_interchange_mw |
| net_actual_interchange_mw | 19 | 0.9396 | 0.1589 | 0.0526 | gross_sched_interchange_mw |
| metered_load_mw | 30 | 0.9409 | 0.1146 | 0.0333 | gross_inadv_interchange_mw |
| da_as_total_mw_primary_reserve | 18 | 0.9432 | 0.1615 | 0.0556 | da_as_nsr_mw_primary_reserve |
| total_gen | 27 | 0.9460 | 0.1166 | 0.0370 | gross_inadv_interchange_mw |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.9478 | 0.1359 | 0.0417 | da_as_nsr_mw_primary_reserve |
| total_losses | 22 | 0.9499 | 0.1442 | 0.0455 | gross_inadv_interchange_mw |