# 注意力记录 — enc2_linear / gat_noprior

- confidential 目标数: 12
- 平均归一化注意力熵: **0.9438** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **8/12**
- 判定: **注意力弱区分（略偏离均匀）**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| total_lmp_da | 29 | 0.8516 | 0.2157 | 0.0345 | gross_inadv_interchange_mw |
| da_as_total_mw_primary_reserve | 18 | 0.8644 | 0.2690 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.8807 | 0.2819 | 0.0667 | da_as_nsr_mw_primary_reserve |
| gross_actual_interchange_mw | 19 | 0.9451 | 0.1925 | 0.0526 | gross_inadv_interchange_mw |
| marginal_loss_price_da | 23 | 0.9591 | 0.1017 | 0.0435 | gross_inadv_interchange_mw |
| congestion_price_rt | 19 | 0.9626 | 0.1504 | 0.0526 | total_lmp_rt |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.9657 | 0.1101 | 0.0417 | system_energy_price_da |
| net_actual_interchange_mw | 19 | 0.9787 | 0.0903 | 0.0526 | forecast_load_mw_latest_available |
| metered_load_mw | 30 | 0.9787 | 0.0757 | 0.0333 | forecast_load_mw_latest_available |
| congestion_price_da | 20 | 0.9791 | 0.0999 | 0.0500 | system_energy_price_da |
| total_gen | 27 | 0.9800 | 0.0800 | 0.0370 | forecast_load_mw_latest_available |
| total_losses | 22 | 0.9801 | 0.0771 | 0.0455 | forecast_load_mw_latest_available |