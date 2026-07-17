# 注意力记录 — enc1_linear_noact / gat_noprior

- confidential 目标数: 12
- 平均归一化注意力熵: **0.8007** (1=完全平均, 0=完全集中)
- 接近均匀(熵≥0.95)的目标数: **0/12**
- 判定: **注意力有区分性（明显偏离均匀），起到了选择邻居的作用**

注意力越平均(熵→1)说明 q·k 没学出有用的邻居偏好、退化为均匀聚合；
越集中(熵小)说明注意力真正在选择信息量大的 general 源。

| target | degree | entropy_norm | max_weight | uniform_weight | top_source |
|---|---:|---:|---:|---:|---|
| total_lmp_da | 29 | 0.7426 | 0.2548 | 0.0345 | system_energy_price_da |
| marginal_loss_price_da | 23 | 0.7560 | 0.2583 | 0.0435 | system_energy_price_da |
| da_as_total_mw_thirty_minutes_reserve | 24 | 0.7608 | 0.2580 | 0.0417 | system_energy_price_da |
| gross_actual_interchange_mw | 19 | 0.7658 | 0.2621 | 0.0526 | system_energy_price_da |
| metered_load_mw | 30 | 0.7741 | 0.2538 | 0.0333 | system_energy_price_da |
| total_gen | 27 | 0.7839 | 0.2548 | 0.0370 | system_energy_price_da |
| congestion_price_da | 20 | 0.7905 | 0.2599 | 0.0500 | system_energy_price_da |
| total_losses | 22 | 0.7906 | 0.2593 | 0.0455 | system_energy_price_da |
| congestion_price_rt | 19 | 0.8392 | 0.2714 | 0.0526 | system_energy_price_da |
| net_actual_interchange_mw | 19 | 0.8508 | 0.2622 | 0.0526 | system_energy_price_da |
| da_as_total_mw_primary_reserve | 18 | 0.8771 | 0.2618 | 0.0556 | da_as_nsr_mw_primary_reserve |
| da_as_total_mw_synchronized_reserve | 15 | 0.8774 | 0.2675 | 0.0667 | da_as_nsr_mw_primary_reserve |