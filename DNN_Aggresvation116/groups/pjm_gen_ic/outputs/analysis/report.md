# PJM 现实口径（补全候选）｜pjm_gen_ic：自动生成的数值汇总（结论与解读见 CHANGELOG.md）

## 目标级汇总

| 目标                        | 规则依据                  |   M0均值 |   M1均值 |   M2均值 |   M3均值 |   背景升级均值_M2减M0 |   K饱和_M1除M2 |   K饱和_M2除M3 |   最强单字段V |   全部规模≤4集合最大V |   τ0.5_单字段即危险 |   τ0.5_K2关键 |   τ0.5_单看安全组合危险 |   τ0.5_危险小组合 |   τ0.5_单字段定级后仍暴露 |   τ0.5_最少扣留 |   τ0.7_单字段即危险 |   τ0.7_K2关键 |   τ0.7_单看安全组合危险 |   τ0.7_危险小组合 |   τ0.7_单字段定级后仍暴露 |   τ0.7_最少扣留 |   τ0.9_单字段即危险 |   τ0.9_K2关键 |   τ0.9_单看安全组合危险 |   τ0.9_危险小组合 |   τ0.9_单字段定级后仍暴露 |   τ0.9_最少扣留 |   V_MAE |   V_MAE_单种子 |   M2_MAE |   M2排序Spearman |   认证下界比_top1 |   认证下界比_top3 |
|:--------------------------|:----------------------|-------:|-------:|-------:|-------:|---------------:|------------:|------------:|---------:|--------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------:|------------:|---------:|---------------:|-------------:|-------------:|
| total_gen                 | 附表 6.57 发电总出力（次日披露）   | 0.2573 | 0.3072 | 0.3352 | 0.3365 |         0.0779 |      0.9164 |      0.9963 |   0.9742 |        0.987  |             2 |          24 |              22 |          179 |              177 |          11 |             2 |          20 |              18 |           39 |               37 |           5 |             2 |           2 |               0 |            2 |                0 |           2 |  0.0533 |      0.0586 |   0.0428 |         0.8722 |       0.8641 |       0.9288 |
| net_actual_interchange_mw | 附表 6.65 联络线输电情况（次日披露） | 0.1201 | 0.1979 | 0.2213 | 0.2276 |         0.1012 |      0.8945 |      0.9724 |   0.4126 |        0.7484 |             0 |          24 |              24 |          102 |              102 |           3 |             0 |           3 |               3 |            1 |                1 |           1 |             0 |           0 |               0 |            0 |                0 |           0 |  0.0541 |      0.0582 |   0.0236 |         0.9452 |       0.9062 |       0.9793 |

## 攻击器族构成

| 攻击器     |   被val选中比例 |
|:--------|-----------:|
| 单目标 DNN |     0.1852 |
| 多目标 DNN |     0.1493 |
| 梯度提升树   |     0.6654 |

## 按集合规模

|   规模 |   正式真值均值 |   通用模型MAE |
|-----:|---------:|----------:|
|    1 |   0.1887 |    0.0209 |
|    2 |   0.3428 |    0.0387 |
|    3 |   0.4626 |    0.0561 |
|    4 |   0.5535 |  nan      |

## 通用模型查询耗时

新主口径 E（三种子预测平均）114 ms/集合


## 字段风险表（按目标、M^(2) 降序）

| 目标                        | 字段                                     |    r2 |    M0 |    M1 |    M2 |    M3 |   估计M2 |   升级_M2减M0 | K2见证                                                                |
|:--------------------------|:---------------------------------------|------:|------:|------:|------:|------:|-------:|-----------:|:--------------------------------------------------------------------|
| net_actual_interchange_mw | da_as_as_req_mw_primary_reserve        | 0.163 | 0.413 | 0.434 | 0.434 | 0.434 |  0.369 |      0.021 | da_as_mcp_synchronized_reserve                                      |
| net_actual_interchange_mw | da_as_total_mw_primary_reserve         | 0.136 | 0.294 | 0.343 | 0.343 | 0.343 |  0.252 |      0.049 | marginal_loss_price_da                                              |
| net_actual_interchange_mw | gross_sched_interchange_mw             | 0.201 | 0.234 | 0.308 | 0.313 | 0.342 |  0.306 |      0.079 | total_pjm_assigned_reg + marginal_loss_price_rt                     |
| net_actual_interchange_mw | forecast_load_mw_day_ahead             | 0.031 | 0.072 | 0.228 | 0.294 | 0.302 |  0.292 |      0.222 | system_energy_price_da + marginal_loss_price_da                     |
| net_actual_interchange_mw | forecast_load_mw_latest_available      | 0.028 | 0.071 | 0.227 | 0.294 | 0.299 |  0.286 |      0.223 | system_energy_price_da + marginal_loss_price_da                     |
| net_actual_interchange_mw | da_as_as_req_mw_thirty_minutes_reserve | 0.161 | 0.264 | 0.292 | 0.293 | 0.297 |  0.31  |      0.029 | da_as_mcp_synchronized_reserve + da_as_ss_mw_primary_reserve        |
| net_actual_interchange_mw | system_energy_price_da                 | 0.012 | 0.103 | 0.259 | 0.292 | 0.292 |  0.284 |      0.189 | forecast_load_mw_day_ahead + gross_sched_interchange_mw             |
| net_actual_interchange_mw | da_as_total_mw_synchronized_reserve    | 0.073 | 0.2   | 0.281 | 0.291 | 0.294 |  0.218 |      0.091 | da_as_mcp_primary_reserve + marginal_loss_price_da                  |
| net_actual_interchange_mw | total_pjm_assigned_reg                 | 0.076 | 0.095 | 0.215 | 0.277 | 0.286 |  0.245 |      0.182 | total_pjm_self_sched_reg + gross_sched_interchange_mw               |
| net_actual_interchange_mw | da_as_total_mw_thirty_minutes_reserve  | 0.12  | 0.139 | 0.259 | 0.276 | 0.276 |  0.256 |      0.137 | forecast_load_mw_latest_available + forecast_load_mw_day_ahead      |
| net_actual_interchange_mw | total_lmp_da                           | 0.008 | 0.097 | 0.247 | 0.275 | 0.275 |  0.27  |      0.178 | forecast_load_mw_day_ahead + gross_sched_interchange_mw             |
| net_actual_interchange_mw | total_pjm_reg_purchases                | 0.059 | 0.178 | 0.222 | 0.246 | 0.259 |  0.229 |      0.068 | total_pjm_self_sched_reg + gross_sched_interchange_mw               |
| net_actual_interchange_mw | da_as_nsr_mw_primary_reserve           | 0.023 | 0.231 | 0.232 | 0.244 | 0.244 |  0.179 |      0.013 | da_as_mcp_synchronized_reserve + da_as_ss_mw_primary_reserve        |
| net_actual_interchange_mw | marginal_loss_price_rt                 | 0.01  | 0.036 | 0.159 | 0.207 | 0.207 |  0.2   |      0.171 | forecast_load_mw_day_ahead + gross_sched_interchange_mw             |
| net_actual_interchange_mw | total_pjm_self_sched_reg               | 0.005 | 0.026 | 0.146 | 0.177 | 0.186 |  0.148 |      0.152 | total_pjm_assigned_reg + da_as_mcp_synchronized_reserve             |
| net_actual_interchange_mw | marginal_loss_price_da                 | 0.005 | 0.064 | 0.168 | 0.177 | 0.177 |  0.165 |      0.113 | forecast_load_mw_day_ahead + gross_sched_interchange_mw             |
| net_actual_interchange_mw | total_lmp_rt                           | 0.008 | 0.04  | 0.117 | 0.148 | 0.148 |  0.137 |      0.109 | forecast_load_mw_day_ahead + gross_sched_interchange_mw             |
| net_actual_interchange_mw | congestion_price_da                    | 0.006 | 0.081 | 0.123 | 0.13  | 0.13  |  0.114 |      0.049 | forecast_load_mw_day_ahead + total_pjm_self_sched_reg               |
| net_actual_interchange_mw | da_as_mcp_synchronized_reserve         | 0.009 | 0.025 | 0.087 | 0.117 | 0.117 |  0.108 |      0.092 | forecast_load_mw_day_ahead + da_as_ss_mw_primary_reserve            |
| net_actual_interchange_mw | rmpcp                                  | 0.012 | 0.073 | 0.084 | 0.116 | 0.145 |  0.073 |      0.042 | da_as_ss_mw_primary_reserve + total_lmp_da                          |
| net_actual_interchange_mw | da_as_ss_mw_primary_reserve            | 0.001 | 0.068 | 0.096 | 0.102 | 0.125 |  0.095 |      0.034 | forecast_load_mw_latest_available + da_as_mcp_synchronized_reserve  |
| net_actual_interchange_mw | congestion_price_rt                    | 0.025 | 0.034 | 0.07  | 0.098 | 0.098 |  0.09  |      0.064 | forecast_load_mw_day_ahead + gross_sched_interchange_mw             |
| net_actual_interchange_mw | da_as_mcp_primary_reserve              | 0.011 | 0.015 | 0.096 | 0.096 | 0.096 |  0.106 |      0.08  | da_as_total_mw_synchronized_reserve                                 |
| net_actual_interchange_mw | rmccp                                  | 0.016 | 0.028 | 0.059 | 0.07  | 0.09  |  0.072 |      0.042 | forecast_load_mw_latest_available + gross_sched_interchange_mw      |
| total_gen                 | forecast_load_mw_latest_available      | 0.973 | 0.974 | 0.974 | 0.974 | 0.974 |  0.974 |      0     | ∅                                                                   |
| total_gen                 | forecast_load_mw_day_ahead             | 0.966 | 0.968 | 0.968 | 0.968 | 0.968 |  0.968 |      0     | ∅                                                                   |
| total_gen                 | total_lmp_da                           | 0.328 | 0.41  | 0.46  | 0.489 | 0.489 |  0.483 |      0.079 | total_pjm_assigned_reg + da_as_total_mw_thirty_minutes_reserve      |
| total_gen                 | system_energy_price_da                 | 0.309 | 0.382 | 0.421 | 0.468 | 0.468 |  0.459 |      0.086 | total_pjm_assigned_reg + da_as_total_mw_thirty_minutes_reserve      |
| total_gen                 | da_as_nsr_mw_primary_reserve           | 0.007 | 0.415 | 0.415 | 0.454 | 0.454 |  0.18  |      0.038 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | da_as_total_mw_synchronized_reserve    | 0.066 | 0.331 | 0.354 | 0.422 | 0.422 |  0.264 |      0.091 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | da_as_as_req_mw_primary_reserve        | 0.137 | 0.239 | 0.404 | 0.404 | 0.404 |  0.369 |      0.164 | da_as_total_mw_thirty_minutes_reserve                               |
| total_gen                 | da_as_total_mw_primary_reserve         | 0.083 | 0.253 | 0.39  | 0.39  | 0.39  |  0.339 |      0.137 | da_as_total_mw_thirty_minutes_reserve                               |
| total_gen                 | marginal_loss_price_rt                 | 0.154 | 0.327 | 0.354 | 0.36  | 0.365 |  0.362 |      0.033 | total_pjm_self_sched_reg + total_pjm_assigned_reg                   |
| total_gen                 | da_as_as_req_mw_thirty_minutes_reserve | 0.133 | 0.169 | 0.334 | 0.336 | 0.336 |  0.339 |      0.167 | total_pjm_self_sched_reg + da_as_total_mw_thirty_minutes_reserve    |
| total_gen                 | total_lmp_rt                           | 0.142 | 0.284 | 0.303 | 0.323 | 0.323 |  0.334 |      0.038 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| total_gen                 | da_as_total_mw_thirty_minutes_reserve  | 0.002 | 0.117 | 0.282 | 0.318 | 0.318 |  0.298 |      0.201 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| total_gen                 | marginal_loss_price_da                 | 0.245 | 0.296 | 0.315 | 0.315 | 0.315 |  0.311 |      0.019 | total_pjm_self_sched_reg                                            |
| total_gen                 | congestion_price_da                    | 0.116 | 0.23  | 0.241 | 0.25  | 0.256 |  0.248 |      0.021 | da_as_as_req_mw_thirty_minutes_reserve + gross_sched_interchange_mw |
| total_gen                 | total_pjm_reg_purchases                | 0.005 | 0.132 | 0.153 | 0.21  | 0.21  |  0.164 |      0.078 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | da_as_mcp_synchronized_reserve         | 0.077 | 0.113 | 0.152 | 0.166 | 0.172 |  0.218 |      0.054 | da_as_as_req_mw_primary_reserve + gross_sched_interchange_mw        |
| total_gen                 | gross_sched_interchange_mw             | 0.069 | 0.092 | 0.123 | 0.165 | 0.165 |  0.121 |      0.074 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | rmpcp                                  | 0.001 | 0.081 | 0.101 | 0.164 | 0.164 |  0.075 |      0.082 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | congestion_price_rt                    | 0.064 | 0.096 | 0.15  | 0.162 | 0.168 |  0.168 |      0.066 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| total_gen                 | da_as_mcp_primary_reserve              | 0.061 | 0.111 | 0.124 | 0.149 | 0.149 |  0.181 |      0.038 | total_pjm_self_sched_reg + total_pjm_assigned_reg                   |
| total_gen                 | total_pjm_assigned_reg                 | 0.002 | 0.019 | 0.078 | 0.149 | 0.156 |  0.087 |      0.13  | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | rmccp                                  | 0.006 | 0.046 | 0.089 | 0.141 | 0.141 |  0.1   |      0.095 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | total_pjm_self_sched_reg               | 0.005 | 0.012 | 0.072 | 0.137 | 0.137 |  0.093 |      0.125 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve             |
| total_gen                 | da_as_ss_mw_primary_reserve            | 0.008 | 0.077 | 0.119 | 0.133 | 0.133 |  0.1   |      0.056 | rmccp + total_pjm_self_sched_reg                                    |