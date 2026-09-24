# PJM 现实口径｜pjm_gen_ic：自动生成的数值汇总（结论与解读见 CHANGELOG.md）

## 目标级汇总

| 目标                        | 规则依据                  |   M0均值 |   M1均值 |   M2均值 |   M3均值 |   背景升级均值_M2减M0 |   K饱和_M1除M2 |   K饱和_M2除M3 |   最强单字段V |   全部规模≤4集合最大V |   τ0.5_单字段即危险 |   τ0.5_K2关键 |   τ0.5_单看安全组合危险 |   τ0.5_危险小组合 |   τ0.5_单字段定级后仍暴露 |   τ0.5_最少扣留 |   τ0.7_单字段即危险 |   τ0.7_K2关键 |   τ0.7_单看安全组合危险 |   τ0.7_危险小组合 |   τ0.7_单字段定级后仍暴露 |   τ0.7_最少扣留 |   τ0.9_单字段即危险 |   τ0.9_K2关键 |   τ0.9_单看安全组合危险 |   τ0.9_危险小组合 |   τ0.9_单字段定级后仍暴露 |   τ0.9_最少扣留 |   V_MAE |   V_MAE_单种子 |   M2_MAE |   M2排序Spearman |   认证下界比_top1 |   认证下界比_top3 |
|:--------------------------|:----------------------|-------:|-------:|-------:|-------:|---------------:|------------:|------------:|---------:|--------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------:|------------:|---------:|---------------:|-------------:|-------------:|
| total_gen                 | 附表 6.57 发电总出力（次日披露）   | 0.2612 | 0.2894 | 0.3266 | 0.3276 |         0.0654 |      0.8862 |      0.9969 |   0.9743 |        0.9868 |             2 |          17 |              15 |           52 |               50 |           6 |             2 |          14 |              12 |           13 |               11 |           4 |             2 |           2 |               0 |            2 |                0 |           2 |  0.0483 |      0.056  |   0.0441 |         0.9142 |       0.8196 |       0.8859 |
| net_actual_interchange_mw | 附表 6.65 联络线输电情况（次日披露） | 0.116  | 0.1887 | 0.2108 | 0.2164 |         0.0948 |      0.8952 |      0.9742 |   0.4126 |        0.7422 |             0 |          17 |              17 |           33 |               33 |           2 |             0 |           3 |               3 |            1 |                1 |           1 |             0 |           0 |               0 |            0 |                0 |           0 |  0.0544 |      0.0656 |   0.0156 |         0.9853 |       0.961  |       0.9708 |

## 攻击器族构成

| 攻击器     |   被val选中比例 |
|:--------|-----------:|
| 单目标 DNN |     0.1878 |
| 多目标 DNN |     0.1813 |
| 梯度提升树   |     0.6309 |

## 按集合规模

|   规模 |   正式真值均值 |   通用模型MAE |
|-----:|---------:|----------:|
|    1 |   0.1886 |    0.0132 |
|    2 |   0.3449 |    0.0301 |
|    3 |   0.4691 |    0.0444 |
|    4 |   0.5648 |    0.0548 |

## 通用模型查询耗时

三种子 33.94 ms/集合；单种子 3.07 ms/集合；预训练 [35.0, 32.0, 34.0] 秒/种子


## 字段风险表（按目标、M^(2) 降序）

| 目标                        | 字段                                     |    r2 |    M0 |    M1 |    M2 |    M3 |   估计M2 |   升级_M2减M0 | K2见证                                                               |
|:--------------------------|:---------------------------------------|------:|------:|------:|------:|------:|-------:|-----------:|:-------------------------------------------------------------------|
| net_actual_interchange_mw | da_as_as_req_mw_primary_reserve        | 0.163 | 0.413 | 0.434 | 0.434 | 0.434 |  0.398 |      0.021 | da_as_mcp_synchronized_reserve                                     |
| net_actual_interchange_mw | gross_sched_interchange_mw             | 0.201 | 0.234 | 0.298 | 0.312 | 0.317 |  0.308 |      0.078 | total_pjm_assigned_reg + marginal_loss_price_rt                    |
| net_actual_interchange_mw | da_as_as_req_mw_thirty_minutes_reserve | 0.161 | 0.263 | 0.292 | 0.293 | 0.293 |  0.287 |      0.03  | da_as_mcp_synchronized_reserve + da_as_ss_mw_primary_reserve       |
| net_actual_interchange_mw | system_energy_price_da                 | 0.012 | 0.105 | 0.263 | 0.289 | 0.289 |  0.287 |      0.185 | forecast_load_mw_day_ahead + gross_sched_interchange_mw            |
| net_actual_interchange_mw | total_pjm_assigned_reg                 | 0.076 | 0.093 | 0.218 | 0.27  | 0.29  |  0.242 |      0.177 | total_pjm_self_sched_reg + gross_sched_interchange_mw              |
| net_actual_interchange_mw | forecast_load_mw_latest_available      | 0.028 | 0.068 | 0.227 | 0.251 | 0.254 |  0.244 |      0.183 | da_as_mcp_synchronized_reserve + system_energy_price_da            |
| net_actual_interchange_mw | forecast_load_mw_day_ahead             | 0.031 | 0.068 | 0.224 | 0.249 | 0.262 |  0.246 |      0.181 | da_as_mcp_synchronized_reserve + system_energy_price_da            |
| net_actual_interchange_mw | total_pjm_reg_purchases                | 0.059 | 0.177 | 0.224 | 0.245 | 0.254 |  0.238 |      0.068 | total_pjm_self_sched_reg + gross_sched_interchange_mw              |
| net_actual_interchange_mw | da_as_nsr_mw_primary_reserve           | 0.023 | 0.231 | 0.231 | 0.244 | 0.244 |  0.206 |      0.013 | da_as_mcp_synchronized_reserve + da_as_ss_mw_primary_reserve       |
| net_actual_interchange_mw | marginal_loss_price_rt                 | 0.01  | 0.046 | 0.161 | 0.206 | 0.206 |  0.205 |      0.161 | forecast_load_mw_latest_available + gross_sched_interchange_mw     |
| net_actual_interchange_mw | total_pjm_self_sched_reg               | 0.005 | 0.025 | 0.15  | 0.172 | 0.194 |  0.149 |      0.148 | total_pjm_assigned_reg + da_as_mcp_synchronized_reserve            |
| net_actual_interchange_mw | total_lmp_rt                           | 0.008 | 0.039 | 0.121 | 0.146 | 0.146 |  0.141 |      0.108 | forecast_load_mw_latest_available + gross_sched_interchange_mw     |
| net_actual_interchange_mw | da_as_mcp_synchronized_reserve         | 0.009 | 0.025 | 0.08  | 0.117 | 0.117 |  0.091 |      0.092 | forecast_load_mw_day_ahead + da_as_ss_mw_primary_reserve           |
| net_actual_interchange_mw | rmpcp                                  | 0.012 | 0.073 | 0.077 | 0.106 | 0.11  |  0.076 |      0.033 | da_as_ss_mw_primary_reserve + system_energy_price_da               |
| net_actual_interchange_mw | da_as_ss_mw_primary_reserve            | 0.001 | 0.068 | 0.097 | 0.102 | 0.122 |  0.064 |      0.034 | forecast_load_mw_latest_available + da_as_mcp_synchronized_reserve |
| net_actual_interchange_mw | rmccp                                  | 0.016 | 0.029 | 0.063 | 0.074 | 0.074 |  0.074 |      0.045 | forecast_load_mw_latest_available + gross_sched_interchange_mw     |
| net_actual_interchange_mw | da_as_mcp_primary_reserve              | 0.011 | 0.015 | 0.048 | 0.072 | 0.072 |  0.063 |      0.057 | forecast_load_mw_day_ahead + da_as_ss_mw_primary_reserve           |
| total_gen                 | forecast_load_mw_latest_available      | 0.973 | 0.974 | 0.974 | 0.974 | 0.974 |  0.974 |      0     | ∅                                                                  |
| total_gen                 | forecast_load_mw_day_ahead             | 0.966 | 0.968 | 0.968 | 0.968 | 0.968 |  0.968 |      0     | ∅                                                                  |
| total_gen                 | da_as_nsr_mw_primary_reserve           | 0.007 | 0.415 | 0.415 | 0.454 | 0.454 |  0.223 |      0.038 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | system_energy_price_da                 | 0.309 | 0.382 | 0.42  | 0.43  | 0.43  |  0.439 |      0.048 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve  |
| total_gen                 | marginal_loss_price_rt                 | 0.154 | 0.321 | 0.355 | 0.358 | 0.36  |  0.372 |      0.037 | total_pjm_self_sched_reg + total_pjm_assigned_reg                  |
| total_gen                 | da_as_as_req_mw_primary_reserve        | 0.137 | 0.239 | 0.283 | 0.356 | 0.356 |  0.244 |      0.117 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | total_lmp_rt                           | 0.142 | 0.28  | 0.305 | 0.317 | 0.317 |  0.312 |      0.037 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve  |
| total_gen                 | da_as_as_req_mw_thirty_minutes_reserve | 0.133 | 0.169 | 0.214 | 0.291 | 0.291 |  0.238 |      0.122 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | total_pjm_reg_purchases                | 0.005 | 0.129 | 0.145 | 0.21  | 0.21  |  0.172 |      0.081 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | da_as_mcp_synchronized_reserve         | 0.077 | 0.113 | 0.152 | 0.166 | 0.172 |  0.204 |      0.054 | da_as_as_req_mw_primary_reserve + gross_sched_interchange_mw       |
| total_gen                 | gross_sched_interchange_mw             | 0.069 | 0.096 | 0.113 | 0.165 | 0.165 |  0.126 |      0.07  | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | rmpcp                                  | 0.001 | 0.081 | 0.095 | 0.164 | 0.164 |  0.109 |      0.082 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | total_pjm_assigned_reg                 | 0.002 | 0.018 | 0.075 | 0.149 | 0.149 |  0.103 |      0.13  | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | da_as_mcp_primary_reserve              | 0.061 | 0.117 | 0.124 | 0.141 | 0.15  |  0.163 |      0.025 | total_pjm_self_sched_reg + total_pjm_assigned_reg                  |
| total_gen                 | rmccp                                  | 0.006 | 0.044 | 0.089 | 0.141 | 0.141 |  0.122 |      0.097 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | total_pjm_self_sched_reg               | 0.005 | 0.016 | 0.073 | 0.137 | 0.137 |  0.109 |      0.121 | da_as_mcp_primary_reserve + da_as_ss_mw_primary_reserve            |
| total_gen                 | da_as_ss_mw_primary_reserve            | 0.008 | 0.077 | 0.119 | 0.131 | 0.131 |  0.091 |      0.054 | rmccp + total_pjm_self_sched_reg                                   |