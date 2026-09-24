# PJM 现实口径（补全候选）｜pjm_load：自动生成的数值汇总（结论与解读见 CHANGELOG.md）

## 目标级汇总

| 目标              | 规则依据               |   M0均值 |   M1均值 |   M2均值 |   M3均值 |   背景升级均值_M2减M0 |   K饱和_M1除M2 |   K饱和_M2除M3 |   最强单字段V |   全部规模≤4集合最大V |   τ0.5_单字段即危险 |   τ0.5_K2关键 |   τ0.5_单看安全组合危险 |   τ0.5_危险小组合 |   τ0.5_单字段定级后仍暴露 |   τ0.5_最少扣留 |   τ0.7_单字段即危险 |   τ0.7_K2关键 |   τ0.7_单看安全组合危险 |   τ0.7_危险小组合 |   τ0.7_单字段定级后仍暴露 |   τ0.7_最少扣留 |   τ0.9_单字段即危险 |   τ0.9_K2关键 |   τ0.9_单看安全组合危险 |   τ0.9_危险小组合 |   τ0.9_单字段定级后仍暴露 |   τ0.9_最少扣留 |   V_MAE |   V_MAE_单种子 |   M2_MAE |   M2排序Spearman |   认证下界比_top1 |   认证下界比_top3 |
|:----------------|:-------------------|-------:|-------:|-------:|-------:|---------------:|------------:|------------:|---------:|--------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------:|------------:|---------:|---------------:|-------------:|-------------:|
| metered_load_mw | 附表 6.61 实际负荷（次日披露） | 0.2013 | 0.2557 | 0.2708 | 0.2788 |         0.0695 |      0.9445 |      0.9711 |   0.4756 |        0.8444 |             0 |          22 |              22 |          237 |              237 |           9 |             0 |          14 |              14 |           44 |               44 |           3 |             0 |           0 |               0 |            0 |                0 |           0 |  0.0598 |      0.0657 |   0.0321 |         0.9266 |       0.8766 |       0.9529 |

## 攻击器族构成

| 攻击器     |   被val选中比例 |
|:--------|-----------:|
| 单目标 DNN |     0.3053 |
| 梯度提升树   |     0.6947 |

## 按集合规模

|   规模 |   正式真值均值 |   通用模型MAE |
|-----:|---------:|----------:|
|    1 |   0.2013 |    0.0248 |
|    2 |   0.3627 |    0.0449 |
|    3 |   0.4856 |    0.0625 |
|    4 |   0.578  |  nan      |

## 通用模型查询耗时

新主口径 E（三种子预测平均）64 ms/集合


## 字段风险表（按目标、M^(2) 降序）

| 目标              | 字段                                     |    r2 |    M0 |    M1 |    M2 |    M3 |   估计M2 |   升级_M2减M0 | K2见证                                                                |
|:----------------|:---------------------------------------|------:|------:|------:|------:|------:|-------:|-----------:|:--------------------------------------------------------------------|
| metered_load_mw | total_lmp_da                           | 0.369 | 0.476 | 0.503 | 0.522 | 0.522 |  0.517 |      0.046 | total_pjm_assigned_reg + da_as_total_mw_thirty_minutes_reserve      |
| metered_load_mw | system_energy_price_da                 | 0.351 | 0.451 | 0.487 | 0.502 | 0.504 |  0.494 |      0.051 | total_pjm_assigned_reg + da_as_total_mw_thirty_minutes_reserve      |
| metered_load_mw | marginal_loss_price_rt                 | 0.178 | 0.382 | 0.402 | 0.409 | 0.409 |  0.398 |      0.027 | total_pjm_self_sched_reg + total_pjm_assigned_reg                   |
| metered_load_mw | da_as_nsr_mw_primary_reserve           | 0.004 | 0.405 | 0.405 | 0.405 | 0.405 |  0.179 |      0     | ∅                                                                   |
| metered_load_mw | marginal_loss_price_da                 | 0.273 | 0.348 | 0.356 | 0.374 | 0.374 |  0.364 |      0.026 | total_pjm_self_sched_reg + gross_sched_interchange_mw               |
| metered_load_mw | da_as_total_mw_primary_reserve         | 0.058 | 0.208 | 0.371 | 0.371 | 0.371 |  0.324 |      0.163 | da_as_total_mw_thirty_minutes_reserve                               |
| metered_load_mw | total_lmp_rt                           | 0.163 | 0.339 | 0.362 | 0.37  | 0.377 |  0.358 |      0.032 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_as_req_mw_primary_reserve        | 0.103 | 0.183 | 0.369 | 0.369 | 0.369 |  0.339 |      0.186 | da_as_total_mw_thirty_minutes_reserve                               |
| metered_load_mw | da_as_total_mw_thirty_minutes_reserve  | 0.01  | 0.122 | 0.308 | 0.342 | 0.352 |  0.322 |      0.22  | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_total_mw_synchronized_reserve    | 0.05  | 0.312 | 0.341 | 0.341 | 0.341 |  0.249 |      0.029 | da_as_total_mw_thirty_minutes_reserve                               |
| metered_load_mw | da_as_as_req_mw_thirty_minutes_reserve | 0.1   | 0.133 | 0.315 | 0.315 | 0.315 |  0.318 |      0.182 | da_as_total_mw_thirty_minutes_reserve                               |
| metered_load_mw | congestion_price_da                    | 0.116 | 0.254 | 0.26  | 0.289 | 0.304 |  0.256 |      0.036 | da_as_as_req_mw_primary_reserve + gross_sched_interchange_mw        |
| metered_load_mw | da_as_mcp_synchronized_reserve         | 0.091 | 0.136 | 0.175 | 0.196 | 0.236 |  0.217 |      0.06  | da_as_as_req_mw_thirty_minutes_reserve + gross_sched_interchange_mw |
| metered_load_mw | congestion_price_rt                    | 0.082 | 0.12  | 0.176 | 0.182 | 0.209 |  0.176 |      0.062 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_mcp_primary_reserve              | 0.074 | 0.132 | 0.14  | 0.168 | 0.168 |  0.19  |      0.036 | total_pjm_self_sched_reg + gross_sched_interchange_mw               |
| metered_load_mw | total_pjm_reg_purchases                | 0.012 | 0.125 | 0.143 | 0.158 | 0.165 |  0.154 |      0.033 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_ss_mw_primary_reserve            | 0.009 | 0.08  | 0.114 | 0.133 | 0.134 |  0.092 |      0.052 | rmpcp + da_as_total_mw_thirty_minutes_reserve                       |
| metered_load_mw | rmccp                                  | 0.011 | 0.06  | 0.107 | 0.126 | 0.149 |  0.105 |      0.066 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | gross_sched_interchange_mw             | 0.041 | 0.057 | 0.088 | 0.11  | 0.11  |  0.087 |      0.053 | total_pjm_self_sched_reg + da_as_mcp_primary_reserve                |
| metered_load_mw | rmpcp                                  | 0.002 | 0.072 | 0.092 | 0.097 | 0.101 |  0.074 |      0.025 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | total_pjm_self_sched_reg               | 0.004 | 0.015 | 0.054 | 0.095 | 0.108 |  0.067 |      0.08  | total_pjm_assigned_reg + da_as_total_mw_thirty_minutes_reserve      |
| metered_load_mw | total_pjm_assigned_reg                 | 0     | 0.019 | 0.058 | 0.082 | 0.11  |  0.068 |      0.063 | total_pjm_self_sched_reg + da_as_total_mw_thirty_minutes_reserve    |