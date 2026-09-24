# PJM 现实口径｜pjm_load：自动生成的数值汇总（结论与解读见 CHANGELOG.md）

## 目标级汇总

| 目标              | 规则依据               |   M0均值 |   M1均值 |   M2均值 |   M3均值 |   背景升级均值_M2减M0 |   K饱和_M1除M2 |   K饱和_M2除M3 |   最强单字段V |   全部规模≤4集合最大V |   τ0.5_单字段即危险 |   τ0.5_K2关键 |   τ0.5_单看安全组合危险 |   τ0.5_危险小组合 |   τ0.5_单字段定级后仍暴露 |   τ0.5_最少扣留 |   τ0.7_单字段即危险 |   τ0.7_K2关键 |   τ0.7_单看安全组合危险 |   τ0.7_危险小组合 |   τ0.7_单字段定级后仍暴露 |   τ0.7_最少扣留 |   τ0.9_单字段即危险 |   τ0.9_K2关键 |   τ0.9_单看安全组合危险 |   τ0.9_危险小组合 |   τ0.9_单字段定级后仍暴露 |   τ0.9_最少扣留 |   V_MAE |   V_MAE_单种子 |   M2_MAE |   M2排序Spearman |   认证下界比_top1 |   认证下界比_top3 |
|:----------------|:-------------------|-------:|-------:|-------:|-------:|---------------:|------------:|------------:|---------:|--------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------:|------------:|---------:|---------------:|-------------:|-------------:|
| metered_load_mw | 附表 6.61 实际负荷（次日披露） |  0.173 | 0.1991 | 0.2142 | 0.2269 |         0.0411 |      0.9297 |       0.944 |   0.4506 |        0.8287 |             0 |          15 |              15 |           63 |               63 |           4 |             0 |           7 |               7 |            5 |                5 |           2 |             0 |           0 |               0 |            0 |                0 |           0 |  0.0678 |      0.0716 |   0.0268 |         0.9508 |        0.861 |       0.9504 |

## 攻击器族构成

| 攻击器     |   被val选中比例 |
|:--------|-----------:|
| 单目标 DNN |     0.2974 |
| 梯度提升树   |     0.7026 |

## 按集合规模

|   规模 |   正式真值均值 |   通用模型MAE |
|-----:|---------:|----------:|
|    1 |   0.173  |    0.0208 |
|    2 |   0.3202 |    0.0401 |
|    3 |   0.4408 |    0.0598 |
|    4 |   0.5364 |    0.0731 |

## 通用模型查询耗时

三种子 22.56 ms/集合；单种子 2.80 ms/集合；预训练 [34.0, 29.0, 34.0] 秒/种子


## 字段风险表（按目标、M^(2) 降序）

| 目标              | 字段                                     |    r2 |    M0 |    M1 |    M2 |    M3 |   估计M2 |   升级_M2减M0 | K2见证                                                                |
|:----------------|:---------------------------------------|------:|------:|------:|------:|------:|-------:|-----------:|:--------------------------------------------------------------------|
| metered_load_mw | system_energy_price_da                 | 0.351 | 0.451 | 0.484 | 0.501 | 0.504 |  0.488 |      0.05  | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_nsr_mw_primary_reserve           | 0.004 | 0.405 | 0.405 | 0.405 | 0.405 |  0.244 |      0     | ∅                                                                   |
| metered_load_mw | marginal_loss_price_rt                 | 0.178 | 0.391 | 0.403 | 0.404 | 0.404 |  0.419 |      0.013 | total_pjm_self_sched_reg + total_pjm_assigned_reg                   |
| metered_load_mw | total_lmp_rt                           | 0.163 | 0.331 | 0.359 | 0.37  | 0.377 |  0.357 |      0.039 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_as_req_mw_primary_reserve        | 0.103 | 0.183 | 0.225 | 0.239 | 0.255 |  0.217 |      0.056 | da_as_mcp_synchronized_reserve + da_as_ss_mw_primary_reserve        |
| metered_load_mw | da_as_mcp_synchronized_reserve         | 0.091 | 0.136 | 0.175 | 0.193 | 0.236 |  0.202 |      0.057 | da_as_as_req_mw_thirty_minutes_reserve + gross_sched_interchange_mw |
| metered_load_mw | da_as_as_req_mw_thirty_minutes_reserve | 0.1   | 0.133 | 0.173 | 0.191 | 0.206 |  0.226 |      0.058 | da_as_mcp_synchronized_reserve + da_as_ss_mw_primary_reserve        |
| metered_load_mw | da_as_mcp_primary_reserve              | 0.074 | 0.133 | 0.14  | 0.161 | 0.175 |  0.173 |      0.028 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | total_pjm_reg_purchases                | 0.012 | 0.125 | 0.137 | 0.156 | 0.165 |  0.181 |      0.031 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | da_as_ss_mw_primary_reserve            | 0.009 | 0.08  | 0.114 | 0.131 | 0.131 |  0.105 |      0.051 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | rmccp                                  | 0.011 | 0.065 | 0.107 | 0.126 | 0.149 |  0.137 |      0.062 | total_pjm_self_sched_reg + da_as_as_req_mw_thirty_minutes_reserve   |
| metered_load_mw | rmpcp                                  | 0.002 | 0.072 | 0.087 | 0.094 | 0.101 |  0.088 |      0.022 | total_pjm_self_sched_reg + gross_sched_interchange_mw               |
| metered_load_mw | gross_sched_interchange_mw             | 0.041 | 0.057 | 0.068 | 0.088 | 0.097 |  0.105 |      0.031 | total_pjm_self_sched_reg + da_as_mcp_primary_reserve                |
| metered_load_mw | total_pjm_assigned_reg                 | 0     | 0.019 | 0.057 | 0.079 | 0.107 |  0.105 |      0.06  | total_pjm_self_sched_reg + da_as_mcp_primary_reserve                |
| metered_load_mw | total_pjm_self_sched_reg               | 0.004 | 0.015 | 0.053 | 0.073 | 0.091 |  0.085 |      0.059 | total_pjm_assigned_reg + da_as_mcp_synchronized_reserve             |