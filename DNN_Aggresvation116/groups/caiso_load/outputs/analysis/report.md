# CAISO 现实口径（补全候选）｜caiso_load：自动生成的数值汇总（结论与解读见 CHANGELOG.md）

## 目标级汇总

| 目标                          | 规则依据               |   M0均值 |   M1均值 |   M2均值 |   M3均值 |   背景升级均值_M2减M0 |   K饱和_M1除M2 |   K饱和_M2除M3 |   最强单字段V |   全部规模≤4集合最大V |   τ0.5_单字段即危险 |   τ0.5_K2关键 |   τ0.5_单看安全组合危险 |   τ0.5_危险小组合 |   τ0.5_单字段定级后仍暴露 |   τ0.5_最少扣留 |   τ0.7_单字段即危险 |   τ0.7_K2关键 |   τ0.7_单看安全组合危险 |   τ0.7_危险小组合 |   τ0.7_单字段定级后仍暴露 |   τ0.7_最少扣留 |   τ0.9_单字段即危险 |   τ0.9_K2关键 |   τ0.9_单看安全组合危险 |   τ0.9_危险小组合 |   τ0.9_单字段定级后仍暴露 |   τ0.9_最少扣留 |   V_MAE |   V_MAE_单种子 |   M2_MAE |   M2排序Spearman |   认证下界比_top1 |   认证下界比_top3 |
|:----------------------------|:-------------------|-------:|-------:|-------:|-------:|---------------:|------------:|------------:|---------:|--------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------:|------------:|---------:|---------------:|-------------:|-------------:|
| actual_load__mw__ca_iso_tac | 附表 6.61 实际负荷（次日披露） |  0.236 | 0.3842 | 0.4092 | 0.4137 |         0.1732 |      0.9388 |      0.9891 |   0.8248 |          0.95 |             5 |          35 |              30 |          542 |              537 |          17 |             3 |          35 |              32 |          335 |              332 |          10 |             0 |          32 |              32 |          287 |              287 |           5 |  0.0288 |      0.0342 |    0.019 |         0.9815 |       0.9768 |        0.992 |

## 攻击器族构成

| 攻击器     |   被val选中比例 |
|:--------|-----------:|
| 单目标 DNN |     0.4761 |
| 梯度提升树   |     0.5239 |

## 按集合规模

|   规模 |   正式真值均值 |   通用模型MAE |
|-----:|---------:|----------:|
|    1 |   0.236  |    0.0125 |
|    2 |   0.4594 |    0.0219 |
|    3 |   0.6232 |    0.0295 |
|    4 |   0.7342 |  nan      |

## 通用模型查询耗时

新主口径 E（三种子预测平均）122 ms/集合


## 字段风险表（按目标、M^(2) 降序）

| 目标                          | 字段                                                                     |    r2 |    M0 |    M1 |    M2 |    M3 |   估计M2 |   升级_M2减M0 | K2见证                                                                                                               |
|:----------------------------|:-----------------------------------------------------------------------|------:|------:|------:|------:|------:|-------:|-----------:|:-------------------------------------------------------------------------------------------------------------------|
| actual_load__mw__ca_iso_tac | dam_load_forecast__mw__sce_tac                                         | 0.817 | 0.825 | 0.852 | 0.852 | 0.852 |  0.846 |      0.027 | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd                                                           |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__load__tac_ecntr                                      | 0.801 | 0.804 | 0.839 | 0.839 | 0.839 |  0.831 |      0.035 | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd                                                           |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__generation__caiso_totals                             | 0.795 | 0.796 | 0.802 | 0.802 | 0.802 |  0.797 |      0.006 | dam_renewable_forecast__mw__sp15__wind                                                                             |
| actual_load__mw__ca_iso_tac | dam_load_forecast__mw__pge_tac                                         | 0.566 | 0.617 | 0.773 | 0.773 | 0.773 |  0.783 |      0.156 | dam_renewable_forecast__mw__sp15__solar                                                                            |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__load__tac_north                                      | 0.62  | 0.645 | 0.755 | 0.755 | 0.755 |  0.769 |      0.11  | dam_renewable_forecast__mw__sp15__solar                                                                            |
| actual_load__mw__ca_iso_tac | dam_load_forecast__mw__sdge_tac                                        | 0.315 | 0.496 | 0.699 | 0.699 | 0.699 |  0.706 |      0.203 | dam_renewable_forecast__mw__sp15__solar                                                                            |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__load__tac_south                                      | 0.202 | 0.316 | 0.625 | 0.637 | 0.637 |  0.632 |      0.321 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso__spinning_reserve__minimum            | 0.463 | 0.42  | 0.558 | 0.566 | 0.566 |  0.553 |      0.147 | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd + rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd |
| actual_load__mw__ca_iso_tac | dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                          | 0.075 | 0.141 | 0.422 | 0.512 | 0.512 |  0.501 |      0.372 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso__regulation_down__minimum             | 0.047 | 0.402 | 0.497 | 0.497 | 0.497 |  0.366 |      0.096 | dam_renewable_forecast__mw__sp15__wind + dam_as_requirement__mw__as_caiso__regulation_up__minimum                  |
| actual_load__mw__ca_iso_tac | dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd                             | 0.045 | 0.135 | 0.415 | 0.491 | 0.491 |  0.485 |      0.356 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__export__caiso_totals                                 | 0.342 | 0.349 | 0.445 | 0.458 | 0.458 |  0.446 |      0.11  | dam_renewable_forecast__mw__sp15__wind + dam_schedule__mw__import__caiso_totals                                    |
| actual_load__mw__ca_iso_tac | dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd                             | 0.029 | 0.111 | 0.364 | 0.439 | 0.439 |  0.435 |      0.328 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__sp15__solar                                | 0.063 | 0.107 | 0.416 | 0.416 | 0.416 |  0.407 |      0.309 | dam_schedule__mw__load__tac_south                                                                                  |
| actual_load__mw__ca_iso_tac | dam_as_total_procured__mw__rd__as_caiso                                | 0.042 | 0.319 | 0.416 | 0.416 | 0.416 |  0.341 |      0.097 | dam_renewable_forecast__mw__sp15__wind + dam_as_requirement__mw__as_caiso__regulation_up__minimum                  |
| actual_load__mw__ca_iso_tac | dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd                             | 0.077 | 0.101 | 0.336 | 0.413 | 0.418 |  0.419 |      0.312 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__zp26__solar                                   |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__zp26__solar                                | 0.069 | 0.12  | 0.395 | 0.4   | 0.4   |  0.4   |      0.28  | dam_renewable_forecast__mw__np15__wind + dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                             |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__np15__solar                                | 0.072 | 0.128 | 0.393 | 0.393 | 0.393 |  0.394 |      0.266 | dam_renewable_forecast__mw__np15__wind + dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                             |
| actual_load__mw__ca_iso_tac | dam_as_total_procured__mw__nr__as_caiso                                | 0.182 | 0.141 | 0.286 | 0.392 | 0.392 |  0.396 |      0.252 | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd + dam_as_total_procured__mw__sr__as_caiso                  |
| actual_load__mw__ca_iso_tac | dam_lmp__loss_usd_per_mwh__th_sp15_gen_apnd                            | 0.112 | 0.208 | 0.307 | 0.37  | 0.37  |  0.307 |      0.162 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_total_procured__mw__sr__as_caiso                                | 0.112 | 0.151 | 0.296 | 0.339 | 0.367 |  0.328 |      0.188 | dam_schedule__mw__import__caiso_totals + dam_as_total_procured__mw__nr__as_caiso                                   |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd                | 0.04  | 0.081 | 0.291 | 0.325 | 0.325 |  0.319 |      0.244 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso_exp__regulation_mileage_down__minimum | 0.09  | 0.167 | 0.284 | 0.297 | 0.297 |  0.288 |      0.13  | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd + dam_schedule__mw__import__caiso_totals                  |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_zp26_gen_apnd                | 0.027 | 0.067 | 0.272 | 0.288 | 0.29  |  0.286 |      0.221 | dam_renewable_forecast__mw__sp15__wind + dam_renewable_forecast__mw__zp26__solar                                   |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd                | 0.073 | 0.081 | 0.226 | 0.26  | 0.287 |  0.257 |      0.179 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__zp26__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso__regulation_up__minimum               | 0.021 | 0.154 | 0.251 | 0.251 | 0.251 |  0.177 |      0.097 | dam_renewable_forecast__mw__sp15__wind + dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                             |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__import__caiso_totals                                 | 0.018 | 0.045 | 0.231 | 0.234 | 0.241 |  0.227 |      0.189 | dam_renewable_forecast__mw__np15__solar + dam_renewable_forecast__mw__sp15__wind                                   |
| actual_load__mw__ca_iso_tac | dam_lmp__congestion_usd_per_mwh__th_sp15_gen_apnd                      | 0.004 | 0.077 | 0.163 | 0.195 | 0.197 |  0.184 |      0.118 | dam_renewable_forecast__mw__np15__wind + dam_as_requirement__mw__as_caiso__spinning_reserve__minimum               |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__np15__wind                                 | 0.052 | 0.045 | 0.129 | 0.166 | 0.178 |  0.167 |      0.12  | dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd + dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd                         |
| actual_load__mw__ca_iso_tac | dam_as_total_procured__mw__ru__as_caiso                                | 0.01  | 0.089 | 0.161 | 0.161 | 0.17  |  0.112 |      0.072 | dam_as_total_procured__mw__rd__as_caiso                                                                            |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd               | 0.006 | 0.009 | 0.144 | 0.15  | 0.18  |  0.148 |      0.141 | dam_renewable_forecast__mw__sp15__solar + dam_as_requirement__mw__as_caiso_exp__regulation_mileage_up__minimum     |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd         | 0.002 | 0.069 | 0.131 | 0.15  | 0.15  |  0.119 |      0.081 | dam_renewable_forecast__mw__np15__wind + rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd                   |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__sp15__wind                                 | 0.001 | 0     | 0.107 | 0.145 | 0.178 |  0.127 |      0.145 | dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd + rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd              |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso_exp__regulation_mileage_up__minimum   | 0.021 | 0.038 | 0.107 | 0.129 | 0.129 |  0.11  |      0.091 | dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd + dam_schedule__mw__import__caiso_totals                                |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__zp26__wind                                 | 0.022 | 0.008 | 0.057 | 0.11  | 0.11  |  0.081 |      0.101 | dam_as_requirement__mw__as_caiso__spinning_reserve__minimum + dam_as_total_procured__mw__ru__as_caiso              |