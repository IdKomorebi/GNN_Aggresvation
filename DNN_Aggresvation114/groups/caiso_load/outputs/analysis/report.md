# CAISO 现实口径｜caiso_load：自动生成的数值汇总（结论与解读见 CHANGELOG.md）

## 目标级汇总

| 目标                          | 规则依据               |   M0均值 |   M1均值 |   M2均值 |   M3均值 |   背景升级均值_M2减M0 |   K饱和_M1除M2 |   K饱和_M2除M3 |   最强单字段V |   全部规模≤4集合最大V |   τ0.5_单字段即危险 |   τ0.5_K2关键 |   τ0.5_单看安全组合危险 |   τ0.5_危险小组合 |   τ0.5_单字段定级后仍暴露 |   τ0.5_最少扣留 |   τ0.7_单字段即危险 |   τ0.7_K2关键 |   τ0.7_单看安全组合危险 |   τ0.7_危险小组合 |   τ0.7_单字段定级后仍暴露 |   τ0.7_最少扣留 |   τ0.9_单字段即危险 |   τ0.9_K2关键 |   τ0.9_单看安全组合危险 |   τ0.9_危险小组合 |   τ0.9_单字段定级后仍暴露 |   τ0.9_最少扣留 |   V_MAE |   V_MAE_单种子 |   M2_MAE |   M2排序Spearman |   认证下界比_top1 |   认证下界比_top3 |
|:----------------------------|:-------------------|-------:|-------:|-------:|-------:|---------------:|------------:|------------:|---------:|--------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------------:|------------:|----------------:|-------------:|-----------------:|------------:|--------:|------------:|---------:|---------------:|-------------:|-------------:|
| actual_load__mw__ca_iso_tac | 附表 6.61 实际负荷（次日披露） |  0.246 | 0.4076 |  0.429 | 0.4302 |         0.1831 |      0.9501 |      0.9973 |   0.8234 |        0.9503 |             4 |          24 |              20 |          152 |              148 |          12 |             2 |          24 |              22 |          137 |              135 |           8 |             0 |          24 |              24 |          175 |              175 |           5 |  0.0273 |      0.0321 |   0.0155 |         0.9852 |       0.9914 |       0.9957 |

## 攻击器族构成

| 攻击器     |   被val选中比例 |
|:--------|-----------:|
| 单目标 DNN |     0.5452 |
| 梯度提升树   |     0.4548 |

## 按集合规模

|   规模 |   正式真值均值 |   通用模型MAE |
|-----:|---------:|----------:|
|    1 |   0.246  |    0.009  |
|    2 |   0.488  |    0.0182 |
|    3 |   0.6618 |    0.0251 |
|    4 |   0.7737 |    0.028  |

## 通用模型查询耗时

三种子 43.63 ms/集合；单种子 6.32 ms/集合；预训练 [25.0, 31.0, 34.0] 秒/种子


## 字段风险表（按目标、M^(2) 降序）

| 目标                          | 字段                                                                     |    r2 |    M0 |    M1 |    M2 |    M3 |   估计M2 |   升级_M2减M0 | K2见证                                                                                                               |
|:----------------------------|:-----------------------------------------------------------------------|------:|------:|------:|------:|------:|-------:|-----------:|:-------------------------------------------------------------------------------------------------------------------|
| actual_load__mw__ca_iso_tac | dam_load_forecast__mw__sce_tac                                         | 0.817 | 0.823 | 0.854 | 0.854 | 0.854 |  0.849 |      0.031 | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd                                                           |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__load__tac_ecntr                                      | 0.801 | 0.804 | 0.839 | 0.839 | 0.839 |  0.834 |      0.035 | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd                                                           |
| actual_load__mw__ca_iso_tac | dam_load_forecast__mw__pge_tac                                         | 0.566 | 0.62  | 0.773 | 0.773 | 0.773 |  0.783 |      0.153 | dam_renewable_forecast__mw__sp15__solar                                                                            |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__load__tac_north                                      | 0.62  | 0.645 | 0.755 | 0.755 | 0.755 |  0.767 |      0.11  | dam_renewable_forecast__mw__sp15__solar                                                                            |
| actual_load__mw__ca_iso_tac | dam_load_forecast__mw__sdge_tac                                        | 0.315 | 0.492 | 0.699 | 0.699 | 0.699 |  0.704 |      0.208 | dam_renewable_forecast__mw__sp15__solar                                                                            |
| actual_load__mw__ca_iso_tac | dam_schedule__mw__load__tac_south                                      | 0.202 | 0.315 | 0.625 | 0.637 | 0.637 |  0.634 |      0.322 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso__spinning_reserve__minimum            | 0.463 | 0.421 | 0.56  | 0.56  | 0.56  |  0.555 |      0.139 | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd                                                            |
| actual_load__mw__ca_iso_tac | dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                          | 0.075 | 0.14  | 0.426 | 0.512 | 0.512 |  0.506 |      0.371 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso__regulation_down__minimum             | 0.047 | 0.402 | 0.497 | 0.497 | 0.497 |  0.357 |      0.096 | dam_renewable_forecast__mw__sp15__wind + dam_as_requirement__mw__as_caiso__regulation_up__minimum                  |
| actual_load__mw__ca_iso_tac | dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd                             | 0.029 | 0.111 | 0.364 | 0.44  | 0.44  |  0.44  |      0.329 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__sp15__solar                                | 0.063 | 0.107 | 0.417 | 0.417 | 0.417 |  0.411 |      0.31  | dam_schedule__mw__load__tac_south                                                                                  |
| actual_load__mw__ca_iso_tac | dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd                             | 0.077 | 0.101 | 0.338 | 0.409 | 0.409 |  0.414 |      0.308 | dam_renewable_forecast__mw__np15__solar + dam_renewable_forecast__mw__np15__wind                                   |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__zp26__solar                                | 0.069 | 0.12  | 0.395 | 0.404 | 0.404 |  0.403 |      0.284 | dam_renewable_forecast__mw__np15__wind + dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                             |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__np15__solar                                | 0.072 | 0.128 | 0.393 | 0.4   | 0.4   |  0.397 |      0.272 | dam_renewable_forecast__mw__np15__wind + dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd                                |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd                | 0.04  | 0.082 | 0.287 | 0.325 | 0.325 |  0.323 |      0.244 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso_exp__regulation_mileage_down__minimum | 0.09  | 0.167 | 0.273 | 0.294 | 0.294 |  0.287 |      0.127 | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_sp15_gen_apnd + rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_zp26_gen_apnd                | 0.027 | 0.074 | 0.272 | 0.288 | 0.29  |  0.29  |      0.214 | dam_renewable_forecast__mw__sp15__wind + dam_renewable_forecast__mw__zp26__solar                                   |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd                | 0.073 | 0.079 | 0.225 | 0.26  | 0.26  |  0.256 |      0.181 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso__regulation_up__minimum               | 0.021 | 0.154 | 0.25  | 0.257 | 0.257 |  0.169 |      0.103 | dam_renewable_forecast__mw__sp15__wind + dam_lmp__energy_usd_per_mwh__th_sp15_gen_apnd                             |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__np15__wind                                 | 0.052 | 0.047 | 0.129 | 0.167 | 0.167 |  0.158 |      0.119 | dam_renewable_forecast__mw__sp15__solar + dam_lmp__lmp_usd_per_mwh__th_np15_gen_apnd                               |
| actual_load__mw__ca_iso_tac | rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd               | 0.006 | 0.008 | 0.144 | 0.153 | 0.153 |  0.14  |      0.145 | dam_renewable_forecast__mw__np15__wind + dam_renewable_forecast__mw__sp15__solar                                   |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__sp15__wind                                 | 0.001 | 0     | 0.1   | 0.147 | 0.173 |  0.136 |      0.147 | dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd + rt15_lmp_hourly_mean__loss_usd_per_mwh__th_np15_gen_apnd              |
| actual_load__mw__ca_iso_tac | dam_as_requirement__mw__as_caiso_exp__regulation_mileage_up__minimum   | 0.021 | 0.049 | 0.109 | 0.117 | 0.117 |  0.101 |      0.068 | dam_renewable_forecast__mw__sp15__wind + dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd                                |
| actual_load__mw__ca_iso_tac | dam_renewable_forecast__mw__zp26__wind                                 | 0.022 | 0.013 | 0.057 | 0.091 | 0.091 |  0.078 |      0.079 | dam_renewable_forecast__mw__np15__wind + dam_as_requirement__mw__as_caiso__spinning_reserve__minimum               |