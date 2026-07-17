# 61 号：推断链敏感度泄露报告
按主敏感度 s_max（最强推断链强度）降序。
## forecast_load_mw_latest_available
- s_max=0.991, s_or=1.000, s_walk=11.886, 可达机密字段 12/12
- 真实攻击: 直接=0.991, 两跳=0.990
主要泄露链:
- [0.991] forecast_load_mw_latest_available --0.99--> metered_load_mw
- [0.973] forecast_load_mw_latest_available --0.97--> total_gen
- [0.574] forecast_load_mw_latest_available --0.57--> total_lmp_da
- [0.416] forecast_load_mw_latest_available --0.42--> marginal_loss_price_da
- [0.389] forecast_load_mw_latest_available --0.39--> total_losses

## forecast_load_mw_day_ahead
- s_max=0.982, s_or=1.000, s_walk=11.916, 可达机密字段 12/12
- 真实攻击: 直接=0.982, 两跳=0.982
主要泄露链:
- [0.982] forecast_load_mw_day_ahead --0.98--> metered_load_mw
- [0.966] forecast_load_mw_day_ahead --0.97--> total_gen
- [0.590] forecast_load_mw_day_ahead --0.59--> total_lmp_da
- [0.414] forecast_load_mw_day_ahead --0.41--> marginal_loss_price_da
- [0.385] forecast_load_mw_day_ahead --0.38--> total_losses

## system_energy_price_da
- s_max=0.947, s_or=0.998, s_walk=8.428, 可达机密字段 12/12
- 真实攻击: 直接=0.947, 两跳=0.807
主要泄露链:
- [0.947] system_energy_price_da --0.95--> total_lmp_da
- [0.669] system_energy_price_da --0.67--> marginal_loss_price_da
- [0.440] system_energy_price_da --0.44--> metered_load_mw
- [0.373] system_energy_price_da --0.37--> total_gen
- [0.322] system_energy_price_da --0.32--> da_as_total_mw_thirty_minutes_reserve

## gen_fuel_gas_mw
- s_max=0.875, s_or=0.998, s_walk=11.112, 可达机密字段 12/12
- 真实攻击: 直接=0.875, 两跳=0.871
主要泄露链:
- [0.875] gen_fuel_gas_mw --0.87--> total_gen
- [0.847] gen_fuel_gas_mw --0.85--> metered_load_mw
- [0.468] gen_fuel_gas_mw --0.47--> total_lmp_da
- [0.291] gen_fuel_gas_mw --0.29--> da_as_total_mw_thirty_minutes_reserve
- [0.279] gen_fuel_gas_mw --0.84--> forecast_load_mw_latest_available --0.42--> marginal_loss_price_da

## gen_fuel_nuclear_pct
- s_max=0.866, s_or=0.998, s_walk=11.136, 可达机密字段 12/12
- 真实攻击: 直接=0.866, 两跳=0.865
主要泄露链:
- [0.866] gen_fuel_nuclear_pct --0.87--> metered_load_mw
- [0.861] gen_fuel_nuclear_pct --0.86--> total_gen
- [0.536] gen_fuel_nuclear_pct --0.54--> total_lmp_da
- [0.339] gen_fuel_nuclear_pct --0.34--> marginal_loss_price_da
- [0.267] gen_fuel_nuclear_pct --0.86--> forecast_load_mw_latest_available --0.39--> total_losses

## gen_fuel_coal_mw
- s_max=0.766, s_or=0.995, s_walk=9.760, 可达机密字段 12/12
- 真实攻击: 直接=0.766, 两跳=0.761
主要泄露链:
- [0.766] gen_fuel_coal_mw --0.77--> total_gen
- [0.750] gen_fuel_coal_mw --0.75--> metered_load_mw
- [0.487] gen_fuel_coal_mw --0.49--> total_losses
- [0.422] gen_fuel_coal_mw --0.42--> total_lmp_da
- [0.355] gen_fuel_coal_mw --0.36--> marginal_loss_price_da

## gen_fuel_oil_mw
- s_max=0.686, s_or=0.987, s_walk=9.628, 可达机密字段 12/12
- 真实攻击: 直接=0.686, 两跳=0.683
主要泄露链:
- [0.686] gen_fuel_oil_mw --0.69--> total_lmp_da
- [0.539] gen_fuel_oil_mw --0.54--> metered_load_mw
- [0.514] gen_fuel_oil_mw --0.51--> marginal_loss_price_da
- [0.509] gen_fuel_oil_mw --0.51--> total_gen
- [0.216] gen_fuel_oil_mw --0.22--> da_as_total_mw_thirty_minutes_reserve

## da_as_mcp_synchronized_reserve
- s_max=0.672, s_or=0.958, s_walk=5.589, 可达机密字段 12/12
- 真实攻击: 直接=0.672, 两跳=0.667
主要泄露链:
- [0.672] da_as_mcp_synchronized_reserve --0.67--> total_lmp_da
- [0.437] da_as_mcp_synchronized_reserve --0.44--> marginal_loss_price_da
- [0.360] da_as_mcp_synchronized_reserve --0.36--> da_as_total_mw_thirty_minutes_reserve
- [0.241] da_as_mcp_synchronized_reserve --0.68--> system_energy_price_da --0.44--> metered_load_mw
- [0.205] da_as_mcp_synchronized_reserve --0.68--> system_energy_price_da --0.37--> total_gen

## da_as_nsr_mw_primary_reserve
- s_max=0.667, s_or=0.994, s_walk=8.513, 可达机密字段 12/12
- 真实攻击: 直接=0.667, 两跳=0.425
主要泄露链:
- [0.667] da_as_nsr_mw_primary_reserve --0.67--> da_as_total_mw_primary_reserve
- [0.467] da_as_nsr_mw_primary_reserve --0.47--> da_as_total_mw_thirty_minutes_reserve
- [0.429] da_as_nsr_mw_primary_reserve --0.43--> total_gen
- [0.426] da_as_nsr_mw_primary_reserve --0.43--> da_as_total_mw_synchronized_reserve
- [0.423] da_as_nsr_mw_primary_reserve --0.42--> metered_load_mw

## gen_fuel_oil_pct
- s_max=0.620, s_or=0.973, s_walk=8.216, 可达机密字段 12/12
- 真实攻击: 直接=0.620, 两跳=0.625
主要泄露链:
- [0.620] gen_fuel_oil_pct --0.62--> total_lmp_da
- [0.483] gen_fuel_oil_pct --0.48--> marginal_loss_price_da
- [0.421] gen_fuel_oil_pct --0.42--> metered_load_mw
- [0.396] gen_fuel_oil_pct --0.97--> gen_fuel_oil_mw --0.51--> total_gen
- [0.246] gen_fuel_oil_pct --0.25--> da_as_total_mw_thirty_minutes_reserve

## gross_inadv_interchange_mw
- s_max=0.618, s_or=0.855, s_walk=2.349, 可达机密字段 12/12
- 真实攻击: 直接=0.618, 两跳=0.255
主要泄露链:
- [0.618] gross_inadv_interchange_mw --0.62--> gross_actual_interchange_mw
- [0.182] gross_inadv_interchange_mw --0.18--> total_gen
- [0.176] gross_inadv_interchange_mw --0.18--> total_losses
- [0.160] gross_inadv_interchange_mw --0.16--> metered_load_mw
- [0.131] gross_inadv_interchange_mw --0.13--> total_lmp_da

## da_as_mcp_primary_reserve
- s_max=0.568, s_or=0.934, s_walk=5.208, 可达机密字段 12/12
- 真实攻击: 直接=0.568, 两跳=0.563
主要泄露链:
- [0.568] da_as_mcp_primary_reserve --0.57--> total_lmp_da
- [0.353] da_as_mcp_primary_reserve --0.35--> marginal_loss_price_da
- [0.324] da_as_mcp_primary_reserve --0.32--> da_as_total_mw_thirty_minutes_reserve
- [0.203] da_as_mcp_primary_reserve --0.58--> system_energy_price_da --0.44--> metered_load_mw
- [0.172] da_as_mcp_primary_reserve --0.58--> system_energy_price_da --0.37--> total_gen

## gen_fuel_nuclear_mw
- s_max=0.562, s_or=0.991, s_walk=8.423, 可达机密字段 12/12
- 真实攻击: 直接=0.562, 两跳=0.553
主要泄露链:
- [0.562] gen_fuel_nuclear_mw --0.56--> total_gen
- [0.526] gen_fuel_nuclear_mw --0.53--> total_losses
- [0.513] gen_fuel_nuclear_mw --0.51--> metered_load_mw
- [0.376] gen_fuel_nuclear_mw --0.38--> da_as_total_mw_thirty_minutes_reserve
- [0.332] gen_fuel_nuclear_mw --0.33--> net_actual_interchange_mw

## da_as_as_req_mw_primary_reserve
- s_max=0.534, s_or=0.968, s_walk=4.820, 可达机密字段 12/12
- 真实攻击: 直接=0.534, 两跳=0.528
主要泄露链:
- [0.534] da_as_as_req_mw_primary_reserve --0.53--> da_as_total_mw_synchronized_reserve
- [0.482] da_as_as_req_mw_primary_reserve --0.48--> da_as_total_mw_primary_reserve
- [0.437] da_as_as_req_mw_primary_reserve --0.44--> net_actual_interchange_mw
- [0.277] da_as_as_req_mw_primary_reserve --0.28--> total_losses
- [0.241] da_as_as_req_mw_primary_reserve --0.24--> total_gen

## da_as_as_req_mw_synchronized_reserve
- s_max=0.534, s_or=0.968, s_walk=4.820, 可达机密字段 12/12
- 真实攻击: 直接=0.534, 两跳=0.528
主要泄露链:
- [0.534] da_as_as_req_mw_synchronized_reserve --0.53--> da_as_total_mw_synchronized_reserve
- [0.482] da_as_as_req_mw_synchronized_reserve --0.48--> da_as_total_mw_primary_reserve
- [0.437] da_as_as_req_mw_synchronized_reserve --0.44--> net_actual_interchange_mw
- [0.277] da_as_as_req_mw_synchronized_reserve --0.28--> total_losses
- [0.241] da_as_as_req_mw_synchronized_reserve --0.24--> total_gen

## gen_fuel_coal_pct
- s_max=0.531, s_or=0.969, s_walk=5.919, 可达机密字段 12/12
- 真实攻击: 直接=0.468, 两跳=0.449
主要泄露链:
- [0.531] gen_fuel_coal_pct --0.87--> gen_fuel_coal_mw --0.77--> total_gen
- [0.519] gen_fuel_coal_pct --0.87--> gen_fuel_coal_mw --0.75--> metered_load_mw
- [0.468] gen_fuel_coal_pct --0.47--> total_losses
- [0.293] gen_fuel_coal_pct --0.87--> gen_fuel_coal_mw --0.42--> total_lmp_da
- [0.246] gen_fuel_coal_pct --0.87--> gen_fuel_coal_mw --0.36--> marginal_loss_price_da

## da_as_as_req_mw_thirty_minutes_reserve
- s_max=0.522, s_or=0.948, s_walk=3.669, 可达机密字段 12/12
- 真实攻击: 直接=0.522, 两跳=0.500
主要泄露链:
- [0.522] da_as_as_req_mw_thirty_minutes_reserve --0.52--> da_as_total_mw_synchronized_reserve
- [0.463] da_as_as_req_mw_thirty_minutes_reserve --0.46--> da_as_total_mw_primary_reserve
- [0.343] da_as_as_req_mw_thirty_minutes_reserve --0.98--> da_as_as_req_mw_synchronized_reserve --0.44--> net_actual_interchange_mw
- [0.257] da_as_as_req_mw_thirty_minutes_reserve --0.26--> total_losses
- [0.189] da_as_as_req_mw_thirty_minutes_reserve --0.98--> da_as_as_req_mw_synchronized_reserve --0.24--> total_gen

## marginal_loss_price_rt
- s_max=0.518, s_or=0.961, s_walk=6.880, 可达机密字段 12/12
- 真实攻击: 直接=0.518, 两跳=0.515
主要泄露链:
- [0.518] marginal_loss_price_rt --0.52--> total_lmp_da
- [0.409] marginal_loss_price_rt --0.41--> marginal_loss_price_da
- [0.400] marginal_loss_price_rt --0.40--> congestion_price_rt
- [0.367] marginal_loss_price_rt --0.37--> metered_load_mw
- [0.304] marginal_loss_price_rt --0.30--> total_gen

## gen_fuel_other_renewables_pct
- s_max=0.507, s_or=0.919, s_walk=6.516, 可达机密字段 12/12
- 真实攻击: 直接=0.507, 两跳=0.507
主要泄露链:
- [0.507] gen_fuel_other_renewables_pct --0.51--> total_gen
- [0.505] gen_fuel_other_renewables_pct --0.50--> metered_load_mw
- [0.229] gen_fuel_other_renewables_pct --0.48--> forecast_load_mw_day_ahead --0.59--> total_lmp_da
- [0.174] gen_fuel_other_renewables_pct --0.17--> marginal_loss_price_da
- [0.154] gen_fuel_other_renewables_pct --0.49--> forecast_load_mw_latest_available --0.39--> total_losses

## total_lmp_rt
- s_max=0.501, s_or=0.948, s_walk=6.559, 可达机密字段 12/12
- 真实攻击: 直接=0.501, 两跳=0.501
主要泄露链:
- [0.501] total_lmp_rt --0.50--> total_lmp_da
- [0.450] total_lmp_rt --0.45--> congestion_price_rt
- [0.360] total_lmp_rt --0.36--> marginal_loss_price_da
- [0.325] total_lmp_rt --0.32--> metered_load_mw
- [0.270] total_lmp_rt --0.27--> total_gen

## gen_fuel_gas_pct
- s_max=0.399, s_or=0.855, s_walk=3.688, 可达机密字段 12/12
- 真实攻击: 直接=0.235, 两跳=0.231
主要泄露链:
- [0.399] gen_fuel_gas_pct --0.57--> gen_fuel_gas_mw --0.87--> total_gen
- [0.387] gen_fuel_gas_pct --0.57--> gen_fuel_gas_mw --0.85--> metered_load_mw
- [0.214] gen_fuel_gas_pct --0.57--> gen_fuel_gas_mw --0.47--> total_lmp_da
- [0.133] gen_fuel_gas_pct --0.57--> gen_fuel_gas_mw --0.29--> da_as_total_mw_thirty_minutes_reserve
- [0.127] gen_fuel_gas_pct --0.57--> gen_fuel_gas_mw --0.84--> forecast_load_mw_latest_available --0.42--> marginal_loss_price_da

## gen_fuel_hydro_mw
- s_max=0.304, s_or=0.853, s_walk=5.614, 可达机密字段 12/12
- 真实攻击: 直接=0.304, 两跳=0.305
主要泄露链:
- [0.304] gen_fuel_hydro_mw --0.30--> metered_load_mw
- [0.292] gen_fuel_hydro_mw --0.29--> total_gen
- [0.280] gen_fuel_hydro_mw --0.28--> total_lmp_da
- [0.182] gen_fuel_hydro_mw --0.18--> marginal_loss_price_da
- [0.134] gen_fuel_hydro_mw --0.13--> da_as_total_mw_thirty_minutes_reserve

## gen_fuel_multiple_fuels_mw
- s_max=0.280, s_or=0.829, s_walk=5.520, 可达机密字段 12/12
- 真实攻击: 直接=0.280, 两跳=0.279
主要泄露链:
- [0.280] gen_fuel_multiple_fuels_mw --0.28--> metered_load_mw
- [0.239] gen_fuel_multiple_fuels_mw --0.24--> da_as_total_mw_thirty_minutes_reserve
- [0.234] gen_fuel_multiple_fuels_mw --0.23--> total_gen
- [0.205] gen_fuel_multiple_fuels_mw --0.20--> total_lmp_da
- [0.156] gen_fuel_multiple_fuels_mw --0.16--> marginal_loss_price_da

## gen_fuel_hydro_pct
- s_max=0.225, s_or=0.763, s_walk=3.403, 可达机密字段 12/12
- 真实攻击: 直接=0.156, 两跳=0.153
主要泄露链:
- [0.225] gen_fuel_hydro_pct --0.92--> gen_fuel_hydro_mw --0.30--> metered_load_mw
- [0.216] gen_fuel_hydro_pct --0.92--> gen_fuel_hydro_mw --0.29--> total_gen
- [0.207] gen_fuel_hydro_pct --0.92--> gen_fuel_hydro_mw --0.28--> total_lmp_da
- [0.135] gen_fuel_hydro_pct --0.92--> gen_fuel_hydro_mw --0.18--> marginal_loss_price_da
- [0.102] gen_fuel_hydro_pct --0.10--> da_as_total_mw_thirty_minutes_reserve

## gross_sched_interchange_mw
- s_max=0.217, s_or=0.527, s_walk=0.394, 可达机密字段 12/12
- 真实攻击: 直接=0.217, 两跳=0.191
主要泄露链:
- [0.217] gross_sched_interchange_mw --0.22--> gross_actual_interchange_mw
- [0.206] gross_sched_interchange_mw --0.21--> net_actual_interchange_mw
- [0.049] gross_sched_interchange_mw --0.08--> gen_fuel_coal_mw --0.77--> total_gen
- [0.048] gross_sched_interchange_mw --0.08--> gen_fuel_coal_mw --0.75--> metered_load_mw
- [0.041] gross_sched_interchange_mw --0.10--> gen_fuel_nuclear_mw --0.53--> total_losses

## gen_fuel_wind_pct
- s_max=0.215, s_or=0.632, s_walk=2.314, 可达机密字段 12/12
- 真实攻击: 直接=0.169, 两跳=0.164
主要泄露链:
- [0.215] gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.87--> total_gen
- [0.208] gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.85--> metered_load_mw
- [0.115] gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.47--> total_lmp_da
- [0.072] gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.29--> da_as_total_mw_thirty_minutes_reserve
- [0.069] gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.84--> forecast_load_mw_latest_available --0.42--> marginal_loss_price_da

## gen_fuel_multiple_fuels_pct
- s_max=0.212, s_or=0.763, s_walk=3.178, 可达机密字段 12/12
- 真实攻击: 直接=0.212, 两跳=0.169
主要泄露链:
- [0.212] gen_fuel_multiple_fuels_pct --0.21--> da_as_total_mw_thirty_minutes_reserve
- [0.203] gen_fuel_multiple_fuels_pct --0.90--> gen_fuel_multiple_fuels_mw --0.28--> metered_load_mw
- [0.169] gen_fuel_multiple_fuels_pct --0.90--> gen_fuel_multiple_fuels_mw --0.23--> total_gen
- [0.163] gen_fuel_multiple_fuels_pct --0.16--> net_actual_interchange_mw
- [0.148] gen_fuel_multiple_fuels_pct --0.90--> gen_fuel_multiple_fuels_mw --0.20--> total_lmp_da

## total_pjm_rmccp_cr
- s_max=0.167, s_or=0.556, s_walk=1.851, 可达机密字段 12/12
- 真实攻击: 直接=0.167, 两跳=0.175
主要泄露链:
- [0.167] total_pjm_rmccp_cr --0.17--> total_lmp_da
- [0.153] total_pjm_rmccp_cr --0.15--> congestion_price_rt
- [0.100] total_pjm_rmccp_cr --0.31--> marginal_loss_price_rt --0.41--> marginal_loss_price_da
- [0.090] total_pjm_rmccp_cr --0.31--> marginal_loss_price_rt --0.37--> metered_load_mw
- [0.074] total_pjm_rmccp_cr --0.31--> marginal_loss_price_rt --0.30--> total_gen

## rmccp
- s_max=0.165, s_or=0.533, s_walk=1.867, 可达机密字段 12/12
- 真实攻击: 直接=0.165, 两跳=0.173
主要泄露链:
- [0.165] rmccp --0.17--> total_lmp_da
- [0.125] rmccp --0.12--> congestion_price_rt
- [0.098] rmccp --0.30--> marginal_loss_price_rt --0.41--> marginal_loss_price_da
- [0.088] rmccp --0.30--> marginal_loss_price_rt --0.37--> metered_load_mw
- [0.073] rmccp --0.30--> marginal_loss_price_rt --0.30--> total_gen

## gen_fuel_wind_mw
- s_max=0.164, s_or=0.526, s_walk=0.855, 可达机密字段 12/12
- 真实攻击: 直接=0.047, 两跳=0.056
主要泄露链:
- [0.164] gen_fuel_wind_mw --0.95--> gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.87--> total_gen
- [0.159] gen_fuel_wind_mw --0.95--> gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.85--> metered_load_mw
- [0.088] gen_fuel_wind_mw --0.95--> gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.47--> total_lmp_da
- [0.055] gen_fuel_wind_mw --0.95--> gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.29--> da_as_total_mw_thirty_minutes_reserve
- [0.052] gen_fuel_wind_mw --0.95--> gen_fuel_wind_pct --0.31--> gen_fuel_gas_mw --0.84--> forecast_load_mw_latest_available --0.42--> marginal_loss_price_da

## total_pjm_rmpcp_cr
- s_max=0.153, s_or=0.479, s_walk=1.136, 可达机密字段 12/12
- 真实攻击: 直接=0.153, 两跳=0.104
主要泄露链:
- [0.153] total_pjm_rmpcp_cr --0.15--> net_actual_interchange_mw
- [0.104] total_pjm_rmpcp_cr --0.10--> total_gen
- [0.093] total_pjm_rmpcp_cr --0.09--> da_as_total_mw_thirty_minutes_reserve
- [0.086] total_pjm_rmpcp_cr --0.09--> metered_load_mw
- [0.040] total_pjm_rmpcp_cr --0.11--> gen_fuel_gas_mw --0.47--> total_lmp_da

## total_pjm_reg_purchases
- s_max=0.151, s_or=0.669, s_walk=2.988, 可达机密字段 12/12
- 真实攻击: 直接=0.151, 两跳=0.152
主要泄露链:
- [0.151] total_pjm_reg_purchases --0.15--> total_lmp_da
- [0.144] total_pjm_reg_purchases --0.14--> da_as_total_mw_thirty_minutes_reserve
- [0.140] total_pjm_reg_purchases --0.14--> net_actual_interchange_mw
- [0.116] total_pjm_reg_purchases --0.12--> total_gen
- [0.113] total_pjm_reg_purchases --0.11--> metered_load_mw

## rmpcp
- s_max=0.136, s_or=0.487, s_walk=1.351, 可达机密字段 12/12
- 真实攻击: 直接=0.136, 两跳=0.083
主要泄露链:
- [0.136] rmpcp --0.14--> da_as_total_mw_thirty_minutes_reserve
- [0.087] rmpcp --0.09--> net_actual_interchange_mw
- [0.085] rmpcp --0.08--> total_gen
- [0.074] rmpcp --0.07--> metered_load_mw
- [0.070] rmpcp --0.07--> da_as_total_mw_primary_reserve

## gen_fuel_solar_pct
- s_max=0.135, s_or=0.479, s_walk=1.447, 可达机密字段 12/12
- 真实攻击: 直接=0.135, 两跳=0.121
主要泄露链:
- [0.135] gen_fuel_solar_pct --0.13--> gross_actual_interchange_mw
- [0.121] gen_fuel_solar_pct --0.12--> total_losses
- [0.073] gen_fuel_solar_pct --0.07--> metered_load_mw
- [0.070] gen_fuel_solar_pct --0.10--> gen_fuel_gas_mw --0.87--> total_gen
- [0.047] gen_fuel_solar_pct --0.39--> total_pjm_reg_purchases --0.15--> total_lmp_da

## gen_fuel_solar_mw
- s_max=0.133, s_or=0.463, s_walk=0.917, 可达机密字段 12/12
- 真实攻击: 直接=0.133, 两跳=0.107
主要泄露链:
- [0.133] gen_fuel_solar_mw --0.13--> gross_actual_interchange_mw
- [0.115] gen_fuel_solar_mw --0.12--> total_losses
- [0.055] gen_fuel_solar_mw --0.95--> gen_fuel_solar_pct --0.07--> metered_load_mw
- [0.053] gen_fuel_solar_mw --0.95--> gen_fuel_solar_pct --0.10--> gen_fuel_gas_mw --0.87--> total_gen
- [0.051] gen_fuel_solar_mw --0.05--> marginal_loss_price_da

## da_as_ss_mw_synchronized_reserve
- s_max=0.125, s_or=0.644, s_walk=4.058, 可达机密字段 12/12
- 真实攻击: 直接=0.125, 两跳=0.124
主要泄露链:
- [0.125] da_as_ss_mw_synchronized_reserve --0.13--> total_gen
- [0.125] da_as_ss_mw_synchronized_reserve --0.12--> metered_load_mw
- [0.113] da_as_ss_mw_synchronized_reserve --0.11--> total_losses
- [0.107] da_as_ss_mw_synchronized_reserve --0.11--> gross_actual_interchange_mw
- [0.101] da_as_ss_mw_synchronized_reserve --0.10--> net_actual_interchange_mw

## da_as_ss_mw_thirty_minutes_reserve
- s_max=0.125, s_or=0.644, s_walk=4.058, 可达机密字段 12/12
- 真实攻击: 直接=0.125, 两跳=0.124
主要泄露链:
- [0.125] da_as_ss_mw_thirty_minutes_reserve --0.13--> total_gen
- [0.125] da_as_ss_mw_thirty_minutes_reserve --0.12--> metered_load_mw
- [0.113] da_as_ss_mw_thirty_minutes_reserve --0.11--> total_losses
- [0.107] da_as_ss_mw_thirty_minutes_reserve --0.11--> gross_actual_interchange_mw
- [0.101] da_as_ss_mw_thirty_minutes_reserve --0.10--> net_actual_interchange_mw

## da_as_ss_mw_primary_reserve
- s_max=0.125, s_or=0.644, s_walk=4.058, 可达机密字段 12/12
- 真实攻击: 直接=0.125, 两跳=0.124
主要泄露链:
- [0.125] da_as_ss_mw_primary_reserve --0.13--> total_gen
- [0.125] da_as_ss_mw_primary_reserve --0.12--> metered_load_mw
- [0.113] da_as_ss_mw_primary_reserve --0.11--> total_losses
- [0.107] da_as_ss_mw_primary_reserve --0.11--> gross_actual_interchange_mw
- [0.101] da_as_ss_mw_primary_reserve --0.10--> net_actual_interchange_mw

## gen_fuel_storage_mw
- s_max=0.122, s_or=0.477, s_walk=1.524, 可达机密字段 12/12
- 真实攻击: 直接=0.122, 两跳=0.122
主要泄露链:
- [0.122] gen_fuel_storage_mw --0.12--> total_lmp_da
- [0.091] gen_fuel_storage_mw --0.09--> da_as_total_mw_thirty_minutes_reserve
- [0.076] gen_fuel_storage_mw --0.31--> gen_fuel_hydro_mw --0.30--> metered_load_mw
- [0.074] gen_fuel_storage_mw --0.07--> congestion_price_rt
- [0.073] gen_fuel_storage_mw --0.31--> gen_fuel_hydro_mw --0.29--> total_gen

## gen_fuel_other_renewables_mw
- s_max=0.105, s_or=0.515, s_walk=1.253, 可达机密字段 12/12
- 真实攻击: 直接=0.105, 两跳=0.107
主要泄露链:
- [0.105] gen_fuel_other_renewables_mw --0.10--> total_gen
- [0.096] gen_fuel_other_renewables_mw --0.23--> gen_fuel_nuclear_mw --0.53--> total_losses
- [0.093] gen_fuel_other_renewables_mw --0.23--> gen_fuel_nuclear_mw --0.51--> metered_load_mw
- [0.083] gen_fuel_other_renewables_mw --0.08--> net_actual_interchange_mw
- [0.068] gen_fuel_other_renewables_mw --0.23--> gen_fuel_nuclear_mw --0.38--> da_as_total_mw_thirty_minutes_reserve

## total_pjm_assigned_reg
- s_max=0.086, s_or=0.463, s_walk=0.560, 可达机密字段 12/12
- 真实攻击: 直接=0.059, 两跳=0.038
主要泄露链:
- [0.086] total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.15--> total_lmp_da
- [0.082] total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.14--> da_as_total_mw_thirty_minutes_reserve
- [0.080] total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.14--> net_actual_interchange_mw
- [0.066] total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.12--> total_gen
- [0.065] total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.11--> metered_load_mw

## total_pjm_loc_credit
- s_max=0.055, s_or=0.246, s_walk=0.432, 可达机密字段 12/12
- 真实攻击: 直接=0.054, 两跳=0.052
主要泄露链:
- [0.055] total_pjm_loc_credit --0.14--> total_lmp_rt --0.50--> total_lmp_da
- [0.050] total_pjm_loc_credit --0.14--> total_lmp_rt --0.45--> congestion_price_rt
- [0.040] total_pjm_loc_credit --0.14--> total_lmp_rt --0.36--> marginal_loss_price_da
- [0.036] total_pjm_loc_credit --0.14--> total_lmp_rt --0.32--> metered_load_mw
- [0.030] total_pjm_loc_credit --0.14--> total_lmp_rt --0.27--> total_gen

## total_pjm_self_sched_reg
- s_max=0.011, s_or=0.079, s_walk=0.028, 可达机密字段 12/12
- 真实攻击: 直接=0.009, 两跳=0.000
主要泄露链:
- [0.011] total_pjm_self_sched_reg --0.17--> total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.15--> total_lmp_da
- [0.011] total_pjm_self_sched_reg --0.17--> total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.14--> da_as_total_mw_thirty_minutes_reserve
- [0.011] total_pjm_self_sched_reg --0.17--> total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.14--> net_actual_interchange_mw
- [0.009] total_pjm_self_sched_reg --0.17--> total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.12--> total_gen
- [0.009] total_pjm_self_sched_reg --0.17--> total_pjm_assigned_reg --0.72--> total_pjm_reg_purchases --0.11--> metered_load_mw

## net_inadv_interchange_mw
- s_max=0.000, s_or=0.000, s_walk=0.000, 可达机密字段 0/12
- 真实攻击: 直接=0.000, 两跳=0.000

