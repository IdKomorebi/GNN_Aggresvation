# 本次运行说明

- 源配置文件：`/Users/haocun/Desktop/AllProjects/GNN_Aggresvation/DNN_Aggresvation1/configs/config.yaml`
- 运行输出目录：`/Users/haocun/Desktop/AllProjects/GNN_Aggresvation/DNN_Aggresvation1/outputs/run20260520_145148`

## data

- `csv_path`: `../pjm_rto_hourly_2025_aligned_processed_one_header.csv`
- `downsample_size`: `300`

## pseudo_label

- `general_label_count`: `20`
- `high_risk_ratio`: `0.7`
- `inference_weight`: `0.65`
- `predictive_test_ratio`: `0.3`
- `dnn_epochs`: `80`
- `dnn_hidden_dim`: `16`
- `dnn_lr`: `0.01`
- `dnn_weight_decay`: `0.0001`
- `dnn_seed`: `2026`
- `dnn_q_cache_dir`: `outputs/dnn_q_cache`

## graph

- `k_neighbors`: `5`
- `theta`: `0.5`

## model

- `hidden_dim`: `64`
- `num_layers`: `2`

## training

- `epochs`: `300`
- `lr`: `0.002`
- `beta_lr`: `0.02`
- `alpha_temperature`: `0.7`
- `w_c`: `2.0`
- `lmbda`: `0.0`
- `seed`: `42`

## outputs

- `dir`: `outputs`

