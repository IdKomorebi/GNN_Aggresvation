# DNN-q 缓存

该目录缓存 General 字段到 Confidential 字段的 DNN 样本外预测 R²。

- `dnn_q_scores.csv`: 每个 General 字段聚合后的 `q_i`。
- `general_to_confidential_predictive_r2.csv`: 每个 General 到每个 Confidential 的 `q_{i,c}`。
- `metadata.json`: 生成该缓存时使用的字段和 DNN 参数签名。
