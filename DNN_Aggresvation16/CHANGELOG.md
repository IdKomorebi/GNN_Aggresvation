# DNN_Aggresvation16 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目16要验证的问题

DNN13-DNN15 的主线中，General-General 部分使用：

```text
全局 alpha_G 融合多相关系数
固定相关性边权
GCN 加权聚合
```

本轮测试一个结构性改动：

```text
全局 alpha_G 仍然保留；
但 General-General 聚合从 fixed weighted GCN 改为 edge-prior GAT。
```

也就是说，General-General 的相关系数权重仍然是全局共享的，但聚合不再只是：

```text
m_i = sum_j A_ij h_j
```

而是改为：

```text
score_ij = q_i^T k_j / sqrt(d) + scale * log(p_ij)
alpha_ij = softmax_j(score_ij)
m_i = sum_j alpha_ij v_j
```

其中 `p_ij` 是由全局 `alpha_G` 融合多相关性指标得到的 General-General 边先验。

Confidential-Source 部分不改，仍使用 DNN13/DNN14 的 target-conditioned gated prior attention。

### 2. 实现内容

在 `src/model.py` 中新增 architecture：

```text
hybrid_bilinear_gated_general_gat
```

新增 General-GAT 所需参数：

```text
general_query_layers
general_key_layers
general_value_layers
general_prior_scales
```

当 architecture 为 `hybrid_bilinear_gated_general_gat` 时，General 节点之间使用 GAT 聚合；其他 architecture 仍保留原 fixed weighted GCN 逻辑。

### 3. 实验设置

本轮只跑两组：

```text
general_gat_maskloss
general_gat_allloss_netgross16
```

其中：

```text
general_gat_maskloss:
  top_k = 8
  threshold = 0.25
  不额外补边
  排除 net_actual_interchange_mw 和 gross_actual_interchange_mw 的 loss
```

```text
general_gat_allloss_netgross16:
  top_k = 8
  threshold = 0.25
  net_actual_interchange_mw   至少 top16
  gross_actual_interchange_mw 至少 top16
  12 个 confidential target 全部进入 loss
```

allloss 实际补边结果：

```text
target_field                  target_top_k  degree_before  degree_after  added_edges
net_actual_interchange_mw      16            9              16            7
gross_actual_interchange_mw    16            11             17            6
```

### 4. 测试结果

```text
experiment                    MSE       cp_da   cp_rt   losses  mlp_da  30min   gross    net
general_gat_maskloss          0.339382  0.2486  0.4738  0.6265  0.4348  0.4066  excl.   excl.
general_gat_allloss_netgross16 0.451012 0.2333  0.4529  0.6543  0.4341  0.3413  0.2111 -0.8511
```

对照关键结果：

```text
DNN14 sharp_attention_maskloss:
  target_MSE = 0.331766
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926

DNN15 target_extra_edges_allloss:
  all_MSE = 0.439477
  cp_da   = 0.2407
  cp_rt   = 0.4817
  losses  = 0.6443
  mlp_da  = 0.4624
  30min   = 0.3666
  gross   = 0.2367
  net     = -0.8812
```

### 5. 结果分析

#### 5.1 maskloss

General-GAT maskloss 的整体结果：

```text
target_MSE = 0.339382
```

没有超过 DNN14 sharp：

```text
target_MSE = 0.331766
```

局部看，`da_as_total_mw_thirty_minutes_reserve` 有提升：

```text
DNN14 sharp:       0.3926
DNN16 General-GAT: 0.4066
```

但 price/loss 类字段普遍下降：

```text
congestion_price_da:    0.2678 -> 0.2486
congestion_price_rt:    0.4796 -> 0.4738
total_losses:           0.6768 -> 0.6265
marginal_loss_price_da: 0.4832 -> 0.4348
```

因此 General-GAT 没有成为更好的 maskloss 主线。

#### 5.2 allloss

General-GAT allloss 的结果：

```text
all_MSE = 0.451012
```

没有超过 DNN15 当前最好的 allloss：

```text
target_extra_edges_allloss all_MSE = 0.439477
```

`net_actual_interchange_mw` 略好一点：

```text
DNN15 target_extra_edges_allloss: -0.8812
DNN16 General-GAT allloss:        -0.8511
```

但 `gross_actual_interchange_mw`、`congestion_price_rt`、`marginal_loss_price_da` 等都更弱：

```text
gross:  0.2367 -> 0.2111
cp_rt:  0.4817 -> 0.4529
mlp_da: 0.4624 -> 0.4341
```

因此 General-GAT allloss 也不建议替代 DNN15 的 allloss 对照。

### 6. 当前结论

DNN16 证明：

```text
1. General-General 从 GCN 改成 GAT 并没有带来整体提升；
2. GAT 对 30min_reserve 有帮助，但牺牲了 price/loss 类字段；
3. 当前数据规模下，General-General 动态注意力可能引入了额外自由度和噪声；
4. General-General fixed weighted GCN 反而更稳定；
5. 主线仍应回到 DNN14 sharp maskloss；
6. allloss 对照仍建议使用 DNN15 target_extra_edges_allloss。
```

因此当前推荐：

```text
maskloss 主线:
  DNN14 sharp_attention_maskloss

allloss 对照:
  DNN15 target_extra_edges_allloss

DNN16 General-GAT:
  作为消融实验保留，不作为主结果。
```
