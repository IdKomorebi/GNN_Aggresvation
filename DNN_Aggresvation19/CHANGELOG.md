# DNN_Aggresvation19 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目19要解决的问题

DNN18 的 uncertainty weighting 能让 target loss 权重分化，但它把易拟合字段权重推得最高，把困难字段权重压低。这符合不确定性建模逻辑，却不符合当前“补短板”的目标。

因此 DNN19 反向测试：

```text
difficulty-aware static target weighting
```

即手动降低 easy target 的 loss 权重，提高困难 target 的 loss 权重，看能否拉高 `congestion_price_da`、`total_losses`、`thirty_minutes_reserve` 等字段。

### 2. 实验设置

本轮运行两组：

```text
difficulty_weighted_maskloss
difficulty_weighted_allloss_netgross16
```

核心权重：

```text
total_gen: 0.6
metered_load_mw: 0.6
total_lmp_da: 0.7
da_as_total_mw_primary_reserve: 0.8
da_as_total_mw_synchronized_reserve: 0.9
total_losses: 2.5
congestion_price_da: 3.0
congestion_price_rt: 2.0
marginal_loss_price_da: 2.0
da_as_total_mw_thirty_minutes_reserve: 2.2
```

allloss 组额外设置：

```text
net_actual_interchange_mw: 0.5
gross_actual_interchange_mw: 0.8
```

并继续补边：

```text
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
```

### 3. 测试结果

#### 3.1 maskloss

```text
target_MSE = 0.349884
all_MSE    = 0.556875
excluded_MSE = 1.591833
```

关键 target R2：

```text
congestion_price_da                     0.2731
congestion_price_rt                     0.4861
total_losses                            0.6231
marginal_loss_price_da                  0.4723
da_as_total_mw_thirty_minutes_reserve   0.2789
```

对比 DNN14 sharp maskloss：

```text
DNN14 sharp:
  target_MSE = 0.331766
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926
```

DNN19 maskloss 只小幅改善 `cp_da` 和 `cp_rt`，但明显损害 `total_losses` 和 `30min`，整体 MSE 也变差。

#### 3.2 allloss

```text
all_MSE = 0.495329
```

关键 target R2：

```text
congestion_price_da                     0.1936
congestion_price_rt                     0.4635
total_losses                            0.6317
marginal_loss_price_da                  0.4609
da_as_total_mw_thirty_minutes_reserve   0.2214
gross_actual_interchange_mw             0.0188
net_actual_interchange_mw              -1.1563
```

allloss 明显差于 DNN15/DNN18，不建议保留。

### 4. 结果分析

DNN19 说明简单静态加权不是好方向：

```text
1. 增大 cp_da 权重确实能把 cp_da 从约 0.24-0.27 推到 0.273；
2. 但共享表示被困难字段牵引后，30min、total_losses 和 easy target 明显下降；
3. allloss 中即使降低 net/gross 权重，两个 interchange target 仍然拖累整体；
4. loss weighting 改变的是训练偏好，不会创造新的可用 source 信息。
```

更关键的是，DNN19 训练中相关系数全局 alpha 被推到 Pearson 过度占优：

```text
maskloss epoch 149 alpha ≈ [0.8838, 0.0249, 0.0285, 0.0286, 0.0342]
allloss  epoch 189 alpha ≈ [0.7867, 0.0362, 0.0437, 0.0850, 0.0483]
```

这说明强行拉弱 target 会让模型退化到更线性的 Pearson 相关解释，反而丢掉 NMI/非线性相关信息。

### 5. 当前结论

DNN19 不建议作为主线。

下一步应停止继续调 loss，转向图结构本身：

```text
1. 保持普通 target 的 top_k=8；
2. 只给困难 target 扩展候选 source 子图；
3. 不改变 loss 权重；
4. 检查是否是 source 信息被 top_k 截断，而不是训练权重不足。
```

因此 DNN20 将测试 target-specific source subgraph expansion。
