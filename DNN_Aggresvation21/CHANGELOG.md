# DNN_Aggresvation21 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目21要解决的问题

DNN18-DNN20 说明继续在 loss weighting 或边数量上调参收益有限：

```text
DNN18 uncertainty weighting:
  权重分化明显，但偏向 easy target，弱 target 没有被救起来。

DNN19 difficulty weighting:
  cp_da 小幅提升，但整体 MSE 和 30min 明显变差。

DNN20 source expansion:
  扩边没有解决 cp_da/total_losses/30min，说明 top8 截断不是主要瓶颈。
```

因此 DNN21 转向时间维度。之前所有主线几乎都使用：

```text
window_size = 1
```

这意味着模型只看当前时刻 general data 去预测当前时刻 confidential data。如果某些价格、备用、交换功率字段存在滞后关系，仅靠同一时刻图结构很难突破。

### 2. 修改思路

DNN21 不改模型主干、不改 loss、不盲目扩边，只把输入窗口改为：

```text
window_size = 3
```

即每个节点输入从当前时刻的 1 个值变为最近 3 个时刻的序列：

```text
h_i^{(0)} = [x_i(t-2), x_i(t-1), x_i(t)]
```

confidential 节点在输入窗口中仍然被置零，防止泄漏：

```text
feat[confidential_indices, :] = 0.0
```

预测目标仍为窗口最后一个时刻的 confidential value：

```text
y(t)
```

### 3. 实验设置

两组实验：

```text
temporal_window3_maskloss
temporal_window3_allloss_netgross16
```

maskloss：

```text
top_k = 8
window_size = 3
排除 net_actual_interchange_mw 和 gross_actual_interchange_mw
```

allloss：

```text
top_k = 8
window_size = 3
net_actual_interchange_mw 补边到 16
gross_actual_interchange_mw 补边到 16
12 个 confidential target 全部进入 loss
```

### 4. 测试结果

#### 4.1 maskloss

```text
target_MSE = 0.331438
all_MSE    = 0.541643
excluded_MSE = 1.592671
```

关键 target R2：

```text
metered_load_mw                         0.9666
da_as_total_mw_primary_reserve          0.9608
total_gen                               0.9357
da_as_total_mw_synchronized_reserve     0.9319
total_lmp_da                            0.8608
total_losses                            0.6605
congestion_price_rt                     0.4673
marginal_loss_price_da                  0.4472
da_as_total_mw_thirty_minutes_reserve   0.3903
congestion_price_da                     0.2996
```

对比 DNN14 sharp maskloss：

```text
DNN14 sharp:
  target_MSE = 0.331766
  all_MSE    = 0.541777
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926
```

DNN21 maskloss 在整体 target_MSE 上略超 DNN14，并显著改善 `congestion_price_da`：

```text
0.2678 -> 0.2996
```

代价是 `marginal_loss_price_da` 和 `total_lmp_da` 有下降。

#### 4.2 allloss

```text
all_MSE = 0.439227
```

关键 target R2：

```text
da_as_total_mw_primary_reserve          0.9622
total_gen                               0.9613
metered_load_mw                         0.9583
da_as_total_mw_synchronized_reserve     0.9177
total_lmp_da                            0.7609
total_losses                            0.6871
congestion_price_rt                     0.4860
marginal_loss_price_da                  0.4756
da_as_total_mw_thirty_minutes_reserve   0.3344
congestion_price_da                     0.2705
gross_actual_interchange_mw             0.2147
net_actual_interchange_mw              -0.8514
```

对比 DNN15 较强 allloss：

```text
DNN15 target_extra_edges_allloss:
  all_MSE = 0.439477
```

DNN21 allloss 略优于 DNN15，并显著改善 `total_losses`：

```text
0.6443 -> 0.6871
```

### 5. 结果分析

DNN21 是 DNN18-DNN21 中第一个真正有价值的正向结果。

主要说明：

```text
1. 部分 confidential target 的信息不只在当前时刻，而存在短期时间依赖；
2. window_size=3 对 cp_da、total_losses、allloss整体有帮助；
3. 这比继续调 loss 或盲目扩边更合理；
4. 但 window_size=3 也会改变不同 target 的取舍，例如 total_lmp_da 和 mlp_da 有下降；
5. net_actual_interchange_mw 仍然为负，说明它可能需要更强时序结构或确实不可由当前字段稳定推断。
```

训练中 alpha 明显偏向 Pearson：

```text
maskloss best附近 alpha ≈ Pearson dominant
allloss  best附近 alpha ≈ Pearson dominant
```

这可能表示短时间窗口已经让线性滞后关系更容易被模型捕捉。

### 6. 当前结论

DNN21 可以作为新的重要候选主线之一：

```text
Stable graph backbone + window_size=3
```

下一步应继续验证：

```text
1. window_size=6 是否继续提升；
2. 如果 window_size=6 变差，说明 3 小时短窗口更合适；
3. 如果 window_size=6 继续提升，可以再考虑轻量 temporal encoder，而不是继续调图边。
```

因此 DNN22 将测试 `window_size=6`。
