# DNN_Aggresvation20 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目20要解决的问题

DNN19 证明单纯通过 loss weighting 强拉弱 target 会破坏共享表示，尤其会让 `da_as_total_mw_thirty_minutes_reserve` 和整体 MSE 明显变差。

因此 DNN20 回到不改 loss 的思路，只测试：

```text
target-specific source subgraph expansion
```

即普通字段仍保持 `top_k=8`，只给困难 confidential target 增加候选 source 数量，验证短板是否来自 top8 截断。

### 2. 实验设置

两组实验：

```text
source_expand_maskloss
source_expand_allloss_netgross16
```

maskloss 扩边：

```text
total_losses: 16
congestion_price_da: 16
congestion_price_rt: 16
marginal_loss_price_da: 16
da_as_total_mw_thirty_minutes_reserve: 12
```

allloss 额外补边：

```text
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
```

### 3. 实际构图结果

maskloss：

```text
total_losses: degree 19 -> 19 (+0)
congestion_price_da: degree 8 -> 16 (+8)
congestion_price_rt: degree 11 -> 16 (+5)
marginal_loss_price_da: degree 21 -> 21 (+0)
da_as_total_mw_thirty_minutes_reserve: degree 18 -> 18 (+0)
总边数: 921
```

allloss：

```text
net_actual_interchange_mw: degree 9 -> 16 (+7)
gross_actual_interchange_mw: degree 11 -> 17 (+6)
total_losses: degree 19 -> 19 (+0)
congestion_price_da: degree 8 -> 16 (+8)
congestion_price_rt: degree 11 -> 16 (+5)
marginal_loss_price_da: degree 21 -> 21 (+0)
da_as_total_mw_thirty_minutes_reserve: degree 18 -> 18 (+0)
总边数: 934
```

### 4. 测试结果

#### 4.1 maskloss

```text
target_MSE = 0.355969
all_MSE    = 0.561947
excluded_MSE = 1.591833
```

关键 target R2：

```text
total_losses                            0.5685
marginal_loss_price_da                  0.5002
congestion_price_rt                     0.4617
da_as_total_mw_thirty_minutes_reserve   0.3184
congestion_price_da                     0.2226
```

对比 DNN14 sharp：

```text
DNN14 sharp:
  target_MSE = 0.331766
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926
```

DNN20 maskloss 只改善 `marginal_loss_price_da`，但 `cp_da`、`total_losses`、`30min` 都明显下降。

#### 4.2 allloss

```text
all_MSE = 0.462766
```

关键 target R2：

```text
total_losses                            0.6528
congestion_price_rt                     0.4763
marginal_loss_price_da                  0.4689
da_as_total_mw_thirty_minutes_reserve   0.4169
congestion_price_da                     0.1985
gross_actual_interchange_mw             0.1062
net_actual_interchange_mw              -1.0554
```

allloss 也没有超过 DNN15/DNN18。

### 5. 结果分析

DNN20 说明：

```text
1. 困难 target 的问题不只是 top_k=8 截断；
2. 扩边会增加候选 source，但也会增加注意力分配难度和噪声；
3. cp_da 对扩边非常敏感，更多边反而降低 R2；
4. total_losses 原本 degree 已经较高，扩边没有新增信息；
5. 30min 原本 degree 也已经高，问题不是边数不足。
```

更重要的是，当前所有实验仍然使用：

```text
window_size = 1
```

即模型只看当前时刻的一行 general data 去预测当前时刻 confidential data。如果某些字段存在滞后影响、调度延迟或价格形成延迟，那么仅靠同一时刻图结构调参很难突破。

### 6. 当前结论

DNN20 不建议作为主线。下一步应测试时间窗口输入：

```text
1. 保持稳定图主干；
2. 不再调 loss；
3. 不再盲目扩边；
4. 将 window_size 从 1 提高到 3，允许每个 general 节点携带最近几个时刻的信息。
```

因此 DNN21 将测试 short temporal window input。
