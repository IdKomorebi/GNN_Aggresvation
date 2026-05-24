# DNN_Aggresvation23 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目23要解决的问题

DNN21 证明 `window_size=3` 是正向结果；DNN22 证明 `window_size=6` 对 `total_losses` 和 `marginal_loss_price_da` 有帮助，但整体不如 window=3。

因此 DNN23 测试折中：

```text
window_size = 4
```

目标是同时保留 DNN21 的整体 MSE 优势，以及 DNN22 对 `total_losses / marginal_loss_price_da` 的提升。

### 2. 实验设置

两组实验：

```text
temporal_window4_maskloss
temporal_window4_allloss_netgross16
```

设置：

```text
General-General: fixed weighted GCN
Confidential-Source: edge-specific gated attention
top_k = 8
window_size = 4
```

allloss 组：

```text
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
```

### 3. 测试结果

#### 3.1 maskloss

```text
target_MSE = 0.333296
all_MSE    = 0.543267
excluded_MSE = 1.593118
```

关键 target R2：

```text
metered_load_mw                         0.9660
da_as_total_mw_primary_reserve          0.9546
total_gen                               0.9413
da_as_total_mw_synchronized_reserve     0.9152
total_lmp_da                            0.7503
total_losses                            0.6912
marginal_loss_price_da                  0.4962
congestion_price_rt                     0.4710
da_as_total_mw_thirty_minutes_reserve   0.3749
congestion_price_da                     0.3010
```

#### 3.2 allloss

```text
all_MSE = 0.436237
```

关键 target R2：

```text
da_as_total_mw_primary_reserve          0.9653
metered_load_mw                         0.9640
total_gen                               0.9566
da_as_total_mw_synchronized_reserve     0.8989
total_lmp_da                            0.8480
total_losses                            0.6862
marginal_loss_price_da                  0.4675
congestion_price_rt                     0.4191
da_as_total_mw_thirty_minutes_reserve   0.4091
gross_actual_interchange_mw             0.3044
congestion_price_da                     0.2714
net_actual_interchange_mw              -0.8312
```

### 4. 与 DNN21/DNN22 对比

maskloss：

```text
DNN21 window3:
  target_MSE = 0.331438
  cp_da      = 0.2996
  losses     = 0.6605
  mlp_da     = 0.4472

DNN22 window6:
  target_MSE = 0.335776
  cp_da      = 0.2746
  losses     = 0.6993
  mlp_da     = 0.4960

DNN23 window4:
  target_MSE = 0.333296
  cp_da      = 0.3010
  losses     = 0.6912
  mlp_da     = 0.4962
```

allloss：

```text
DNN21 window3:
  all_MSE = 0.439227
  gross   = 0.2147
  net     = -0.8514

DNN22 window6:
  all_MSE = 0.449651
  gross   = 0.0317
  net     = -0.8186

DNN23 window4:
  all_MSE = 0.436237
  gross   = 0.3044
  net     = -0.8312
```

### 5. 结果分析

DNN23 是当前最有价值的结果之一：

```text
1. window=4 的 allloss 达到当前最好 all_MSE = 0.436237；
2. gross_actual_interchange_mw 提升到 0.3044，是目前最好的 gross 结果；
3. maskloss 下 cp_da 达到 0.3010，是目前较高水平；
4. maskloss 下 total_losses 和 mlp_da 接近 window=6 的强项；
5. net_actual_interchange_mw 仍为负，说明它可能需要更强时序结构或不可稳定推断。
```

与 DNN21 相比，DNN23 的 maskloss 整体 MSE 稍差，但弱 target 更均衡；与 DNN22 相比，DNN23 的 allloss 明显更好。

### 6. 当前结论

DNN23 可以作为新的 allloss 主线：

```text
Stable graph backbone + window_size=4 + net/gross=16
```

下一步不应继续随意改边或 loss，而应沿时间建模继续：

```text
当前时间窗口输入编码器只是 Linear(window_size -> hidden_dim) + ReLU。
```

因此 DNN24 将测试轻量 temporal MLP encoder，在不改变图结构的前提下增强短时序特征提取能力。
