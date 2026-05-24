# DNN_Aggresvation22 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目22要解决的问题

DNN21 的 `window_size=3` 是一个正向结果，说明短期时间窗口能帮助部分 confidential target，尤其是：

```text
congestion_price_da
total_losses
allloss整体MSE
```

DNN22 继续测试更长短期历史：

```text
window_size = 6
```

目的是判断时间窗口收益是否随长度继续增长，还是 3 小时附近已经是较好的折中。

### 2. 实验设置

两组实验：

```text
temporal_window6_maskloss
temporal_window6_allloss_netgross16
```

其余设置保持 DNN21 一致：

```text
General-General: fixed weighted GCN
Confidential-Source: edge-specific gated attention
top_k = 8
attention_temperature = 1.0
edge_alpha_temperature = 0.35
alpha_init_std = 1.0
```

allloss 组继续：

```text
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
```

### 3. 测试结果

#### 3.1 maskloss

```text
target_MSE = 0.335776
all_MSE    = 0.545416
excluded_MSE = 1.593613
```

关键 target R2：

```text
metered_load_mw                         0.9647
da_as_total_mw_primary_reserve          0.9637
total_gen                               0.9320
da_as_total_mw_synchronized_reserve     0.9240
total_lmp_da                            0.8637
total_losses                            0.6993
marginal_loss_price_da                  0.4960
congestion_price_rt                     0.4383
da_as_total_mw_thirty_minutes_reserve   0.3814
congestion_price_da                     0.2746
```

#### 3.2 allloss

```text
all_MSE = 0.449651
```

关键 target R2：

```text
metered_load_mw                         0.9648
da_as_total_mw_primary_reserve          0.9588
total_gen                               0.9524
da_as_total_mw_synchronized_reserve     0.9298
total_lmp_da                            0.8842
total_losses                            0.6606
marginal_loss_price_da                  0.4875
congestion_price_rt                     0.4197
da_as_total_mw_thirty_minutes_reserve   0.3762
congestion_price_da                     0.2651
gross_actual_interchange_mw             0.0317
net_actual_interchange_mw              -0.8186
```

### 4. 与 DNN21 对比

maskloss：

```text
DNN21 window3:
  target_MSE = 0.331438
  cp_da      = 0.2996
  cp_rt      = 0.4673
  losses     = 0.6605
  mlp_da     = 0.4472
  30min      = 0.3903

DNN22 window6:
  target_MSE = 0.335776
  cp_da      = 0.2746
  cp_rt      = 0.4383
  losses     = 0.6993
  mlp_da     = 0.4960
  30min      = 0.3814
```

allloss：

```text
DNN21 window3:
  all_MSE = 0.439227
  losses  = 0.6871
  cp_rt   = 0.4860
  cp_da   = 0.2705
  gross   = 0.2147
  net     = -0.8514

DNN22 window6:
  all_MSE = 0.449651
  losses  = 0.6606
  cp_rt   = 0.4197
  cp_da   = 0.2651
  gross   = 0.0317
  net     = -0.8186
```

### 5. 结果分析

DNN22 的结论是：

```text
1. 6小时窗口对 total_losses 和 marginal_loss_price_da 有明显帮助；
2. 但 6小时窗口损害 cp_da、cp_rt 和 allloss整体；
3. window_size=6 比 window_size=3 更容易引入冗余历史噪声；
4. 不同 confidential target 的最佳时间窗口可能不同。
```

这说明“时间维度是对的”，但简单全局窗口长度并不完美。

### 6. 当前结论

DNN22 不替代 DNN21 作为整体主线，但它提供了重要线索：

```text
window=3 更适合整体；
window=6 更适合 total_losses / marginal_loss_price_da；
不同target可能需要不同时间尺度。
```

下一步先测试折中窗口：

```text
window_size = 4
```

如果 window=4 能兼顾 DNN21 的整体表现和 DNN22 的 losses/mlp 优势，则可作为新主线；否则后续需要 target-specific temporal encoder 或多窗口融合。
