# DNN_Aggresvation26 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目26要解决的问题

DNN25 的 staged linear 成为当前最强 allloss：

```text
phase1_epochs = 100
all_MSE = 0.431317
net_actual_interchange_mw = -0.6866
gross_actual_interchange_mw = 0.1835
```

它显著改善 net，但 gross 低于 DNN23 的 0.3044。因此 DNN26 只调整 staged training 的 phase1 长度：

```text
phase1_epochs = 60
phase1_epochs = 140
```

目标是判断 phase1=100 是否最合适，还是可以通过更短或更长预训练取得更好的 net/gross 平衡。

### 2. 实验设置

两组实验：

```text
staged_linear_phase60_window4_allloss_netgross16
staged_linear_phase140_window4_allloss_netgross16
```

共同设置：

```text
input_encoder = linear
window_size = 4
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
phase1: 排除 net/gross
phase2: 全部 12 个 confidential target
```

### 3. 测试结果

#### 3.1 phase1 = 60

```text
all_MSE = 0.436862
```

关键 target R2：

```text
metered_load_mw                         0.9656
da_as_total_mw_primary_reserve          0.9597
total_gen                               0.9471
da_as_total_mw_synchronized_reserve     0.8918
total_lmp_da                            0.8648
total_losses                            0.6963
marginal_loss_price_da                  0.4890
congestion_price_rt                     0.4479
da_as_total_mw_thirty_minutes_reserve   0.3708
gross_actual_interchange_mw             0.3270
congestion_price_da                     0.2434
net_actual_interchange_mw              -0.8354
```

#### 3.2 phase1 = 140

```text
all_MSE = 0.449831
```

关键 target R2：

```text
metered_load_mw                         0.9666
total_gen                               0.9527
da_as_total_mw_primary_reserve          0.9505
da_as_total_mw_synchronized_reserve     0.8851
total_lmp_da                            0.8676
total_losses                            0.6871
marginal_loss_price_da                  0.4773
congestion_price_rt                     0.4482
da_as_total_mw_thirty_minutes_reserve   0.3667
congestion_price_da                     0.2587
gross_actual_interchange_mw             0.1815
net_actual_interchange_mw              -0.9202
```

### 4. 与 DNN25 对比

```text
DNN26 phase60:
  all_MSE = 0.436862
  gross   = 0.3270
  net     = -0.8354

DNN25 phase100:
  all_MSE = 0.431317
  gross   = 0.1835
  net     = -0.6866

DNN26 phase140:
  all_MSE = 0.449831
  gross   = 0.1815
  net     = -0.9202
```

### 5. 结果分析

DNN26 说明 phase1 长度存在明显 trade-off：

```text
phase1=60:
  gross最好，但net变差，整体不如DNN25。

phase1=100:
  all_MSE最好，net最好，是当前最优折中。

phase1=140:
  预训练过久，phase2难以修正net/gross，整体变差。
```

这说明 staged training 的确有效，但不能无限延长 phase1。过短时 net/gross 过早扰动共享表示；过长时共享表示过度适配 10 个稳定 target，net/gross 难以融入。

### 6. 当前结论

DNN26 不替代 DNN25。当前推荐：

```text
allloss主线:
  DNN25 staged_linear_window4_allloss_netgross16

gross优先对照:
  DNN26 staged_linear_phase60_window4_allloss_netgross16

maskloss主线:
  DNN21 window3 或 DNN23 linear window4
```

继续提升的关键不再是简单延长 phase1，而可能需要：

```text
1. 对 net/gross 单独建 temporal/edge 分支；
2. 对不同 target 使用不同 temporal window；
3. 或承认 net_actual_interchange_mw 当前字段集合下存在较强不可推断性。
```
