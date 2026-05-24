# DNN_Aggresvation25 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目25要解决的问题

DNN23/DNN24 说明时间窗口是有效方向：

```text
DNN23 linear window4 allloss:
  all_MSE = 0.436237

DNN24 MLP window4 allloss:
  all_MSE = 0.435431
```

但 allloss 从一开始就引入 `net_actual_interchange_mw` 和 `gross_actual_interchange_mw`，可能会在训练早期扰乱共享图表示。

因此 DNN25 测试 staged training：

```text
phase 1: 先训练稳定10个target，排除net/gross
phase 2: 再引入全部12个target进行微调
```

### 2. 修改内容

在 `src/train.py` 中新增 staged training 支持：

```text
training:
  staged_training:
    enabled: true
    phase1_epochs: 100
    phase1_loss_exclude_confidential:
      - net_actual_interchange_mw
      - gross_actual_interchange_mw
```

训练逻辑：

```text
epoch <= phase1_epochs:
  使用 phase1 target 集合训练

epoch > phase1_epochs:
  使用最终 target 集合训练
  从 phase2 开始记录 best_state 和 early stopping
```

### 3. 实验设置

两组实验：

```text
staged_linear_window4_allloss_netgross16
staged_mlp_window4_allloss_netgross16
```

共同设置：

```text
window_size = 4
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
phase1_epochs = 100
```

区别：

```text
staged_linear:
  input_encoder = linear

staged_mlp:
  input_encoder = mlp
```

### 4. 测试结果

#### 4.1 staged linear

```text
all_MSE = 0.431317
```

关键 target R2：

```text
metered_load_mw                         0.9679
total_gen                               0.9566
da_as_total_mw_primary_reserve          0.9543
da_as_total_mw_synchronized_reserve     0.8936
total_lmp_da                            0.8685
total_losses                            0.6815
marginal_loss_price_da                  0.4713
congestion_price_rt                     0.4612
da_as_total_mw_thirty_minutes_reserve   0.3749
congestion_price_da                     0.2625
gross_actual_interchange_mw             0.1835
net_actual_interchange_mw              -0.6866
```

#### 4.2 staged MLP

```text
all_MSE = 0.464593
```

关键 target R2：

```text
metered_load_mw                         0.9690
da_as_total_mw_primary_reserve          0.9667
total_gen                               0.9578
da_as_total_mw_synchronized_reserve     0.9401
total_lmp_da                            0.8876
total_losses                            0.6649
marginal_loss_price_da                  0.4815
congestion_price_rt                     0.4647
da_as_total_mw_thirty_minutes_reserve   0.3624
congestion_price_da                     0.2542
gross_actual_interchange_mw             0.0631
net_actual_interchange_mw              -1.1635
```

### 5. 结果分析

DNN25 的 staged linear 是目前最好的 allloss 结果：

```text
DNN23 linear window4 allloss: 0.436237
DNN24 MLP window4 allloss:    0.435431
DNN25 staged linear allloss:  0.431317
```

主要收益来自：

```text
net_actual_interchange_mw:
  DNN24: -0.7847
  DNN25 staged linear: -0.6866
```

说明先训练稳定 target，再引入 net/gross，确实能缓和 net 对共享表示的破坏。

但 staged linear 也有代价：

```text
gross_actual_interchange_mw:
  DNN23: 0.3044
  DNN24: 0.2440
  DNN25: 0.1835
```

这说明 phase1 排除 net/gross 过久后，gross 虽然进入 phase2，但可能已经错过了对共享表示的早期塑形。

staged MLP 结果明显较差，说明：

```text
1. staged training 更适合线性时间窗口编码；
2. MLP encoder 的额外容量在 staged setting 下反而过拟合或扰乱 phase2；
3. 后续不应继续用 staged MLP。
```

### 6. 当前结论

当前最强 allloss 主线更新为：

```text
DNN25 staged_linear_window4_allloss_netgross16
```

下一步应只调 staged linear 的 phase1 长度：

```text
phase1_epochs = 60
phase1_epochs = 140
```

目标是找到 net/gross 的更好平衡：

```text
phase1过短: 可能回到普通allloss，net扰动较大
phase1过长: gross可能受损
```
