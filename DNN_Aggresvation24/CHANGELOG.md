# DNN_Aggresvation24 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目24要解决的问题

DNN23 的 `window_size=4` 成为新的重要候选主线，尤其 allloss 达到：

```text
all_MSE = 0.436237
```

但 DNN23 中每个节点的时间窗口编码仍然很简单：

```text
Linear(window_size -> hidden_dim) + ReLU
```

因此 DNN24 测试更强一点的时间窗口编码器：

```text
Temporal MLP encoder
```

目标是在不改变图结构、不改变 loss、不继续扩边的情况下，提高短时序模式表达能力。

### 2. 修改内容

在 `src/model.py` 中新增：

```text
input_encoder: linear | mlp
input_encoder_dropout
```

当：

```text
input_encoder = mlp
```

节点输入编码从：

```text
Linear(input_dim, hidden_dim) -> ReLU
```

改为：

```text
Linear(input_dim, hidden_dim)
-> ReLU
-> Dropout(input_encoder_dropout)
-> Linear(hidden_dim, hidden_dim)
-> ReLU
```

DNN24 使用：

```text
window_size = 4
input_encoder = mlp
input_encoder_dropout = 0.05
```

### 3. 实验设置

两组实验：

```text
temporal_mlp_window4_maskloss
temporal_mlp_window4_allloss_netgross16
```

allloss 组继续：

```text
net_actual_interchange_mw: 16
gross_actual_interchange_mw: 16
```

### 4. 测试结果

#### 4.1 maskloss

```text
target_MSE = 0.336127
all_MSE    = 0.545625
excluded_MSE = 1.593118
```

关键 target R2：

```text
metered_load_mw                         0.9665
da_as_total_mw_primary_reserve          0.9637
total_gen                               0.9536
da_as_total_mw_synchronized_reserve     0.9245
total_lmp_da                            0.7764
total_losses                            0.6631
marginal_loss_price_da                  0.4760
congestion_price_rt                     0.4749
da_as_total_mw_thirty_minutes_reserve   0.3285
congestion_price_da                     0.2992
```

#### 4.2 allloss

```text
all_MSE = 0.435431
```

关键 target R2：

```text
da_as_total_mw_primary_reserve          0.9735
metered_load_mw                         0.9626
total_gen                               0.9543
da_as_total_mw_synchronized_reserve     0.9387
total_lmp_da                            0.8605
total_losses                            0.6710
marginal_loss_price_da                  0.4941
congestion_price_rt                     0.3994
da_as_total_mw_thirty_minutes_reserve   0.3627
congestion_price_da                     0.2860
gross_actual_interchange_mw             0.2440
net_actual_interchange_mw              -0.7847
```

### 5. 与 DNN23 对比

maskloss：

```text
DNN23 linear window4:
  target_MSE = 0.333296
  cp_da      = 0.3010
  losses     = 0.6912
  mlp_da     = 0.4962
  30min      = 0.3749

DNN24 MLP window4:
  target_MSE = 0.336127
  cp_da      = 0.2992
  losses     = 0.6631
  mlp_da     = 0.4760
  30min      = 0.3285
```

maskloss 中 MLP encoder 不如线性 window4。

allloss：

```text
DNN23 linear window4:
  all_MSE = 0.436237
  gross   = 0.3044
  net     = -0.8312
  cp_da   = 0.2714
  losses  = 0.6862
  mlp_da  = 0.4675

DNN24 MLP window4:
  all_MSE = 0.435431
  gross   = 0.2440
  net     = -0.7847
  cp_da   = 0.2860
  losses  = 0.6710
  mlp_da  = 0.4941
```

allloss 中 MLP encoder 小幅刷新当前最好 all_MSE，并改善 `net_actual_interchange_mw`、`cp_da` 和 `marginal_loss_price_da`，但损害 `gross_actual_interchange_mw` 和 `total_losses`。

### 6. 结果分析

DNN24 说明：

```text
1. 更强时间编码对 allloss 有帮助；
2. 但对 maskloss 不稳定，说明 MLP 额外容量可能带来过拟合；
3. window4 的线性编码已经很强，不应轻易替换；
4. net_actual_interchange_mw 仍为负，但从 DNN23 的 -0.8312 改善到 -0.7847；
5. gross_actual_interchange_mw 从 DNN23 的 0.3044 降到 0.2440，说明 net/gross 的最佳结构可能不同。
```

### 7. 当前结论

当前推荐保留两条候选主线：

```text
maskloss主线:
  DNN21 window3 或 DNN23 linear window4

allloss主线:
  DNN24 temporal MLP window4
```

如果继续追求提升，下一步不建议盲目加深模型，而应测试：

```text
staged training:
  phase 1: 先训练稳定10个target，排除net/gross；
  phase 2: 再以较小学习率引入全部12个target微调。
```

这可能比从一开始就 allloss 更稳。
