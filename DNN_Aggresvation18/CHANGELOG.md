# DNN_Aggresvation18 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目18要解决的问题

DNN17 在稳定主干上加入 target-specific residual expert，但没有超过 DNN14/DNN15 主线。这说明当前瓶颈不只是输出层表达能力不足，而更可能来自多目标训练时不同 confidential target 的难度差异。

子项目18测试第二个方向：

```text
Target-specific loss weighting / uncertainty weighting
```

核心问题是：12 个 confidential target 的可推断难度差别很大，如果所有 target 用相同 MSE 权重训练，容易出现两类问题：

```text
1. 高噪声或难推断 target 的大误差支配共享表示；
2. 易推断 target 被快速拟合后仍占据训练目标，弱 target 改善有限。
```

因此 DNN18 引入可学习的不确定性损失权重，让模型自己估计每个 target 的训练权重。

### 2. 修改思路

对每个 confidential target 学习一个参数：

```text
log_var_c
```

对应回归不确定性。训练损失从普通平均 MSE：

```text
L = mean_c MSE_c
```

改为 Kendall 式 uncertainty weighting：

```text
L_c = 0.5 * exp(-log_var_c) * MSE_c + 0.5 * log_var_c
```

其中：

```text
loss_weight_c = exp(-log_var_c)
```

如果某个 target 的残差更像高噪声任务，模型可以提高 `log_var_c`，从而降低该 target 的梯度权重；如果某个 target 能被稳定拟合，模型会降低 `log_var_c`，提高其精度权重。

本轮没有改模型结构，仍使用当前稳定主干：

```text
General-General: fixed weighted GCN
Confidential-Source: edge-specific gated attention
```

### 3. 修改内容

修改文件：

```text
DNN_Aggresvation18/src/train.py
DNN_Aggresvation18/scripts/run_pipeline.py
DNN_Aggresvation18/scripts/run_experiments.py
DNN_Aggresvation18/scripts/compare_runs.py
DNN_Aggresvation18/configs/tuning/uncertainty_maskloss.yaml
DNN_Aggresvation18/configs/tuning/uncertainty_allloss_netgross16.yaml
```

新增训练配置：

```text
target_loss_weighting: uncertainty
uncertainty_init_log_var: 0.0
uncertainty_log_var_clamp: [-3.0, 3.0]
uncertainty_lr_multiplier: 5.0
```

新增输出：

```text
training/target_loss_weights.csv
```

该文件记录每个 confidential target 的：

```text
final_log_var
final_loss_weight = exp(-final_log_var)
```

### 4. 实验设置

本轮运行两组实验：

```text
uncertainty_maskloss
uncertainty_allloss_netgross16
```

其中：

```text
uncertainty_maskloss:
  top_k = 8
  排除 net_actual_interchange_mw 和 gross_actual_interchange_mw 的 loss

uncertainty_allloss_netgross16:
  top_k = 8
  net_actual_interchange_mw 补边到 16
  gross_actual_interchange_mw 补边到 16
  12 个 confidential target 全部进入 loss
```

输出目录：

```text
DNN_Aggresvation18/outputs_uncertainty/
```

### 5. 测试结果

#### 5.1 maskloss

```text
experiment: uncertainty_maskloss
target_MSE = 0.336464
all_MSE    = 0.545692
excluded_MSE = 1.591833
```

关键 target R2：

```text
metered_load_mw                         0.9802
da_as_total_mw_primary_reserve          0.9678
total_gen                               0.9567
da_as_total_mw_synchronized_reserve     0.9254
total_lmp_da                            0.9153
total_losses                            0.6156
congestion_price_rt                     0.4779
marginal_loss_price_da                  0.4753
da_as_total_mw_thirty_minutes_reserve   0.4071
congestion_price_da                     0.2398
gross_actual_interchange_mw            -0.5567  excluded
net_actual_interchange_mw              -1.9254  excluded
```

#### 5.2 allloss 加边

```text
experiment: uncertainty_allloss_netgross16
all_MSE = 0.439875
```

实际补边结果：

```text
net_actual_interchange_mw:   degree 9  -> 16
gross_actual_interchange_mw: degree 11 -> 17
```

关键 target R2：

```text
metered_load_mw                         0.9795
total_gen                               0.9723
da_as_total_mw_primary_reserve          0.9680
da_as_total_mw_synchronized_reserve     0.9407
total_lmp_da                            0.9046
total_losses                            0.6503
congestion_price_rt                     0.4729
marginal_loss_price_da                  0.4639
da_as_total_mw_thirty_minutes_reserve   0.3617
congestion_price_da                     0.2279
gross_actual_interchange_mw             0.1603
net_actual_interchange_mw              -0.8033
```

### 6. 与主线对比

当前 maskloss 最优仍是 DNN14 `sharp_attention_maskloss`：

```text
DNN14 sharp_attention_maskloss:
  target_MSE = 0.331766
  all_MSE    = 0.541777
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926

DNN18 uncertainty_maskloss:
  target_MSE = 0.336464
  all_MSE    = 0.545692
  cp_da      = 0.2398
  cp_rt      = 0.4779
  losses     = 0.6156
  mlp_da     = 0.4753
  30min      = 0.4071
```

DNN18 的 maskloss 比 DNN17 好，但仍未超过 DNN14。它改善了高 R2 target 和 `thirty_minutes_reserve`，但损害了 `congestion_price_da` 和 `total_losses`。

当前 allloss 较强基线为 DNN15 `target_extra_edges_allloss`：

```text
DNN15 target_extra_edges_allloss:
  all_MSE = 0.439477

DNN18 uncertainty_allloss_netgross16:
  all_MSE = 0.439875
```

DNN18 的 allloss 与 DNN15 几乎持平，但略差。局部看：

```text
gross_actual_interchange_mw: 0.1603
net_actual_interchange_mw:  -0.8033
```

说明 uncertainty weighting 对 net/gross 有一点帮助，但没有根本解决。

### 7. 权重分析

maskloss 中学习到的 loss weights：

```text
total_gen                               20.0855
metered_load_mw                         20.0855
total_lmp_da                            20.0855
da_as_total_mw_primary_reserve          18.0621
da_as_total_mw_synchronized_reserve     12.7064
total_losses                             7.1056
da_as_total_mw_thirty_minutes_reserve    5.8305
marginal_loss_price_da                   4.7046
congestion_price_rt                      2.2349
congestion_price_da                      1.6089
```

allloss 中学习到的 loss weights：

```text
total_gen                               20.0855
metered_load_mw                         20.0855
total_lmp_da                            20.0855
da_as_total_mw_primary_reserve          17.6232
da_as_total_mw_synchronized_reserve     12.2533
total_losses                             6.0709
da_as_total_mw_thirty_minutes_reserve    5.4461
marginal_loss_price_da                   4.4641
gross_actual_interchange_mw              3.9125
net_actual_interchange_mw                3.6159
congestion_price_rt                      2.0836
congestion_price_da                      1.5576
```

这个结果非常关键：uncertainty weighting 并没有重点照顾弱 target，而是把易拟合、低残差 target 的权重推到最高。这符合概率意义上的 precision weighting，但不符合当前实验目标。

### 8. 结论

DNN18 的结论是：

```text
1. uncertainty weighting 能正常训练，且权重确实发生了明显分化；
2. 它能改善部分高可推断字段和 allloss 的 net/gross；
3. 但它会降低困难 target 的训练权重，导致 congestion_price_da、total_losses 等短板没有改善；
4. 因此 uncertainty weighting 不适合作为当前主线；
5. 下一步应该反过来测试 difficulty-aware static weighting，主动提高弱 target 的训练权重。
```

因此 DNN18 保留为 loss weighting 消融实验，不建议替代 DNN14/DNN15 主线。
