# DNN_Aggresvation14 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目14要解决的问题

DNN13 的整体效果已经明显优于前面版本，但少数 confidential target 仍然明显低于高 R2 字段：

```text
congestion_price_da                   R2 = 0.2670
da_as_total_mw_thirty_minutes_reserve R2 = 0.3624
marginal_loss_price_da                R2 = 0.4362
congestion_price_rt                   R2 = 0.4674
```

不过这些低值不能一概视为模型失败。对比 DNN4 的 single-target probe：

```text
congestion_price_da:
  probe R2 = 0.0476
  DNN13 R2 = 0.2670
  说明该字段本身很难从 General 数据推断，DNN13 已经明显超过单目标 probe。

da_as_total_mw_thirty_minutes_reserve:
  probe R2 = 0.4569
  DNN13 R2 = 0.3624
  说明该字段是 DNN13 当前真正没有吃满的 target。

marginal_loss_price_da:
  probe R2 = 0.4407
  DNN13 R2 = 0.4362
  基本接近 probe。

congestion_price_rt:
  probe R2 = 0.4562
  DNN13 R2 = 0.4674
  已经略高于 probe。
```

因此 DNN14 不是继续盲目加结构，而是围绕三个问题做小规模调参验证：

```text
1. 弱 target 是否因为 top_k 不够导致重要 source 没进图；
2. 弱 target 是否因为 attention 太分散；
3. 弱 target 是否因为训练 loss 权重不足；
4. per-edge alpha/prior 是否需要更强约束。
```

### 2. 基础结构

DNN14 沿用 DNN13 的结构：

```text
General-General:
  fixed weighted GCN

Confidential-Source:
  directed edge-specific alpha
  gated log-prior attention

Loss:
  默认 maskloss，排除 net_actual_interchange_mw 和 gross_actual_interchange_mw
```

### 3. 新增训练能力

在 `src/train.py` 中新增：

```text
training.target_loss_weights
```

该配置只影响训练 loss，不影响测试时记录的未加权 MSE/R2。因此可以测试“给弱 target 更大训练权重”是否能改善它们的 R2，而不会改变最终评价口径。

### 4. 实验组

本轮计划运行五组 maskloss 实验：

```text
baseline_dnn13_maskloss
top12_maskloss
sharp_attention_maskloss
strong_prior_alpha_maskloss
weak_target_weighted_maskloss
```

#### 4.1 baseline_dnn13_maskloss

复现 DNN13 maskloss，用于 DNN14 同目录对照。

#### 4.2 top12_maskloss

将：

```text
top_k = 8
```

改为：

```text
top_k = 12
```

用于测试弱 target 是否受限于邻居不足。

#### 4.3 sharp_attention_maskloss

将：

```text
attention_temperature = 2.0
```

改为：

```text
attention_temperature = 1.0
```

用于测试弱 target 是否因为 attention 太分散。

#### 4.4 strong_prior_alpha_maskloss

将：

```text
edge_alpha_temperature = 0.35
gate_bias_init = -3.0
alpha_lr_multiplier = 12.0
```

改为：

```text
edge_alpha_temperature = 0.25
gate_bias_init = -3.5
alpha_lr_multiplier = 16.0
```

用于测试是否需要更强的关系先验和更尖锐的 edge alpha。

#### 4.5 weak_target_weighted_maskloss

保留 DNN13 结构，但对弱 target 加权：

```text
congestion_price_da: 2.5
congestion_price_rt: 1.8
marginal_loss_price_da: 1.6
da_as_total_mw_thirty_minutes_reserve: 2.0
```

用于测试低 R2 是否只是因为这些 target 在平均 MSE 中权重不足。

### 5. 待观察

重点看：

```text
1. target MSE 是否能超过 DNN13 的 0.335532；
2. congestion_price_da 是否能继续提升；
3. thirty_minutes_reserve 是否能接近 probe R2 = 0.4569；
4. marginal_loss_price_da / congestion_price_rt 是否会被 trade-off 损害；
5. attention entropy 是否下降。
```

### 6. 测试结果

五组实验已完成，输出目录为：

```text
DNN_Aggresvation14/outputs/
```

汇总文件为：

```text
DNN_Aggresvation14/outputs/comparison_summary.csv
```

结果如下：

```text
experiment                    target_MSE   all_MSE    cp_da   cp_rt   losses  mlp_da  30min_reserve
baseline_dnn13_maskloss       0.341029     0.549496   0.2687  0.4470  0.6303  0.4424  0.3899
top12_maskloss                0.342804     0.550975   0.2172  0.4727  0.6574  0.4670  0.3775
sharp_attention_maskloss      0.331766     0.541777   0.2678  0.4796  0.6768  0.4832  0.3926
strong_prior_alpha_maskloss   0.339687     0.548378   0.2817  0.4457  0.6409  0.4318  0.3876
weak_target_weighted_maskloss 0.344660     0.552522   0.2592  0.4917  0.6252  0.4155  0.3795
```

其中：

```text
cp_da        = congestion_price_da
cp_rt        = congestion_price_rt
losses       = total_losses
mlp_da       = marginal_loss_price_da
30min_reserve= da_as_total_mw_thirty_minutes_reserve
```

### 7. 结果分析

#### 7.1 最好的整体配置

本轮最好的整体配置是：

```text
sharp_attention_maskloss
```

它将 attention temperature 从 2.0 降到 1.0，使 softmax 注意力更尖锐。该配置取得：

```text
target_MSE = 0.331766
all_MSE    = 0.541777
```

相比 DNN14 baseline：

```text
target_MSE: 0.341029 -> 0.331766
all_MSE:    0.549496 -> 0.541777
```

说明 DNN13/14 的确存在一定程度的注意力过分平均问题。让注意力更集中后，整体误差下降，并且多个中等难度 target 改善明显：

```text
congestion_price_rt:           0.4470 -> 0.4796
total_losses:                  0.6303 -> 0.6768
marginal_loss_price_da:        0.4424 -> 0.4832
thirty_minutes_reserve:        0.3899 -> 0.3926
```

不过：

```text
congestion_price_da:           0.2687 -> 0.2678
```

几乎没有改善，说明该 target 的瓶颈不是简单 attention 分散。

#### 7.2 top_k 不是主要瓶颈

将 top_k 从 8 增大到 12 后：

```text
target_MSE: 0.341029 -> 0.342804
```

整体没有改善。部分 target 提升：

```text
congestion_price_rt:    0.4470 -> 0.4727
total_losses:           0.6303 -> 0.6574
marginal_loss_price_da: 0.4424 -> 0.4670
```

但也出现明显损害：

```text
congestion_price_da:    0.2687 -> 0.2172
thirty_minutes_reserve: 0.3899 -> 0.3775
```

这说明邻居更多并不一定更好。对这类小图任务，top_k 增大可能引入额外噪声邻居，使某些本来弱的 target 更难从注意力中筛出有效 source。

#### 7.3 更强 alpha/prior 只能小幅改善 congestion_price_da

`strong_prior_alpha_maskloss` 中降低 edge alpha temperature、增大 alpha 学习率并让 gate 更偏向 prior。结果：

```text
congestion_price_da: 0.2687 -> 0.2817
```

这是本轮对 congestion_price_da 最好的结果，但整体收益不大：

```text
target_MSE: 0.341029 -> 0.339687
```

同时部分字段下降：

```text
marginal_loss_price_da: 0.4424 -> 0.4318
congestion_price_rt:    0.4470 -> 0.4457
```

说明更强先验能帮助最难的 price target 一点点，但会牺牲另一些 target 的动态匹配能力。

#### 7.4 target loss 加权有明显 trade-off

`weak_target_weighted_maskloss` 对几个弱 target 增大训练权重后，效果并不稳定：

```text
congestion_price_rt: 0.4470 -> 0.4917
```

这是本轮 congestion_price_rt 的最高值。但其他弱 target 反而下降：

```text
congestion_price_da:    0.2687 -> 0.2592
marginal_loss_price_da: 0.4424 -> 0.4155
thirty_minutes_reserve: 0.3899 -> 0.3795
```

因此问题不是简单的“弱 target 在平均 loss 中权重不够”。加权会把模型容量向部分 target 拉过去，但不能保证所有弱 target 同时提升。

### 8. 注意力与 alpha 诊断

`sharp_attention_maskloss` 的 normalized attention entropy 相比 baseline 明显下降：

```text
target                          baseline_entropy_norm  sharp_entropy_norm
congestion_price_da              0.9938                 0.9775
congestion_price_rt              0.9534                 0.9281
total_losses                     0.9775                 0.9501
marginal_loss_price_da           0.9820                 0.9520
thirty_minutes_reserve           0.9888                 0.9590
```

这与 R2 改善方向基本一致：注意力更集中后，中等 target 受益明显。

但即便 sharp attention 后，很多 target 的 entropy 仍接近 1，说明注意力还没有变成强选择机制，而只是从“非常平均”变成“稍微更有倾向”。

edge alpha 的现象也很重要。以 `sharp_attention_maskloss` 为例：

```text
congestion_price_da:
  alpha: pearson 0.18, spearman 0.16, kendall 0.16, nmi 0.25, distance_corr 0.25
  有效邻居数: 8

congestion_price_rt:
  alpha: pearson 0.18, spearman 0.18, kendall 0.18, nmi 0.25, distance_corr 0.21
  有效邻居数: 11

total_losses:
  alpha: pearson 0.18, spearman 0.16, kendall 0.16, nmi 0.34, distance_corr 0.16
  有效邻居数: 19

marginal_loss_price_da:
  alpha: pearson 0.20, spearman 0.18, kendall 0.20, nmi 0.23, distance_corr 0.18
  有效邻居数: 21

thirty_minutes_reserve:
  alpha: pearson 0.17, spearman 0.20, kendall 0.19, nmi 0.26, distance_corr 0.17
  有效邻居数: 18
```

因此，DNN13/14 的 per-edge alpha 已经能分化，但对很多 target 来说并不是极端单一指标主导。它更像是在“关系先验”层面提供温和偏置，真正决定预测的仍然是动态 attention 和后续 readout。

### 9. 为什么个别 confidential 仍然差

综合 DNN13 与 DNN14，本轮判断如下。

第一，`congestion_price_da` 的低 R2 不完全是模型问题。DNN4 single-target probe 中它只有：

```text
probe R2 = 0.0476
```

DNN13/14 可以做到约：

```text
0.267 ~ 0.282
```

这已经大幅超过单目标 DNN probe。也就是说，它本身从当前 General 字段中可推断性弱，图结构和 confidential 间传播提供了一部分额外信息，但很难靠小调参继续大幅提升。

第二，`da_as_total_mw_thirty_minutes_reserve` 是真正值得继续追的问题。它的 probe 是：

```text
probe R2 = 0.4569
```

DNN14 最好只有：

```text
0.3926
```

这说明当前多目标 GNN 没有吃满该 target 的可推断信息。可能原因是：该字段与 reserve 相关字段的关系比较局部、专门，而多目标共享模型更容易优先优化 total_gen、metered_load、reserve primary/synchronized 等高信号目标。

第三，`marginal_loss_price_da` 和 `congestion_price_rt` 基本已经接近或超过 single-target probe。DNN14 sharp attention 下：

```text
marginal_loss_price_da: 0.4832
congestion_price_rt:    0.4796
```

它们不再是主要短板。

### 10. 当前建议

从本轮实验看，继续无方向地调参意义不大。比较合理的下一步是：

```text
1. 保留 DNN13/DNN14 的 per-edge confidential alpha + gated prior attention 主线；
2. 默认采用 sharp_attention_maskloss 的 attention_temperature = 1.0；
3. 不建议把 top_k 默认增大到 12；
4. 不建议把 weak target loss weighting 作为默认方案；
5. 如果继续做子项目15，重点应该针对 da_as_total_mw_thirty_minutes_reserve 做 target-specific branch 或 auxiliary single-target head，而不是继续全局加复杂度。
```

也就是说，DNN14 证明：

```text
attention sharpening 有效；
top_k 增大不是主要解；
loss 加权不是稳定解；
congestion_price_da 可能接近当前数据可推断上限；
thirty_minutes_reserve 才是下一轮最值得单独处理的 target。
```
