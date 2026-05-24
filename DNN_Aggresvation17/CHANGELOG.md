# DNN_Aggresvation17 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目17要解决的问题

DNN16 将 General-General 聚合从固定相关性 GCN 改为 GAT 后，效果没有超过主线：

```text
DNN16 general_gat_maskloss:
  target_MSE = 0.339382
  all_MSE    = 0.548124

DNN16 general_gat_allloss_netgross16:
  all_MSE    = 0.451012
```

这说明 General-General 部分可能不适合继续增加动态注意力。原因是当前监督主要落在 confidential target 上，general 节点之间缺少直接监督，过强的 General-GAT 反而容易学习到不稳定的 general 表示。

因此 DNN17 回退到 DNN14/DNN15 的稳定主线：

```text
General-General: fixed weighted GCN
Confidential-Source: edge-specific gated attention
```

在此基础上只针对 confidential target 增加一个轻量残差专家，希望补偿某些 target 的非线性残差，而不破坏 General-General 的稳定传播。

### 2. 修改思路

原模型已经有 target-specific head，但 head 只读取每个 confidential 节点自身的最终表示。DNN17 增加 target-specific residual expert，使每个 confidential target 额外读取一份由其边先验加权得到的邻域上下文：

```text
context_c = sum_j normalized_prior(c,j) * h_j
```

然后将 confidential 节点表示和邻域上下文拼接：

```text
expert_input_c = [h_c || context_c]
```

每个 confidential target 使用一个独立小 MLP 产生残差：

```text
residual_c = Expert_c(expert_input_c)
```

最终输出为：

```text
y_hat_c = base_head_c(h_c) + scale_c * residual_c
```

其中 `scale_c` 为可学习缩放，初始值约为 `0.1`，避免一开始就让残差专家主导预测。

这个设计的直觉是：

```text
基础主干负责稳定的图传播；
残差专家只给每个 confidential target 补一小段 target-specific 修正。
```

### 3. 修改内容

修改文件：

```text
DNN_Aggresvation17/src/model.py
DNN_Aggresvation17/scripts/run_pipeline.py
DNN_Aggresvation17/scripts/run_experiments.py
DNN_Aggresvation17/scripts/compare_runs.py
DNN_Aggresvation17/configs/tuning/residual_expert_maskloss.yaml
DNN_Aggresvation17/configs/tuning/residual_expert_allloss_netgross16.yaml
```

`InferenceDrivenGNN` 新增配置：

```text
use_target_residual_expert: true
residual_init_scale: 0.1
```

新增模块：

```text
self.residual_experts
self.residual_scale_logits
```

新增前向计算：

```text
base_output = target_specific_head(h_conf)
residual = target_residual(h, h_conf, conf_prior)
output = base_output + residual
```

本轮只运行两组实验：

```text
residual_expert_maskloss
residual_expert_allloss_netgross16
```

其中：

```text
residual_expert_maskloss:
  top_k = 8
  排除 net_actual_interchange_mw 和 gross_actual_interchange_mw 的 loss

residual_expert_allloss_netgross16:
  top_k = 8
  net_actual_interchange_mw 补边到 16
  gross_actual_interchange_mw 补边到 16
  12 个 confidential target 全部进入 loss
```

输出目录：

```text
DNN_Aggresvation17/outputs_residual_expert/
```

### 4. 测试结果

两组实验已完成。汇总文件：

```text
DNN_Aggresvation17/outputs_residual_expert/comparison_summary.csv
```

#### 4.1 maskloss

```text
experiment: residual_expert_maskloss
target_MSE = 0.340547
all_MSE    = 0.549095
excluded_MSE = 1.591833
```

关键 target R2：

```text
metered_load_mw                         0.9702
da_as_total_mw_primary_reserve          0.9564
total_gen                               0.9348
total_lmp_da                            0.8963
da_as_total_mw_synchronized_reserve     0.8851
total_losses                            0.6568
congestion_price_rt                     0.4777
marginal_loss_price_da                  0.4752
da_as_total_mw_thirty_minutes_reserve   0.3751
congestion_price_da                     0.2508
gross_actual_interchange_mw            -0.5567  excluded
net_actual_interchange_mw              -1.9254  excluded
```

#### 4.2 allloss 加边

```text
experiment: residual_expert_allloss_netgross16
all_MSE = 0.464070
```

实际补边结果：

```text
net_actual_interchange_mw:   degree 9  -> 16
gross_actual_interchange_mw: degree 11 -> 17
```

关键 target R2：

```text
metered_load_mw                         0.9697
total_gen                               0.9588
da_as_total_mw_primary_reserve          0.9280
da_as_total_mw_synchronized_reserve     0.9268
total_lmp_da                            0.8961
total_losses                            0.6446
marginal_loss_price_da                  0.4679
congestion_price_rt                     0.4178
da_as_total_mw_thirty_minutes_reserve   0.3828
congestion_price_da                     0.2481
gross_actual_interchange_mw             0.0751
net_actual_interchange_mw              -0.9526
```

### 5. 与当前主线对比

#### 5.1 maskloss 对比

当前 maskloss 主线仍是 DNN14 `sharp_attention_maskloss`：

```text
DNN14 sharp_attention_maskloss:
  target_MSE = 0.331766
  all_MSE    = 0.541777
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926

DNN17 residual_expert_maskloss:
  target_MSE = 0.340547
  all_MSE    = 0.549095
  cp_da      = 0.2508
  cp_rt      = 0.4777
  losses     = 0.6568
  mlp_da     = 0.4752
  30min      = 0.3751
```

DNN17 没有超过 DNN14。除 `metered_load_mw`、`primary_reserve` 等本来已经很高的字段外，几个关键中等难度 target 都略有下降。

#### 5.2 allloss 对比

当前 allloss 较好的控制组仍是 DNN15 `target_extra_edges_allloss`：

```text
DNN15 target_extra_edges_allloss:
  all_MSE = 0.439477
  cp_da   = 0.2407
  cp_rt   = 0.4817
  losses  = 0.6443
  mlp_da  = 0.4624
  30min   = 0.3666

DNN17 residual_expert_allloss_netgross16:
  all_MSE = 0.464070
  cp_da   = 0.2481
  cp_rt   = 0.4178
  losses  = 0.6446
  mlp_da  = 0.4679
  30min   = 0.3828
```

DNN17 对 `cp_da`、`mlp_da`、`30min` 有小幅局部改善，但 `cp_rt` 明显下降，整体 all_MSE 也不如 DNN15 的 allloss 控制组。

### 6. 结果分析

#### 6.1 残差专家没有成为有效归纳偏置

DNN17 的残差专家增加了 target-specific 容量，但测试集没有提升，说明当前瓶颈不只是输出层表达能力不足。

更可能的原因是：

```text
1. 某些 target 的可推断信息主要由边选择和 source 信息决定；
2. 如果图里没有保留正确 source，残差专家无法凭空恢复信息；
3. 如果 source 信息已经充分，额外残差专家会增加方差，带来过拟合风险；
4. 当前数据量和监督 target 数量有限，target-specific expert 的自由度可能偏高。
```

#### 6.2 net/gross 仍然没有被解决

allloss 加边后：

```text
gross_actual_interchange_mw: 0.0751
net_actual_interchange_mw:  -0.9526
```

这说明仅增加残差输出专家不能解决这两个字段。之前的实验已经多次表明它们可能属于统计相关弱、时间对齐复杂、或者从当前 general 字段集合中不可稳定推断的目标。

#### 6.3 General-General 仍应保留 GCN

DNN16 证明 General-GAT 不优；DNN17 回退 GCN 后也没有靠残差专家超过 DNN14/DNN15。结合结果看，当前最稳的结构仍是：

```text
General-General: 全局相关系数权重 + fixed weighted GCN
Confidential-Source: edge-specific alpha + gated prior attention
```

后续如果继续改，不建议先动 General-General 主干，而应该围绕 confidential 的输入边选择、target-specific 先验、以及困难 target 的单独策略做。

### 7. 当前结论

DNN17 的结论是：

```text
1. target-specific residual expert 可以正常训练，但没有超过当前主线；
2. maskloss 主线仍保留 DNN14 sharp_attention_maskloss；
3. allloss 控制组仍保留 DNN15 target_extra_edges_allloss 或 net/gross=16 的加边方案；
4. 残差专家不是解决困难 confidential target 的关键；
5. 后续更值得测试的是边选择策略、target-specific source 子图、或者困难 target 的专门损失/分支。
```

因此 DNN17 不建议作为下一阶段主方案，只作为一次容量增强消融实验保留。
