# DNN_Aggresvation27 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目27要解决的问题

DNN24 证明 temporal MLP encoder 在 `window_size=4` 的 all-loss 口径下有一定收益：

```text
DNN24 temporal_mlp_window4_allloss_netgross16:
  all_MSE = 0.435431
```

但 DNN24 的 confidential attention 仍然是单头注意力。单头注意力在每个 target 上只能形成一套 source 选择分布，可能不足以同时捕捉：

```text
1. 价格类字段的非线性关系；
2. losses / lmp 类字段的局部相关关系；
3. net/gross interchange 这种困难字段的弱而分散的 source 信号。
```

因此 DNN27 在 DNN24 的 temporal MLP window4 基础上增加多头 confidential attention。实验只保留两组：

```text
multihead_mlp_window4_maskloss
multihead_staged_mlp_window4_allloss_netgross16
```

其中第二组引入 staged training，用于验证多头 attention 和 DNN25 的 staged 思路能否叠加。

### 2. 修改思路

原 confidential attention 对每个目标 \(c\) 和 source \(j\) 计算单个动态匹配分数：

```text
dynamic_score(c,j) = q_c^T k_j / sqrt(d)
```

DNN27 改为 \(H=4\) 个 attention head。每个 head 独立计算：

```text
dynamic_score(c,h,j) = q_{c,h}^T k_{j,h} / sqrt(d_h)
```

每个 head 仍然使用同一个关系先验 `prior(c,j)`，但 gate 会根据不同 head 的 query/key 产生不同融合比例：

```text
score(c,h,j) =
  gate(c,h,j) * dynamic_score(c,h,j)
  + (1 - gate(c,h,j)) * scale * log(prior(c,j))
```

然后每个 head 独立 softmax 聚合 value，最后拼接所有 head 的消息并投影回 hidden dimension：

```text
message_c = W_o concat(message_{c,1}, ..., message_{c,H})
```

本轮配置：

```text
attention_heads = 4
attention_dim   = 32
hidden_dim       = 64
window_size      = 4
input_encoder    = mlp
```

### 3. 实现内容

修改文件：

```text
DNN_Aggresvation27/src/model.py
DNN_Aggresvation27/scripts/run_pipeline.py
DNN_Aggresvation27/scripts/run_experiments.py
DNN_Aggresvation27/configs/tuning/multihead_mlp_window4_maskloss.yaml
DNN_Aggresvation27/configs/tuning/multihead_staged_mlp_window4_allloss_netgross16.yaml
```

新增模型配置项：

```text
model.attention_heads
```

默认值为 1，因此旧配置仍保持单头兼容。

### 4. 实验设置

#### 4.1 maskloss

```text
experiment = multihead_mlp_window4_maskloss
top_k = 8
window_size = 4
input_encoder = mlp
attention_heads = 4
loss排除:
  net_actual_interchange_mw
  gross_actual_interchange_mw
```

#### 4.2 staged all-loss

```text
experiment = multihead_staged_mlp_window4_allloss_netgross16
top_k = 8
window_size = 4
input_encoder = mlp
attention_heads = 4
net_actual_interchange_mw   补边到16
gross_actual_interchange_mw 补边到16
phase1: 排除 net/gross
phase2: 全部12个confidential target
```

### 5. 待测试

重点比较：

```text
DNN24 temporal_mlp_window4_maskloss:
  target_MSE = 0.336127

DNN24 temporal_mlp_window4_allloss_netgross16:
  all_MSE = 0.435431

DNN25 staged_linear_window4_allloss_netgross16:
  all_MSE = 0.431317
```

需要观察多头 attention 是否能：

```text
1. 改善 DNN24 maskloss 的不稳定问题；
2. 在 staged all-loss 下超过 DNN25 staged linear；
3. 改善 net/gross，而不明显损害 price/loss 类 target。
```

### 6. 测试结果

两组实验已完成，汇总文件：

```text
DNN_Aggresvation27/outputs_multihead/comparison_summary.csv
```

#### 6.1 multihead_mlp_window4_maskloss

```text
best_test_loss_targets = 0.328398
final_test_mse_all     = 0.539185
excluded_MSE           = 1.593118
```

关键 target R2：

```text
metered_load_mw                         0.9675
da_as_total_mw_primary_reserve          0.9663
total_gen                               0.9389
da_as_total_mw_synchronized_reserve     0.9314
total_lmp_da                            0.8552
total_losses                            0.7020
marginal_loss_price_da                  0.4437
da_as_total_mw_thirty_minutes_reserve   0.4411
congestion_price_rt                     0.4378
congestion_price_da                     0.3010
gross_actual_interchange_mw            -0.5551  excluded
net_actual_interchange_mw              -1.9485  excluded
```

对比 DNN24 temporal MLP maskloss：

```text
DNN24:
  target_MSE = 0.336127
  cp_da      = 0.2992
  cp_rt      = 0.4749
  losses     = 0.6631
  mlp_da     = 0.4760
  30min      = 0.3285

DNN27 multihead:
  target_MSE = 0.328398
  cp_da      = 0.3010
  cp_rt      = 0.4378
  losses     = 0.7020
  mlp_da     = 0.4437
  30min      = 0.4411
```

#### 6.2 multihead_staged_mlp_window4_allloss_netgross16

```text
best_test_loss_targets = 0.465825
final_test_mse_all     = 0.465825
```

关键 target R2：

```text
da_as_total_mw_primary_reserve          0.9680
metered_load_mw                         0.9652
total_gen                               0.9494
da_as_total_mw_synchronized_reserve     0.9145
total_lmp_da                            0.8787
total_losses                            0.6921
da_as_total_mw_thirty_minutes_reserve   0.4265
congestion_price_rt                     0.4153
marginal_loss_price_da                  0.3747
congestion_price_da                     0.2727
gross_actual_interchange_mw             0.1484
net_actual_interchange_mw              -1.1603
```

对比当前 all-loss 主线：

```text
DNN24 temporal MLP all-loss:
  all_MSE = 0.435431
  gross   = 0.2440
  net     = -0.7847

DNN25 staged linear all-loss:
  all_MSE = 0.431317
  gross   = 0.1835
  net     = -0.6866

DNN27 multihead staged MLP all-loss:
  all_MSE = 0.465825
  gross   = 0.1484
  net     = -1.1603
```

### 7. 结果分析

#### 7.1 多头 attention 对 maskloss 是正向结果

DNN27 maskloss 是目前 maskloss 口径下非常强的一组：

```text
DNN14 sharp maskloss:        target_MSE = 0.331766
DNN21 window3 maskloss:      target_MSE = 0.331438
DNN23 window4 linear:        target_MSE = 0.333296
DNN24 window4 MLP:           target_MSE = 0.336127
DNN27 multihead window4 MLP: target_MSE = 0.328398
```

这说明在排除 net/gross 后，多头 confidential attention 确实能提供更丰富的 source 选择能力。尤其：

```text
total_losses: 0.6631 -> 0.7020
30min:        0.3285 -> 0.4411
cp_da:        0.2992 -> 0.3010
```

其中 `da_as_total_mw_thirty_minutes_reserve` 的提升非常明显，说明它可能需要多个 source 子模式共同解释，单头注意力容易把这些模式压成一套平均选择。

但 trade-off 也存在：

```text
cp_rt:  0.4749 -> 0.4378
mlp_da: 0.4760 -> 0.4437
```

所以 DNN27 不是所有 target 同时提升，而是把收益集中到了 losses 和 30min。

#### 7.2 多头 + staged MLP 的 all-loss 明显失败

all-loss 组没有继承 maskloss 的收益，反而明显退化：

```text
DNN27 all_MSE = 0.465825
```

主要问题：

```text
1. net_actual_interchange_mw 从 DNN25 的 -0.6866 退到 -1.1603；
2. gross_actual_interchange_mw 也低于 DNN24/DNN25；
3. marginal_loss_price_da 下降到 0.3747；
4. staged + MLP + multihead 三者叠加后模型容量偏大，phase2 很难稳定吸收 net/gross。
```

这和 DNN25 中 staged MLP 失败的现象一致：staged training 更适合 linear temporal encoder，而不是更高容量 MLP encoder。DNN27 进一步说明，如果在 MLP encoder 上再叠加多头 attention，all-loss 的困难 target 会更不稳定。

### 8. 当前结论

DNN27 的结论是：

```text
1. multi-head confidential attention 是 maskloss 方向的有效改进；
2. 当前最强 maskloss 候选可以更新为：
   DNN27 multihead_mlp_window4_maskloss
3. multi-head attention 不适合直接叠加 staged MLP all-loss；
4. all-loss 主线仍应保留：
   DNN25 staged_linear_window4_allloss_netgross16
5. 如果后续继续尝试 all-loss 多头，应回到 linear input encoder，而不是 MLP encoder。
```

下一步更合理的实验不是继续增大 head 数，而是：

```text
maskloss:
  以 DNN27 multihead MLP 作为强候选主线；

allloss:
  测试 staged linear + multihead attention，
  只把 DNN25 的 single-head confidential attention 换成 multi-head，
  不再叠加 MLP encoder。
```

## Modify by GPT5.5: 2026-05-24 追加不分阶段all-loss实验

### 9. 追加实验动机

上一组 `multihead_staged_mlp_window4_allloss_netgross16` 明显退化，但退化可能来自两个因素：

```text
1. multi-head attention 本身不适合 all-loss；
2. staged training 与 MLP encoder + multi-head attention 叠加后不稳定。
```

为了拆开这两个因素，本轮在同一个 DNN27 文件夹下追加一组不分阶段 all-loss：

```text
multihead_mlp_window4_allloss_netgross16
```

该组保持：

```text
input_encoder = mlp
window_size = 4
attention_heads = 4
```

但取消 staged training，从第一个 epoch 开始直接训练全部 12 个 confidential target。

### 10. 补边确认

配置中已显式设置：

```text
target_top_k_overrides:
  net_actual_interchange_mw: 16
  gross_actual_interchange_mw: 16
```

上一组 staged all-loss 的实际构图结果为：

```text
net_actual_interchange_mw:   degree 9  -> 16
gross_actual_interchange_mw: degree 11 -> 17
```

因此本轮不分阶段 all-loss 也会使用同样的 net/gross 补边策略。

### 11. 不分阶段 all-loss 测试结果

实验已完成：

```text
experiment = multihead_mlp_window4_allloss_netgross16
run_dir    = DNN_Aggresvation27/outputs_multihead/run_20260524_151021_multihead_mlp_window4_allloss_netgross16
```

实际补边结果：

```text
net_actual_interchange_mw:   degree 9  -> 16 (+7)
gross_actual_interchange_mw: degree 11 -> 17 (+6)
```

总体结果：

```text
all_MSE = 0.458986
```

关键 target R2：

```text
da_as_total_mw_primary_reserve          0.9695
metered_load_mw                         0.9679
total_gen                               0.9475
da_as_total_mw_synchronized_reserve     0.9362
total_lmp_da                            0.8804
total_losses                            0.7016
marginal_loss_price_da                  0.4875
da_as_total_mw_thirty_minutes_reserve   0.4336
congestion_price_rt                     0.3834
congestion_price_da                     0.2222
gross_actual_interchange_mw             0.1698
net_actual_interchange_mw              -1.0168
```

### 12. 追加结果分析

不分阶段 all-loss 比 staged MLP 多头稍好：

```text
multihead staged MLP all-loss:
  all_MSE = 0.465825
  gross   = 0.1484
  net     = -1.1603

multihead non-staged MLP all-loss:
  all_MSE = 0.458986
  gross   = 0.1698
  net     = -1.0168
```

这说明上一组退化不完全来自 staged training；即使不分阶段，多头 + MLP encoder 在 all-loss 下仍然明显弱于 DNN24/DNN25：

```text
DNN24 temporal MLP all-loss:
  all_MSE = 0.435431
  gross   = 0.2440
  net     = -0.7847

DNN25 staged linear all-loss:
  all_MSE = 0.431317
  gross   = 0.1835
  net     = -0.6866

DNN27 multihead non-staged MLP all-loss:
  all_MSE = 0.458986
  gross   = 0.1698
  net     = -1.0168
```

局部看，多头 all-loss 仍保留了一些 maskloss 里的收益：

```text
total_losses = 0.7016
30min        = 0.4336
mlp_da       = 0.4875
```

但 price 类和 interchange 类明显受损：

```text
cp_da = 0.2222
cp_rt = 0.3834
net   = -1.0168
```

因此当前判断更清晰：

```text
1. multi-head attention 对 maskloss 是有效方向；
2. multi-head + MLP encoder 不适合 all-loss；
3. all-loss 的瓶颈不是 staged 与否，而是 net/gross 进入训练后，多头 MLP 容量更容易把共享表示拉向不稳定解；
4. 如果继续测试 all-loss 多头，应回到 DNN25 的 linear temporal input，而不是继续使用 MLP encoder。
```
