# DNN_Aggresvation11 日志

## Modify by GPT5.5: 2026-05-23

### 1. 子项目11要解决的问题

子项目8、9、10 中 `alpha_general` 的变化很大：

```text
DNN8 hybrid_gated_loss_mask:
  NMI = 0.403370, Pearson = 0.177103

DNN9 original loss_mask:
  Pearson = 0.407593, NMI = 0.129961

DNN9 tuned top8:
  Pearson = 0.758161, NMI = 0.059847

DNN10 maskloss:
  Pearson = 0.815355, NMI = 0.035429
```

这说明 `alpha_general` 的含义并不稳定。一个合理怀疑是：General-General 在 DNN8-DNN10 中一直使用固定加权 GCN 聚合：

```text
m_i = sum_j A_ij h_j
```

其中 `A_ij` 由全局 `alpha_G` 融合相关性指标得到。由于这是所有 General-General 边共用的一套固定加权平均，模型可能倾向于选择最稳定、最线性的 Pearson，来获得平滑传播效果。这不一定说明 Pearson 真正解释了所有 General-General 隐性推断关系。

而 confidential 节点在 DNN10 中使用的是：

```text
dynamic attention + alpha prior + gate
```

所以 `alpha_confidential` 可以明显分化，并且仍然常常偏向 NMI 或 distance correlation。

### 2. 修改思路

DNN11 将 General-General 边也改成 attention 形式，测试 Pearson 断档是否来自固定 GCN 聚合。

保留 DNN10 的 confidential 公式不变，同时新增：

```text
General-General: gated log-prior attention
Confidential-Source: gated log-prior attention
```

### 3. General-General 新公式

General-General 仍然使用一组全局指标融合权重：

```text
alpha_G = softmax(beta_G / temperature)
```

对 general target `i` 和 general source `j`：

```text
p_G(i,j) = sum_m alpha_G,m * r(i,j,m)
prior_G(i,j) = log(p_G(i,j) + eps)
```

动态匹配项：

```text
d_G(i,j) = q_i^T k_j / sqrt(d)
```

门控项：

```text
lambda_G(i,j) = sigmoid(W_q q_i + W_k k_j + W_p prior_G(i,j) + b)
```

最终 attention 分数：

```text
score_G(i,j)
  = lambda_G(i,j) * d_G(i,j)
    + (1 - lambda_G(i,j)) * softplus(tau_G) * prior_G(i,j)
```

然后对 general source 做 softmax：

```text
omega_G(i,j) = softmax_j(score_G(i,j))
```

聚合：

```text
m_i = sum_j omega_G(i,j) v_j
```

### 4. Confidential 公式

沿用 DNN10 的设计，不再让 raw relation vector 绕过 `alpha_c`：

```text
p_C(c,j) = sum_m alpha_c,m * r(c,j,m)
prior_C(c,j) = log(p_C(c,j) + eps)
d_C(c,j) = q_c^T k_j / sqrt(d)
score_C(c,j)
  = lambda_C(c,j) * d_C(c,j)
    + (1 - lambda_C(c,j)) * softplus(tau_C) * prior_C(c,j)
```

### 5. 实验设置

只跑两个实验：

```text
general_attention_gated_maskloss
general_attention_gated_allloss
```

共同设置：

```text
top_k = 8
attention_temperature = 2.0
edge_alpha_temperature = 0.35
alpha_init_std = 1.0
prior_scale_init = 1.0
gate_bias_init = -3.0
prior_log_eps = 1e-4
alpha_lr_multiplier = 12.0
epochs = 300
patience = 80
seed = 42
```

### 6. 待观察

重点看三件事：

```text
1. alpha_general 是否仍然 Pearson 断档第一；
2. target MSE 是否优于 DNN10 maskloss；
3. general attention 后是否让训练更不稳定。
```

如果 `alpha_general` 不再 Pearson 断档，同时 MSE 不变差，说明 DNN10 的 Pearson 断档确实很可能是固定 GCN 聚合造成的。反之，如果 General-General attention 后 `alpha_general` 仍偏 Pearson，说明在当前数据中 general 节点间传播确实更依赖线性相关。

---

## Modify by GPT5.5: 2026-05-23 03:34 CST

### 1. 实验完成

已运行：

```text
general_attention_gated_maskloss
general_attention_gated_allloss
```

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation11/scripts/run_experiments.py
```

### 2. 总体结果

| experiment | target MSE | all MSE | excluded MSE | congestion DA R2 | congestion RT R2 | total losses R2 | marginal loss DA R2 | 30min reserve R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| general_attention_gated_maskloss | 0.347546 | 0.554927 | 1.591833 | 0.258090 | 0.407888 | 0.692147 | 0.435968 | 0.391134 |
| general_attention_gated_allloss | 0.486622 | 0.486622 | - | 0.219235 | 0.440375 | 0.645898 | 0.449743 | 0.247781 |

与 DNN10 maskloss 对比：

```text
DNN10 maskloss target MSE = 0.346891
DNN11 maskloss target MSE = 0.347546
```

DNN11 略差于 DNN10，但差距很小。说明把 General-General 也改成 attention 并没有带来整体性能提升，但也没有明显崩掉。

### 3. alpha_general 变化

DNN11 maskloss 的 `alpha_general`：

| metric | alpha |
|---|---:|
| Pearson | 0.707868 |
| Spearman | 0.071533 |
| Kendall | 0.075356 |
| NMI | 0.066943 |
| distance correlation | 0.078300 |

DNN11 allloss 的 `alpha_general`：

| metric | alpha |
|---|---:|
| Pearson | 0.623311 |
| Spearman | 0.067427 |
| Kendall | 0.076428 |
| NMI | 0.152717 |
| distance correlation | 0.080117 |

与 DNN10 maskloss 对比：

```text
DNN10 maskloss Pearson = 0.815355
DNN11 maskloss Pearson = 0.707868
```

因此，General-General attention 确实削弱了 Pearson 断档程度，但没有改变 Pearson 第一的结论。

当前判断：

```text
Pearson 断档有一部分来自固定 GCN 加权平均；
但即使换成 General attention，General-General 传播仍然偏向 Pearson。
```

这说明在当前数据与任务中，general 节点之间的传播可能确实更依赖线性同步关系，而 confidential target 的推断关系更依赖 NMI、distance correlation 或目标特定指标。

### 4. alpha_confidential 仍然分化

DNN11 maskloss 下 `alpha_confidential` 仍然明显分化：

```text
total_gen:
  nmi = 0.611287

metered_load_mw:
  nmi = 0.660509

total_losses:
  nmi = 0.330935
  pearson = 0.267737

congestion_price_da:
  nmi = 0.287932
  kendall = 0.219623

congestion_price_rt:
  nmi = 0.740656

marginal_loss_price_da:
  distance_corr = 0.385333

total_lmp_da:
  nmi = 0.484036
  pearson = 0.344837

da_as_total_mw_primary_reserve:
  pearson = 0.703468

da_as_total_mw_synchronized_reserve:
  nmi = 0.648859
```

这个结果支持一个更细的解释：

```text
General-General 的边是在做一般字段表示传播；
Confidential-Source 的边是在做目标推断；
两类边的统计指标偏好本来就可能不同。
```

因此，`alpha_general` 偏 Pearson 与 `alpha_confidential` 偏 NMI 并不必然矛盾。前者偏向稳定同步传播，后者偏向目标相关的非线性/分布依赖。

### 5. 逐目标变化

DNN11 maskloss 相对 DNN10 maskloss：

```text
total_losses:
  0.6778 -> 0.6921, 提升

thirty_minutes_reserve:
  0.3655 -> 0.3911, 提升

total_lmp_da:
  0.7118 -> 0.8823, 明显提升

congestion_price_da:
  0.2706 -> 0.2581, 略降

congestion_price_rt:
  0.4489 -> 0.4079, 下降

total_gen:
  0.9084 -> 0.8910, 下降

synchronized_reserve:
  0.9338 -> 0.9169, 下降
```

所以 DNN11 不是整体更强，而是改变了 trade-off：对 `total_losses`、`total_lmp_da` 和 `thirty_minutes_reserve` 更好，对 price RT 和部分高 R2 目标更差。

### 6. 当前结论

DNN11 值得保留，因为它回答了一个重要问题：

```text
General-General 固定 GCN 会放大 Pearson 偏好；
但把 General-General 改成 attention 后，Pearson 仍然保持第一。
```

因此，DNN10 中 `alpha_general = Pearson 断档第一` 不能简单认为是 bug。更合理解释是：

```text
General-General 全局传播更适合线性相关；
Confidential target 推断更需要目标条件化的非线性相关指标；
所以 alpha_general 和 alpha_confidential 本来就可能不同。
```

从性能上看：

```text
DNN10 maskloss 仍然略优于 DNN11 maskloss；
DNN11 提供了更合理的结构对照和解释证据；
maskloss 仍明显优于 allloss。
```

当前推荐：

```text
主线性能模型：DNN10 maskloss
解释性结构对照：DNN11 maskloss
```

如果要继续优化 DNN11，不建议马上加更复杂结构。更合理的是调小 `attention_temperature` 或降低 `alpha_lr_multiplier`，因为 DNN11 的 General attention 可能让传播更灵活但也更容易波动，当前测试 loss 曲线中后期有明显震荡。
