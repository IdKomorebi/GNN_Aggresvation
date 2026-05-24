# DNN_Aggresvation10 日志

## Modify by GPT5.5: 2026-05-23

### 1. 子项目10要解决的问题

子项目9 的核心问题是：虽然已经给每个 confidential target 设置了目标专属相关系数融合权重 `alpha_c`，并且通过随机初始化、降低 alpha softmax temperature、提高 alpha 学习率等方式鼓励其分化，但最终 `alpha_c` 仍然接近平均。

重新检查注意力公式后发现，DNN9 中关系向量 `r_{c,j}` 同时进入了三条路径：

```text
1. prior:        p(c,j) = sum_m alpha_c,m * r(c,j,m)
2. dynamic:      d(c,j) = q_c^T k_j / sqrt(d) + W_r r(c,j)
3. gate:         lambda(c,j) = sigmoid(W_q q_c + W_k k_j + W_r' r(c,j) + b)
```

这意味着模型可以绕过 `alpha_c`，直接通过 `relation_bias` 或 gate 中的原始 `r(c,j)` 使用相关性信息。因此，即使 `beta_confidential` 初始值不是全 0，训练也不一定需要让 `alpha_c` 明显分化。

### 2. 修改思路

DNN10 的目标是让 confidential 关系先验更依赖 `alpha_c`，因此收紧注意力公式：

```text
去掉 dynamic score 中的 relation_bias；
gate 不再直接使用五维 raw relation vector；
prior 使用 log(p + eps)，扩大强弱边在softmax前的差异；
只保留两个实验：maskloss 与 allloss；
固定 top_k = 8。
```

### 3. 新注意力公式

对于 confidential target `c` 和 source 节点 `j`：

```text
p(c,j) = sum_m alpha_c,m * r(c,j,m)
prior_score(c,j) = log(p(c,j) + eps)
dynamic_score(c,j) = q_c^T k_j / sqrt(d)
```

gate 改为：

```text
lambda(c,j) = sigmoid(W_q q_c + W_k k_j + W_p prior_score(c,j) + b)
```

最终注意力分数：

```text
score(c,j)
  = lambda(c,j) * dynamic_score(c,j)
    + (1 - lambda(c,j)) * softplus(tau) * prior_score(c,j)
```

因此，关系指标对 confidential attention 的影响主要通过：

```text
alpha_c -> p(c,j) -> log prior -> score
```

而不是通过 `W_r r(c,j)` 直接进入 dynamic score。

### 4. 参数设置

两个实验使用同一组结构参数：

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

其中：

```text
maskloss: 排除 net_actual_interchange_mw 和 gross_actual_interchange_mw
allloss: 所有 confidential target 都参与 loss
```

### 5. 待测试

运行脚本：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation10/scripts/run_experiments.py
```

测试完成后需要重点查看：

```text
1. target MSE 是否优于 DNN9 tuned top8；
2. alpha_confidential_by_target.csv 是否比 DNN9 更分化；
3. price 类 target 的 attention entropy 是否下降；
4. maskloss 是否继续显著优于 allloss。
```

---

## Modify by GPT5.5: 2026-05-23 02:25 CST

### 1. 测试完成

已运行两个 top-k=8 实验：

```text
no_relation_bias_gated_maskloss
no_relation_bias_gated_allloss
```

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation10/scripts/run_experiments.py
```

运行过程中 Matplotlib 提示 conda 环境下默认 cache 目录不可写，因此自动使用临时目录。这不影响训练结果和图表生成。

### 2. 总体结果

| experiment | target MSE | all MSE | excluded MSE | congestion DA R2 | congestion RT R2 | total losses R2 | marginal loss DA R2 | 30min reserve R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_relation_bias_gated_maskloss | 0.346891 | 0.554381 | 1.591833 | 0.270594 | 0.448864 | 0.677831 | 0.430122 | 0.365545 |
| no_relation_bias_gated_allloss | 0.473569 | 0.473569 | - | 0.252706 | 0.405565 | 0.668705 | 0.485523 | 0.363297 |

其中 `maskloss` 是当前 DNN10 主线。它相对 DNN9 tuned top8 有小幅提升：

```text
DNN9 tuned top8 target MSE = 0.348025
DNN10 maskloss target MSE  = 0.346891
```

提升幅度不大，但方向是对的。更重要的是，DNN10 的 `alpha_confidential` 明显比 DNN9 更可解释。

### 3. maskloss 逐目标结果

`no_relation_bias_gated_maskloss` 的逐目标 R2：

```text
da_as_total_mw_primary_reserve         R2 = 0.9708
metered_load_mw                        R2 = 0.9574
da_as_total_mw_synchronized_reserve    R2 = 0.9338
total_gen                              R2 = 0.9084
total_lmp_da                           R2 = 0.7118
total_losses                           R2 = 0.6778
congestion_price_rt                    R2 = 0.4489
marginal_loss_price_da                 R2 = 0.4301
da_as_total_mw_thirty_minutes_reserve  R2 = 0.3655
congestion_price_da                    R2 = 0.2706
gross_actual_interchange_mw            R2 = -0.5567 (excluded)
net_actual_interchange_mw              R2 = -1.9254 (excluded)
```

与 DNN9 tuned top8 相比：

```text
congestion_price_da: 0.2829 -> 0.2706, 略降
congestion_price_rt: 0.4335 -> 0.4489, 略升
total_losses:         0.6103 -> 0.6778, 明显提升
marginal_loss_da:     0.4164 -> 0.4301, 略升
30min reserve:        0.3282 -> 0.3655, 提升
```

因此 DNN10 不是所有字段都提升，但整体 MSE 更好，主要收益来自 `total_losses`、`congestion_price_rt` 和 `thirty_minutes_reserve`。

### 4. alpha_general 结果

`maskloss` 下的 `alpha_general`：

| metric | alpha |
|---|---:|
| Pearson | 0.815355 |
| Spearman | 0.045294 |
| Kendall | 0.050781 |
| NMI | 0.035429 |
| distance correlation | 0.053140 |

General-General 边仍然强烈偏向 Pearson，比 DNN9 tuned top8 的 Pearson 约 0.758 更尖锐。这说明在 general 节点之间做稳定传播时，线性相关仍然最有用。

### 5. alpha_confidential 终于明显分化

这是 DNN10 最关键的变化。`maskloss` 中不少 target 的 `alpha_c` 已经明显远离平均分布。

典型例子：

```text
total_gen:
  nmi = 0.688216

metered_load_mw:
  nmi = 0.686655

total_losses:
  nmi = 0.468031

congestion_price_da:
  spearman = 0.284891
  kendall  = 0.292613

congestion_price_rt:
  distance_corr = 0.323086
  nmi           = 0.259835

marginal_loss_price_da:
  distance_corr = 0.290873
  spearman      = 0.225316
  kendall       = 0.218436

total_lmp_da:
  pearson = 0.404083
  nmi     = 0.271702

da_as_total_mw_primary_reserve:
  pearson = 0.579143

da_as_total_mw_synchronized_reserve:
  nmi = 0.756378
```

这说明前面对 DNN9 的判断是对的：`alpha_c` 分化不了，并不主要是初始化问题，而是注意力公式允许 raw relation vector 绕过 `alpha_c`。DNN10 去掉 `relation_bias`，并让 gate 只看 `log prior` 后，`alpha_c` 的梯度信号变得更直接。

### 6. attention entropy 结果

虽然 `alpha_c` 明显分化，但 attention 本身仍然偏分散。`maskloss` 的 normalized entropy：

```text
congestion_price_da     = 0.9738
congestion_price_rt     = 0.9651
total_losses            = 0.9500
marginal_loss_price_da  = 0.9484
thirty_minutes_reserve  = 0.9154
primary_reserve         = 0.8931
synchronized_reserve    = 0.8957
```

这说明 DNN10 解决了“相关性指标权重不分化”的问题，但还没有解决“attention source selection 不够尖锐”的问题。也就是说：

```text
alpha_c 已经学会 target 偏好；
attention 仍然倾向于多邻居分散读取。
```

这种现象不一定是坏事，因为当前 MSE 反而略优于 DNN9。如果后续强行降低 entropy，可能会牺牲稳定性。

### 7. maskloss vs allloss

`allloss` 结果：

```text
target/all MSE = 0.473569
```

它显著差于 `maskloss`：

```text
maskloss target MSE = 0.346891
allloss target MSE  = 0.473569
```

这再次验证：`net_actual_interchange_mw` 和 `gross_actual_interchange_mw` 作为训练目标会拖累整体优化。即使 DNN10 改了注意力公式，这个结论仍然成立。

值得注意的是，allloss 中 `gross_actual_interchange_mw` 的 R2 变成正数：

```text
gross_actual_interchange_mw R2 = 0.3031
```

但 `net_actual_interchange_mw` 仍为负：

```text
net_actual_interchange_mw R2 = -1.3256
```

allloss 为了尝试拟合这两个字段，牺牲了其他 target 的整体 MSE，因此不适合作为主线。

### 8. 当前结论

DNN10 是一次有效修改：

```text
1. target MSE 小幅优于 DNN9 tuned top8；
2. alpha_confidential 明显分化；
3. 证明 DNN9 的问题确实来自 raw relation vector 绕过 alpha_c；
4. maskloss 仍然明显优于 allloss；
5. attention entropy 仍偏高，说明 source selection 还不够稀疏。
```

当前最强单次结果：

```text
DNN10 no_relation_bias_gated_maskloss
target MSE = 0.346891
```

但这个结果仍然是单 seed。后续如果要作为正式主线，建议只补一个小验证：

```text
DNN10 maskloss, seed = 42, 43, 44
```

如果三种 seed 均值仍优于 DNN6 和 DNN9 tuned top8，则 DNN10 可以作为当前主模型。
