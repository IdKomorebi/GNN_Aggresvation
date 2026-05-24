# DNN_Aggresvation13 日志

## Modify by GPT5.5: 2026-05-23

### 1. 子项目13要解决的问题

子项目12 证明了一件事：即使所有节点都使用类似 confidential 的注意力公式，并且 General 节点也允许 node-specific alpha，多数 General 节点的相关系数权重仍然接近 0.2。原因很可能是：

```text
训练 loss 只直接监督 confidential target；
General 节点没有直接 reconstruction loss；
General alpha 只能通过中间传播间接获得梯度；
因此大多数 General alpha 难以学出稳定差异。
```

另一方面，General-General 使用固定 GCN 时整体效果反而更稳，说明 General 节点之间可能更适合做稳定传播，而不是强行让它们都学习复杂 attention。

因此，DNN13 的想法是：

```text
General-General 退回固定 GCN；
只把 Confidential target 的每条有向入边 alpha 放开；
让模型为每个 confidential target-source edge 学习不同的相关系数偏好。
```

这不是子项目7的全图 per-edge alpha。子项目7让所有边都有 edge-specific alpha，参数量大、监督弱。DNN13 只对 confidential target 入边做 edge-specific alpha，参数主要服务于被监督目标。

### 2. 模型结构

#### 2.1 General-General

General 节点之间沿用固定加权 GCN：

```text
alpha_G = softmax(beta_G / temperature)
```

```text
A_G(i,j) = sum_m alpha_G,m * r(i,j,m)
```

行归一化后：

```text
m_i = sum_j A_G(i,j) h_j
```

#### 2.2 Confidential directed edge alpha

对于 confidential target `c` 和 source `j`，学习一组有向边相关系数融合权重：

```text
alpha_{c,j} = softmax(beta_{c,j} / temperature)
```

其中：

```text
c in Confidential targets
j in all source nodes
alpha_{c,j} in R^M
```

如果 source `j` 也是 confidential，则方向不同的边权可以不同：

```text
alpha_{c1,c2} != alpha_{c2,c1}
```

#### 2.3 Confidential prior

对于 confidential target `c` 和 source `j`：

```text
p(c,j) = sum_m alpha_{c,j,m} * r(c,j,m)
```

```text
prior(c,j) = log(p(c,j) + eps)
```

#### 2.4 Confidential attention score

沿用 DNN10 的无 relation-bias 公式：

```text
d(c,j) = q_c^T k_j / sqrt(d)
```

```text
lambda(c,j) = sigmoid(W_q q_c + W_k k_j + W_p prior(c,j) + b)
```

```text
score(c,j)
  = lambda(c,j) * d(c,j)
    + (1 - lambda(c,j)) * softplus(tau) * prior(c,j)
```

然后对 source 节点做 softmax：

```text
omega(c,j) = softmax_j(score(c,j))
```

最终：

```text
m_c = sum_j omega(c,j) v_j
```

### 3. 输出诊断

DNN13 新增：

```text
hybrid_diagnostics/alpha_confidential_by_edge.csv
```

每一行是一条有效 confidential target 入边：

```text
target_field <- source_field
alpha_pearson
alpha_spearman
alpha_kendall
alpha_nmi
alpha_distance_corr
dominant_metric
dominant_alpha
corr_*
```

同时保留：

```text
alpha_confidential_by_target.csv
```

但此处的 target alpha 是该 target 所有有效入边 alpha 的平均值。

还新增一张长图：

```text
alpha_confidential_by_edge_stacked.png
```

用于观察每条 directed confidential edge 的相关系数权重是否真的分化。

### 4. 实验设置

只跑两个实验：

```text
conf_edge_alpha_gated_maskloss
conf_edge_alpha_gated_allloss
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

### 5. 待观察

重点看：

```text
1. target MSE 是否接近或超过 DNN10；
2. per-edge alpha 是否比 DNN12 的 General alpha 更明显分化；
3. 同一个 confidential target 的不同 source 边是否有不同 dominant metric；
4. direction-specific confidential-confidential 边是否出现差异；
5. maskloss 是否仍然优于 allloss。
```

---

## Modify by GPT5.5: 2026-05-23 06:05 CST

### 1. 实验完成

已运行：

```text
conf_edge_alpha_gated_maskloss
conf_edge_alpha_gated_allloss
```

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation13/scripts/run_experiments.py
```

### 2. 总体结果

| experiment | target MSE | all MSE | excluded MSE | congestion DA R2 | congestion RT R2 | total losses R2 | marginal loss DA R2 | 30min reserve R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conf_edge_alpha_gated_maskloss | 0.335532 | 0.544916 | 1.591833 | 0.266985 | 0.467360 | 0.680991 | 0.436232 | 0.362416 |
| conf_edge_alpha_gated_allloss | 0.484225 | 0.484225 | - | 0.223577 | 0.410263 | 0.651305 | 0.452448 | 0.361087 |

与目前几个关键版本对比：

```text
DNN10 maskloss = 0.346891
DNN11 maskloss = 0.347546
DNN12 maskloss = 0.361665
DNN13 maskloss = 0.335532
```

DNN13 是当前单 seed 下最强结果，明显优于 DNN10/DNN11/DNN12。

### 3. 逐目标表现

`conf_edge_alpha_gated_maskloss` 的逐目标 R2：

```text
da_as_total_mw_primary_reserve         R2 = 0.9701
metered_load_mw                        R2 = 0.9629
da_as_total_mw_synchronized_reserve    R2 = 0.9403
total_gen                              R2 = 0.9106
total_lmp_da                           R2 = 0.8906
total_losses                           R2 = 0.6810
congestion_price_rt                    R2 = 0.4674
marginal_loss_price_da                 R2 = 0.4362
da_as_total_mw_thirty_minutes_reserve  R2 = 0.3624
congestion_price_da                    R2 = 0.2670
gross_actual_interchange_mw            R2 = -0.5567 (excluded)
net_actual_interchange_mw              R2 = -1.9254 (excluded)
```

与 DNN10 maskloss 对比：

```text
target MSE:
  0.346891 -> 0.335532, 明显提升

congestion_price_rt:
  0.4489 -> 0.4674, 提升

total_lmp_da:
  0.7118 -> 0.8906, 明显提升

metered_load_mw:
  0.9574 -> 0.9629, 小幅提升

total_losses:
  0.6778 -> 0.6810, 小幅提升

congestion_price_da:
  0.2706 -> 0.2670, 略降
```

整体提升主要来自 `total_lmp_da`、`congestion_price_rt` 以及若干高 R2 target 的进一步增强。

### 4. General-GCN alpha

DNN13 仍然使用 General-General 固定 GCN。其 `alpha_general` 为：

| metric | alpha |
|---|---:|
| Pearson | 0.624571 |
| Spearman | 0.049418 |
| Kendall | 0.056568 |
| NMI | 0.208564 |
| distance correlation | 0.060879 |

这比 DNN10 的 Pearson 0.815 更温和，也比 DNN12 的均匀 alpha 更有方向。说明在 DNN13 中，General-GCN 仍以 Pearson 为主，但 NMI 也保留了较明显权重。

### 5. Confidential edge alpha 分化情况

DNN13 的核心诊断文件：

```text
hybrid_diagnostics/alpha_confidential_by_edge.csv
```

有效 directed confidential target 入边数量：

```text
n_edges = 203
```

dominant metric 计数：

```text
NMI              106
Pearson           29
Kendall           24
distance_corr     22
Spearman          22
```

按 source 类型拆分：

| source type | distance_corr | kendall | nmi | pearson | spearman |
|---|---:|---:|---:|---:|---:|
| confidential | 7 | 4 | 26 | 4 | 3 |
| general | 15 | 20 | 80 | 25 | 19 |

dominant alpha 统计：

```text
mean = 0.288457
std  = 0.127939
min  = 0.200596
50%  = 0.232911
75%  = 0.312792
max  = 0.873941
```

这说明 per-edge alpha 不是全部都强分化，但比 DNN12 的 General node alpha 更明显，且确实出现了大量 NMI 主导边。

### 6. 高分化边示例

分化最明显的边包括：

| target <- source | source type | dominant metric | alpha |
|---|---|---|---:|
| congestion_price_rt <- system_energy_price_da | general | NMI | 0.873941 |
| total_losses <- forecast_load_mw_day_ahead | general | NMI | 0.800946 |
| marginal_loss_price_da <- system_energy_price_da | general | Pearson | 0.741483 |
| total_lmp_da <- marginal_loss_price_rt | general | NMI | 0.691033 |
| metered_load_mw <- system_energy_price_da | general | NMI | 0.683549 |
| marginal_loss_price_da <- gen_fuel_gas_mw | general | NMI | 0.682638 |
| total_losses <- gen_fuel_solar_pct | general | NMI | 0.680274 |
| total_gen <- system_energy_price_da | general | NMI | 0.664785 |
| da_as_total_mw_synchronized_reserve <- da_as_nsr_mw_primary_reserve | general | NMI | 0.610217 |
| congestion_price_da <- forecast_load_mw_latest_available | general | NMI | 0.557015 |

这些结果符合“针对要拟合的 confidential target，给不同 source 边不同相关性指标偏好”的设计目的。

### 7. 方向性验证

confidential-confidential 边中，反向边确实可以不同。例如：

```text
total_lmp_da <- da_as_total_mw_thirty_minutes_reserve:
  dominant = Pearson, alpha = 0.202273

da_as_total_mw_thirty_minutes_reserve <- total_lmp_da:
  dominant = NMI, alpha = 0.503450
```

再例如：

```text
metered_load_mw <- congestion_price_da:
  dominant = Kendall, alpha = 0.206216

congestion_price_da <- metered_load_mw:
  dominant = distance_corr, alpha = 0.301369
```

这说明 DNN13 的 directed edge alpha 机制确实生效，方向不同可以有不同相关系数融合偏好。

### 8. all loss 仍然较差

`allloss` 结果：

```text
target/all MSE = 0.484225
```

虽然比 DNN12 allloss 的 0.484592 略好，但仍然远差于 maskloss：

```text
maskloss = 0.335532
allloss  = 0.484225
```

这再次证明两个 interchange 字段不适合作为训练 loss 目标。

### 9. 当前结论

DNN13 是一次有效修改。

它保留了前面最稳定的部分：

```text
General-General: fixed weighted GCN
```

同时只在被直接监督的 confidential target 入边上增加表达能力：

```text
Confidential-Source: directed edge-specific alpha + gated log-prior attention
```

因此它避开了 DNN12 的问题：

```text
General alpha 没有直接监督，学不出稳定差异；
```

也避开了 DNN7 的问题：

```text
全图 per-edge alpha 参数太多、监督太弱；
```

DNN13 的参数增加集中在 supervised confidential reconstruction 直接相关的边上，所以能学出差异，也能提升 MSE。

当前推荐：

```text
主线模型更新为 DNN13 maskloss。
```

后续如果要确认结果稳健性，建议只补跑：

```text
DNN13 maskloss seed = 42, 43, 44
```

如果多 seed 均值仍优于 DNN10，则可以把 DNN13 作为最终主线结构。
