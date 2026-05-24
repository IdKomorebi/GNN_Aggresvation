# DNN_Aggresvation12 日志

## Modify by GPT5.5: 2026-05-23

### 1. 子项目12要解决的问题

用户指出，子项目11 仍然没有完全表达当前想法。DNN11 虽然把 General-General 聚合改成了 attention，但 `alpha_general` 仍然是所有 General 节点共享的一组相关系数权重：

```text
alpha_G = [alpha_pearson, alpha_spearman, alpha_kendall, alpha_nmi, alpha_distance_corr]
```

这仍然可能抹平不同 General 节点的差异。

用户真正想测试的是：

```text
不是每条边一组 alpha；
而是每个 target 节点一组 alpha；
同一个 target 节点的所有入边共享这组 alpha；
不同 target 节点的 alpha 可以不同。
```

这与子项目7的 per-edge alpha 不同。子项目7中每条边都有独立 alpha，参数太多，容易学不出稳定差异。DNN12 的设计是 node-specific alpha，参数量明显更小：

```text
General alpha 参数量: n_general * n_metrics
Confidential alpha 参数量: n_confidential * n_metrics
```

而不是：

```text
n_edges * n_metrics
```

### 2. DNN12 的核心修改

在 DNN11 基础上，保持 attention score 公式不变，但把 General-General 的全局 alpha 改成每个 General target 节点一组 alpha。

#### 2.1 General 节点 alpha

对于 General target 节点 `i`：

```text
alpha_i^G = softmax(beta_i^G / temperature)
```

其中：

```text
i in General nodes
alpha_i^G in R^M
```

同一个 target `i` 的所有 General source 入边共享 `alpha_i^G`。

#### 2.2 General-General prior

对于 General target `i` 和 General source `j`：

```text
p_G(i,j) = sum_m alpha_i,m^G * r(i,j,m)
prior_G(i,j) = log(p_G(i,j) + eps)
```

因此，同一个 source `j` 对不同 target `i` 的 prior 可以不同，因为不同 target 有不同的 `alpha_i^G`。

#### 2.3 General-General attention

沿用 DNN11 的 gated log-prior attention：

```text
d_G(i,j) = q_i^T k_j / sqrt(d)
```

```text
lambda_G(i,j) = sigmoid(W_q q_i + W_k k_j + W_p prior_G(i,j) + b)
```

```text
score_G(i,j)
  = lambda_G(i,j) * d_G(i,j)
    + (1 - lambda_G(i,j)) * softplus(tau_G) * prior_G(i,j)
```

```text
omega_G(i,j) = softmax_j(score_G(i,j))
```

```text
m_i = sum_j omega_G(i,j) v_j
```

#### 2.4 Confidential 节点

Confidential 节点沿用 DNN10/DNN11 的 target-specific alpha：

```text
p_C(c,j) = sum_m alpha_c,m * r(c,j,m)
prior_C(c,j) = log(p_C(c,j) + eps)
```

### 3. 输出诊断

DNN12 新增/保留以下 alpha 诊断文件：

```text
hybrid_diagnostics/alpha_general.csv
```

保存 General 节点 alpha 的均值，便于和 DNN8-DNN11 对比。

```text
hybrid_diagnostics/alpha_general_by_target.csv
```

保存每个 General target 节点自己的 alpha。

```text
hybrid_diagnostics/alpha_confidential_by_target.csv
```

保存每个 Confidential target 节点自己的 alpha。

```text
hybrid_diagnostics/alpha_by_all_nodes.csv
```

合并所有节点的 alpha，并标明节点类型：

```text
general / confidential
```

```text
hybrid_diagnostics/alpha_by_all_nodes_stacked.png
```

一张较长的堆叠条形图，展示所有节点的五种相关系数权重，并在字段名前标注：

```text
G | field_name
C | field_name
```

### 4. 实验设置

仍然只跑两个实验：

```text
node_alpha_general_attention_maskloss
node_alpha_general_attention_allloss
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
1. General 节点之间的 alpha 是否出现节点差异；
2. alpha_general 均值是否仍然 Pearson 断档第一；
3. 哪些 General 节点偏向 NMI / distance correlation；
4. target MSE 是否优于 DNN10 / DNN11；
5. maskloss 是否仍然优于 allloss。
```

---

## Modify by GPT5.5: 2026-05-23 03:58 CST

### 1. 实验完成

已运行：

```text
node_alpha_general_attention_maskloss
node_alpha_general_attention_allloss
```

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation12/scripts/run_experiments.py
```

### 2. 总体结果

| experiment | target MSE | all MSE | excluded MSE | congestion DA R2 | congestion RT R2 | total losses R2 | marginal loss DA R2 | 30min reserve R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node_alpha_general_attention_maskloss | 0.361665 | 0.566693 | 1.591833 | 0.279828 | 0.456264 | 0.573907 | 0.411677 | 0.292936 |
| node_alpha_general_attention_allloss | 0.484592 | 0.484592 | - | 0.232014 | 0.440843 | 0.616933 | 0.393576 | 0.318333 |

与前面主线对比：

```text
DNN10 maskloss target MSE = 0.346891
DNN11 maskloss target MSE = 0.347546
DNN12 maskloss target MSE = 0.361665
```

DNN12 性能明显弱于 DNN10/DNN11。说明“每个 General target 一组 alpha”虽然更灵活，但当前训练信号不足以稳定提升性能。

### 3. General alpha 均值

DNN12 maskloss 的 General 节点 alpha 均值为：

| metric | mean alpha |
|---|---:|
| Pearson | 0.213656 |
| Spearman | 0.192201 |
| Kendall | 0.190285 |
| NMI | 0.210719 |
| distance correlation | 0.193139 |

这与 DNN10/DNN11 很不同：

```text
DNN10 alpha_general:
  Pearson = 0.815355

DNN11 alpha_general:
  Pearson = 0.707868

DNN12 mean alpha_general:
  Pearson = 0.213656
  NMI     = 0.210719
```

因此，只看均值，DNN12 不再出现 Pearson 断档第一。但是这不是因为所有 General 节点都学出了清晰差异，而是因为许多 General 节点的 alpha 接近均匀，平均后把差异抹平了。

### 4. General 节点是否真的分化

General 节点 dominant metric 计数：

```text
NMI              23
Pearson           8
Spearman          7
distance_corr     4
Kendall           2
```

General 节点最大 alpha 的统计：

```text
mean = 0.247658
std  = 0.112574
min  = 0.200013
50%  = 0.209123
75%  = 0.223801
max  = 0.677151
```

这说明：

```text
多数 General 节点几乎还是均匀 alpha；
少数 General 节点出现了明显分化。
```

分化最明显的 General 节点包括：

| field | dominant metric | dominant alpha |
|---|---|---:|
| gen_fuel_nuclear_mw | NMI | 0.677151 |
| system_energy_price_da | Pearson | 0.669314 |
| marginal_loss_price_rt | Pearson | 0.575306 |
| total_lmp_rt | NMI | 0.360498 |
| forecast_load_mw_day_ahead | NMI | 0.291825 |
| da_as_nsr_mw_primary_reserve | NMI | 0.276135 |
| gen_fuel_oil_mw | distance_corr | 0.272309 |
| da_as_mcp_synchronized_reserve | Spearman | 0.268551 |

因此 DNN12 确实产生了“不同 General 节点不同 alpha”的现象，但这种现象只集中在少数字段，整体不够强。

### 5. Confidential alpha 仍然明显分化

Confidential 节点 dominant metric 计数：

```text
NMI              5
Pearson          3
distance_corr    2
Kendall          1
Spearman         1
```

Confidential 最大 alpha：

```text
mean = 0.447663
std  = 0.204486
max  = 0.782272
```

典型结果：

```text
da_as_total_mw_synchronized_reserve:
  NMI = 0.782272

total_gen:
  NMI = 0.745735

metered_load_mw:
  NMI = 0.656202

da_as_total_mw_primary_reserve:
  Pearson = 0.589179

congestion_price_rt:
  distance_corr = 0.379813
```

这再次说明：Confidential target 的 alpha 更容易被训练信号推开，因为模型直接监督的是 confidential target reconstruction。

### 6. 长图输出

已生成用户要求的所有节点 alpha 长图：

```text
outputs/run_20260523_033547_node_alpha_general_attention_maskloss/hybrid_diagnostics/alpha_by_all_nodes_stacked.png
```

图中字段名前缀：

```text
G | field_name  表示 General 节点
C | field_name  表示 Confidential 节点
```

同时保存 CSV：

```text
alpha_by_all_nodes.csv
alpha_general_by_target.csv
alpha_confidential_by_target.csv
```

### 7. 当前判断

DNN12 回答了用户提出的问题：

```text
如果 General 节点也像 Confidential 一样，每个 target 节点一组 alpha，
确实可以让部分 General 节点产生不同的相关系数偏好；
但多数 General 节点仍然接近均匀；
整体性能反而下降。
```

这说明之前 DNN10/DNN11 中 Pearson 断档第一并不一定只是坏事。全局或半全局 General alpha 虽然粗糙，但它提供了一个稳定传播先验；DNN12 把 General alpha 拆到每个节点后，参数量增多、直接监督变弱，许多 General 节点没有足够梯度学出明确偏好。

更细地说：

```text
Confidential alpha:
  有直接 reconstruction loss，容易分化。

General node alpha:
  只通过中间传播间接影响 confidential loss，
  梯度路径更长、更弱，
  因此大多数节点保持接近均匀。
```

### 8. 推荐结论

从性能角度：

```text
DNN10 maskloss 仍是当前最强主线。
```

从解释角度：

```text
DNN12 很有价值，因为它证明 node-specific general alpha 可实现，
但在当前监督设置下无法稳定提升性能。
```

如果继续改 DNN12，不建议再增加参数量。更合理的方向是给 General alpha 增加额外监督或约束，例如：

```text
1. 用 mask sensitivity 或 probe R2 给 General 节点 alpha 加辅助目标；
2. 对 General alpha 做 group prior，避免完全自由学习；
3. 只对高敏 General 候选节点启用 node-specific alpha，其余仍共享全局 alpha。
```

当前不建议把 DNN12 作为主线模型。
