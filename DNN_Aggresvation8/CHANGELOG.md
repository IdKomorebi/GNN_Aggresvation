# DNN_Aggresvation8 日志

## Modify by GPT5.5: 2026-05-22 23:05:47 CST

### 1. 子项目 8 的出发点

子项目 7 直接为每条边学习一组相关系数融合权重：

```text
alpha(i, j)
```

该设计保留了最大边级差异，但参数过于分散。实验后发现，大多数边的 alpha 仍然接近均匀分布，只有少数边打开。这说明逐边自由度虽然充足，但 reconstruction loss 对每条边具体应偏向 Pearson、Spearman、Kendall、NMI 还是 distance correlation 的监督信号不够强。

子项目 8 因此采用更收缩、语义更明确的参数化方式：

```text
General-General: 一组全局 alpha_G
Confidential-Source: 每个 confidential target 一组 alpha_c
```

这样既避免 DNN7 中每条边一套 alpha 带来的参数分散，又保留了隐性推断任务里最关键的目标条件化方向性。

### 2. 模型设计

#### 2.1 General-General

General 节点之间使用固定相关性边权 GCN：

```text
alpha_G = softmax(beta_G / temperature)
A_G(i,j) = sum_m alpha_G[m] * r_ij[m]
```

General 节点只从 General 邻居聚合，不从 Confidential 节点读信息。这样做是为了避免 confidential embedding 反向污染 general 表示。

#### 2.2 Confidential-Source

每个 confidential target `c` 学习一组目标专属相关系数融合权重：

```text
alpha_c = softmax(beta_c / temperature)
p(c,j) = sum_m alpha_c[m] * r_cj[m]
```

其中 `p(c,j)` 是 source 节点 `j` 对 confidential target `c` 的关系先验。

#### 2.3 门控注意力

主线 `hybrid_gated` 使用：

```text
d(c,j) = q_c^T k_j / sqrt(d)
lambda(c,j) = sigmoid(W_lambda [q_c || k_j || r_cj] + b)
score(c,j) = lambda(c,j) * d(c,j) + (1 - lambda(c,j)) * p(c,j)
```

其中：

1. `d(c,j)` 是动态 query-key 匹配；
2. `p(c,j)` 是目标专属关系先验；
3. `lambda(c,j)` 控制该边更依赖动态匹配还是统计先验。

实现时没有真的构造大拼接张量，而是使用等价拆分：

```text
W_q q_c + W_k k_j + W_r r_cj + b
```

这样避免 `[B, C, N, 2H+M]` 大张量造成 CPU 训练过慢。

### 3. 修改内容

核心文件：

```text
DNN_Aggresvation8/src/model.py
DNN_Aggresvation8/src/train.py
DNN_Aggresvation8/scripts/run_pipeline.py
DNN_Aggresvation8/scripts/run_experiments.py
DNN_Aggresvation8/scripts/compare_runs.py
```

新增配置：

```text
DNN_Aggresvation8/configs/experiments/hybrid_gated_loss_mask.yaml
DNN_Aggresvation8/configs/experiments/hybrid_gated_all_loss.yaml
DNN_Aggresvation8/configs/experiments/hybrid_no_gate_loss_mask.yaml
DNN_Aggresvation8/configs/experiments/hybrid_prior_only_loss_mask.yaml
```

新增诊断输出：

```text
hybrid_diagnostics/alpha_general.csv
hybrid_diagnostics/alpha_confidential_by_target.csv
hybrid_diagnostics/confidential_prior_edges.csv
hybrid_diagnostics/confidential_gate_by_edge.csv
attention/confidential_attention_top_edges.csv
attention/confidential_attention_entropy.csv
tables/r2_probe_vs_model_booktabs.png
```

训练器新增：

```text
training.alpha_lr_multiplier
```

用于给 `beta_general` 和 `beta_confidential` 设置更高学习率。本轮使用：

```text
alpha_lr_multiplier = 4.0
edge_alpha_temperature = 0.7
prior_scale_init = 0.0
```

### 4. 测试命令

语法检查：

```bash
conda run -n Pytorch310_MacBookAir python -m py_compile \
  DNN_Aggresvation8/src/model.py \
  DNN_Aggresvation8/src/train.py \
  DNN_Aggresvation8/src/sensitivity.py \
  DNN_Aggresvation8/scripts/run_pipeline.py \
  DNN_Aggresvation8/scripts/run_experiments.py \
  DNN_Aggresvation8/scripts/compare_runs.py
```

四组实验：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation8/scripts/run_experiments.py
```

正式四组 run：

```text
DNN_Aggresvation8/outputs/run_20260522_223527_hybrid_gated_loss_mask
DNN_Aggresvation8/outputs/run_20260522_224019_hybrid_gated_all_loss
DNN_Aggresvation8/outputs/run_20260522_224613_hybrid_no_gate_loss_mask
DNN_Aggresvation8/outputs/run_20260522_225104_hybrid_prior_only_loss_mask
```

汇总文件：

```text
DNN_Aggresvation8/outputs/comparison_summary.csv
```

### 5. 实验结果

| experiment | architecture | loss mask | target MSE | all MSE | excluded MSE |
|---|---|---:|---:|---:|---:|
| `hybrid_gated_loss_mask` | `hybrid_gated` | 是 | 0.372758 | 0.610629 | 1.799982 |
| `hybrid_gated_all_loss` | `hybrid_gated` | 否 | 0.481934 | 0.481934 | - |
| `hybrid_no_gate_loss_mask` | `hybrid_no_gate` | 是 | 0.402871 | 0.685136 | 2.096463 |
| `hybrid_prior_only_loss_mask` | `hybrid_prior_only` | 是 | 0.403777 | 0.704892 | 2.210469 |

各关键字段 R2：

| experiment | congestion_price_da | congestion_price_rt | total_losses | marginal_loss_price_da | thirty_minutes_reserve |
|---|---:|---:|---:|---:|---:|
| `hybrid_gated_loss_mask` | 0.189976 | 0.476639 | 0.408349 | 0.439933 | 0.365558 |
| `hybrid_gated_all_loss` | 0.171430 | 0.475629 | 0.421305 | 0.433667 | 0.291344 |
| `hybrid_no_gate_loss_mask` | 0.151620 | 0.422349 | 0.495546 | 0.451861 | 0.219464 |
| `hybrid_prior_only_loss_mask` | 0.173826 | 0.418398 | 0.389151 | 0.358857 | 0.317313 |

本轮最好结果：

```text
hybrid_gated_loss_mask
target MSE = 0.372758
```

该结果略好于子项目 7 最佳：

```text
DNN7 edge_gat_loss_mask target MSE = 0.375254
```

但仍没有超过：

```text
DNN4 gat_loss_mask target MSE = 0.362518
DNN6 target_bilinear_gat_loss_mask target MSE = 0.348861
```

### 6. 诊断分析

#### 6.1 alpha 是否打开

主线 `hybrid_gated_loss_mask` 的 General-General alpha：

| metric | alpha_G |
|---|---:|
| pearson | 0.177103 |
| spearman | 0.130175 |
| kendall | 0.146953 |
| nmi | 0.403370 |
| distance_corr | 0.142399 |

这比 DNN7 好很多。DNN7 的平均 edge alpha 基本贴近 0.2，而 DNN8 的 `alpha_G` 明确偏向 NMI。

每个 confidential target 的 alpha 也出现了差异，例如：

| target | dominant metric | dominant weight |
|---|---|---:|
| `total_lmp_da` | pearson | 0.316279 |
| `da_as_total_mw_synchronized_reserve` | nmi | 0.468473 |
| `marginal_loss_price_da` | pearson | 0.228047 |
| `congestion_price_da` | distance_corr | 0.226328 |
| `metered_load_mw` | nmi | 0.242343 |

说明子项目 8 的目标级 alpha 比子项目 7 的逐边 alpha 更容易被训练打开。

#### 6.2 gate 是否起作用

`hybrid_gated_loss_mask` 的 gate lambda 统计：

```text
count = 185
mean  = 0.755833
std   = 0.152818
min   = 0.245220
max   = 0.976040
```

这表示多数边更依赖动态 query-key 匹配，但并没有完全抛弃关系先验。不同 target 的 gate 也不同：

```text
total_gen mean lambda                 = 0.599979
metered_load_mw mean lambda           = 0.605617
congestion_price_da mean lambda       = 0.876594
total_losses mean lambda              = 0.879581
```

因此门控不是摆设，确实学出了“不同 target/edge 对动态匹配和统计先验依赖程度不同”。

#### 6.3 attention 是否更聚焦

DNN7 中 `congestion_price_da` 的归一化 attention entropy 约为 0.9966，几乎平均分配。DNN8 主线中：

```text
congestion_price_da entropy_norm = 0.685946
congestion_price_rt entropy_norm = 0.690413
marginal_loss_price_da entropy_norm = 0.245934
total_losses entropy_norm = 0.691026
```

说明 DNN8 的 confidential attention 比 DNN7 明显更聚焦，尤其 `marginal_loss_price_da` 的注意力非常集中。

### 7. 当前结论

子项目 8 的结构判断是成立的：

1. 目标级 alpha 比逐边 alpha 更容易训练打开；
2. gated prior attention 明显优于 no-gate 和 prior-only；
3. loss mask 仍然必要；
4. attention 聚焦程度比 DNN7 明显改善；
5. 整体 MSE 略好于 DNN7，但仍未超过 DNN4/DNN6。

因此，DNN8 不是无效分支。它证明“目标条件化、低参数量、门控先验融合”比 DNN7 的逐边自由 alpha 更合理。但从预测结果看，目前收益还不足以成为最优主线。

### 8. 后续建议

如果继续推进 DNN8，下一步不建议再扩大模型，而应做更小的结构修正：

1. 降低 gate 的动态项偏置，让先验能在部分边上占更大比例，例如初始化 gate bias 为负值。
2. 将 `score = lambda*d + (1-lambda)*p` 改成尺度更稳的形式，例如对 `d` 做 `tanh` 或 LayerNorm，避免动态分数天然压过先验。
3. 只让 confidential 从 General 节点读信息，去掉 Confidential-Confidential source，检查是否减少泄漏式 embedding 依赖。
4. 尝试 `edge_alpha_temperature=0.5` 与 `alpha_lr_multiplier=6`，观察 target alpha 是否进一步分化。
5. 与 DNN6 的 target-bilinear 结构融合：保留 DNN8 的 target alpha 和 gate，但把动态项换成 DNN6 的 bilinear target-conditioned score。

当前综合排序仍然是：

```text
DNN6 target_bilinear_gat_loss_mask 最强
DNN4 gat_loss_mask 次之
DNN8 hybrid_gated_loss_mask 有机制价值，略强于DNN7
DNN7 edge_gat_loss_mask 作为负结果/消融保留
```
