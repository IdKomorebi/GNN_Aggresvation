# DNN_Aggresvation7 日志

## Modify by GPT5.5: 2026-05-22 21:03:29 CST

### 1. 子项目 7 的出发点

子项目 4 已经证明了两件事：

1. 将两个几乎不可推断字段 `net_actual_interchange_mw` 和 `gross_actual_interchange_mw` 从 loss 中排除，可以避免模型为了拟合这两个困难目标而牺牲其它 confidential 字段的精度。
2. GAT 化以后，模型能在一定程度上学习不同邻居的重要性，尤其对 `marginal_loss_price_da`、`congestion_price_rt` 等字段有帮助。

但子项目 4 仍有一个结构性限制：相关系数融合权重是全局共享的。也就是说，对任意一条边，Pearson、Spearman、Kendall、NMI、distance correlation 的融合比例都一样：

```text
edge_weight(i, j) = alpha_1 * pearson(i, j)
                  + alpha_2 * spearman(i, j)
                  + alpha_3 * kendall(i, j)
                  + alpha_4 * nmi(i, j)
                  + alpha_5 * dcor(i, j)
```

这个设计参数少、稳定，但也会抹掉边之间的差异。现实中不同字段关系可能很不一样，例如价格类字段之间更偏线性，负荷/燃料/潮流类字段可能存在非线性或单调关系。如果所有边都使用同一组 `alpha`，模型其实只能学到“整体上哪种相关系数更重要”，不能学到“这一条边应该更看 Pearson，那一条边应该更看 NMI”。

因此，子项目 7 的核心问题是：

> 是否允许每条边拥有自己的相关系数融合权重，可以保留更多边级差异，并进一步提升 confidential target 的推断效果？

### 2. 修改思路

将子项目 4 中的全局相关系数权重：

```text
alpha = [alpha_pearson, alpha_spearman, alpha_kendall, alpha_nmi, alpha_dcor]
```

改为边级相关系数权重：

```text
alpha(i, j) = [alpha_pearson(i, j),
               alpha_spearman(i, j),
               alpha_kendall(i, j),
               alpha_nmi(i, j),
               alpha_dcor(i, j)]
```

其中每条边 `(i, j)` 的权重通过 softmax 归一化：

```text
alpha(i, j) = softmax(beta(i, j) / temperature)
```

最终边权为：

```text
edge_weight(i, j) = sum_m alpha_m(i, j) * corr_m(i, j)
```

这样每条边都可以学习不同的相关系数偏好。

该改法的隐患也很明显：参数量从 `5` 个全局权重变成约 `N * N * 5` 个边级权重。当前字段数约 40，因此新增参数约 `40 * 40 * 5 = 8000` 个。这个数量绝对值不算大，但由于监督信号仍然只有 reconstruction loss，且样本窗口有限，边级 `alpha(i, j)` 可能难以充分学习，也可能带来过拟合或优化不稳定。

### 3. 修改内容

#### 3.1 新建子项目目录

从子项目 4 复制出子项目 7：

```text
DNN_Aggresvation7/
```

保留子项目 4 的数据管线、训练方式、target probe 对照结果读取方式，并在模型层做边级相关系数权重改造。

#### 3.2 模型结构改造

修改文件：

```text
DNN_Aggresvation7/src/model.py
```

新增参数：

```text
edge_beta: shape = [num_nodes, num_nodes, num_metrics]
edge_alpha_temperature
```

新增或调整架构：

| 架构名 | 含义 |
|---|---|
| `edge_gcn` | 使用边级相关系数权重，但聚合仍是 GCN 风格 |
| `edge_gat` | 使用边级相关系数权重，并对所有节点启用 GAT 聚合 |
| `edge_conf_gat` | 使用边级相关系数权重，只对 confidential 节点使用 GAT 信息，其它节点保留 GCN 聚合 |

其中 `edge_conf_gat` 是为了测试一个更保守的假设：也许只需要让 confidential target 在聚合邻居时使用注意力，一般节点之间不需要都使用 GAT。这样可以减少 GAT 带来的自由度。

#### 3.3 训练与诊断改造

修改文件：

```text
DNN_Aggresvation7/scripts/run_pipeline.py
```

主要改动：

1. 支持 `edge_alpha_temperature` 配置。
2. 对 `edge_gat` 和 `edge_conf_gat` 输出 attention 诊断。
3. 在每次 run 结束后，将当前模型测试集上的 per-target R2 与子项目 4 的 full-data DNN target probe R2 做对照。
4. 生成可视化三线表图片，而不是只输出 csv。

每个 run 会生成：

```text
tables/r2_probe_vs_model.csv
tables/r2_probe_vs_model_booktabs.png
```

其中 PNG 是三线表风格，可直接用于查看每个 confidential 字段：

1. full-data DNN target probe 的 R2；
2. 当前图模型在 test set 上的 R2；
3. 当前模型相对 target probe 的差值；
4. 该字段是否参与 loss。

#### 3.4 实验配置

新增四组实验配置：

```text
DNN_Aggresvation7/configs/experiments/edge_gcn_all_loss.yaml
DNN_Aggresvation7/configs/experiments/edge_gcn_loss_mask.yaml
DNN_Aggresvation7/configs/experiments/edge_gat_loss_mask.yaml
DNN_Aggresvation7/configs/experiments/edge_conf_gat_loss_mask.yaml
```

四组实验分别用于比较：

1. 边级相关系数权重 + 全部 target loss；
2. 边级相关系数权重 + 排除两个 interchange target；
3. 边级相关系数权重 + 全节点 GAT；
4. 边级相关系数权重 + 只对 confidential 节点使用 GAT。

所有实验 epoch 数设置为 150。

### 4. 测试命令

语法检查：

```bash
conda run -n Pytorch310_MacBookAir python -m py_compile \
  DNN_Aggresvation7/src/model.py \
  DNN_Aggresvation7/src/train.py \
  DNN_Aggresvation7/src/sensitivity.py \
  DNN_Aggresvation7/scripts/run_pipeline.py \
  DNN_Aggresvation7/scripts/run_experiments.py \
  DNN_Aggresvation7/scripts/compare_runs.py
```

四组实验：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation7/scripts/run_experiments.py
```

实验已完成。

### 5. 四组实验结果

汇总文件：

```text
DNN_Aggresvation7/outputs/comparison_summary.csv
```

| experiment | architecture | loss mask | target MSE | all MSE | excluded MSE |
|---|---|---:|---:|---:|---:|
| `edge_gcn_all_loss` | `edge_gcn` | 否 | 0.495646 | 0.495646 | - |
| `edge_gcn_loss_mask` | `edge_gcn` | 是 | 0.393928 | 0.628990 | 1.804302 |
| `edge_gat_loss_mask` | `edge_gat` | 是 | 0.375254 | 0.580064 | 1.604112 |
| `edge_conf_gat_loss_mask` | `edge_conf_gat` | 是 | 0.383171 | 0.583077 | 1.582606 |

各字段 R2：

| experiment | congestion_price_da | congestion_price_rt | total_losses | marginal_loss_price_da | thirty_minutes_reserve |
|---|---:|---:|---:|---:|---:|
| `edge_gcn_all_loss` | 0.164656 | 0.453967 | 0.398942 | 0.386637 | 0.268155 |
| `edge_gcn_loss_mask` | 0.157110 | 0.464344 | 0.345392 | 0.383399 | 0.340483 |
| `edge_gat_loss_mask` | 0.173169 | 0.466601 | 0.407945 | 0.447474 | 0.327457 |
| `edge_conf_gat_loss_mask` | 0.160665 | 0.460525 | 0.415723 | 0.464709 | 0.283644 |

本轮 DNN7 最好的结果是：

```text
edge_gat_loss_mask
target MSE = 0.375254
```

但它没有超过子项目 4 的最佳 `gat_loss_mask`：

```text
DNN4 gat_loss_mask target MSE = 0.362518
```

也没有超过子项目 6 的最佳 target-conditioned 主线：

```text
DNN6 target_bilinear_gat_loss_mask target MSE = 0.348861
```

### 6. 三线表输出位置

每个 run 都生成了一个 PNG 三线表：

```text
DNN_Aggresvation7/outputs/run_20260522_205111_edge_gcn_all_loss/tables/r2_probe_vs_model_booktabs.png
DNN_Aggresvation7/outputs/run_20260522_205252_edge_gcn_loss_mask/tables/r2_probe_vs_model_booktabs.png
DNN_Aggresvation7/outputs/run_20260522_205438_edge_gat_loss_mask/tables/r2_probe_vs_model_booktabs.png
DNN_Aggresvation7/outputs/run_20260522_205748_edge_conf_gat_loss_mask/tables/r2_probe_vs_model_booktabs.png
```

其中最值得优先查看的是：

```text
DNN_Aggresvation7/outputs/run_20260522_205438_edge_gat_loss_mask/tables/r2_probe_vs_model_booktabs.png
```

### 7. 边级 alpha 学习情况

对四个 run 的边级相关系数权重进行诊断，发现：

| experiment | edge alpha std by metric | edge alpha min | edge alpha max |
|---|---|---:|---:|
| `edge_gcn_all_loss` | `[0.007766, 0.005944, 0.005131, 0.015721, 0.005902]` | 0.1258 | 0.3776 |
| `edge_gcn_loss_mask` | `[0.006924, 0.005509, 0.005063, 0.013632, 0.005561]` | 0.1246 | 0.3473 |
| `edge_gat_loss_mask` | `[0.007072, 0.006239, 0.004916, 0.016746, 0.005714]` | 0.1282 | 0.4473 |
| `edge_conf_gat_loss_mask` | `[0.007503, 0.006309, 0.005177, 0.016983, 0.005989]` | 0.1247 | 0.4495 |

结论是：边级 `alpha(i, j)` 确实不是完全一样的，部分边的某些相关系数权重可以到 0.44 左右；但整体标准差仍然偏小，多数边的权重仍接近均匀分布。说明当前监督信号对“每条边应该如何选择相关系数”这个问题约束不够强，150 epoch 内没有学出足够明显、稳定的边级差异。

### 8. 当前结论

子项目 7 验证了一个重要点：

> “每条边使用不同相关系数融合权重”这个想法在机制上可行，但第一次直接把参数放开，并没有带来更好的测试集效果。

具体观察：

1. loss mask 仍然是有效方向。排除两个 interchange target 后，目标字段上的 MSE 明显比全 loss 的 GCN 版本更合理。
2. 在边级 alpha 条件下，全节点 `edge_gat` 优于 `edge_gcn_loss_mask`，说明注意力仍然有帮助。
3. `edge_conf_gat` 没有超过 `edge_gat`，说明只让 confidential 节点使用 GAT 过于保守，或者这种分割并没有解决主要矛盾。
4. 边级 alpha 参数确实学习到了一点差异，但差异不够强，最终没有超过 DNN4/DNN6。
5. 当前瓶颈可能不是“有没有边级 alpha”，而是“边级 alpha 是否有足够监督信号被可靠学习”。

### 9. 后续建议

不建议继续无脑加大模型或继续堆 run。更合理的下一步是控制参数量和增强可解释约束：

1. 低秩边级 alpha：不要直接学习完整的 `N * N * M` 参数，而是学习节点 embedding 后生成边级 alpha，例如 `alpha(i, j) = softmax(MLP([z_i, z_j]))`。
2. 分组边级 alpha：按字段类型学习 alpha，例如 price-price、price-load、price-fuel、general-confidential 等边类型共享一组权重，保留差异但减少参数。
3. 全局 alpha + 边级 residual：以 DNN4 的全局 alpha 为主，只学习较小的边级偏移，避免边级权重漂移过大。
4. 给 edge alpha 单独更高学习率或更低 temperature，观察边级权重是否能更充分打开。
5. 多 seed 复现实验。当前单次 run 只能说明这个版本没有明显提升，还不能完全否定边级权重思路。

当前最稳的主线仍然是：

```text
DNN4 gat_loss_mask
DNN6 target_bilinear_gat_loss_mask
```

子项目 7 更适合作为一个负结果/消融分支：它说明“直接把每条边的相关系数融合权重全部放开”会增加参数，但未必提升泛化性能。

## Modify by GPT5.5: 2026-05-22 21:20:00 CST

### 1. 补充问题

第一次日志中只记录了 `alpha_weights.csv`，但该文件在 edge-specific alpha 模型中保存的是所有有效边上的平均相关系数权重，不是每条边自己的权重。这会导致诊断不充分：如果平均值接近 `[0.2, 0.2, 0.2, 0.2, 0.2]`，仍然无法判断每条边是否真的学出了差异。

因此补充导出逐边 alpha：

```text
edge_alpha/edge_alpha_by_edge.csv
edge_alpha/edge_alpha_summary.csv
edge_alpha/edge_alpha_tensor.npy
edge_alpha/edge_weight_matrix.npy
edge_alpha/edge_alpha_matrix_pearson.csv
edge_alpha/edge_alpha_matrix_spearman.csv
edge_alpha/edge_alpha_matrix_kendall.csv
edge_alpha/edge_alpha_matrix_nmi.csv
edge_alpha/edge_alpha_matrix_distance_corr.csv
```

对于 GAT 模型，额外导出：

```text
edge_alpha/attention_prior_scales.csv
```

### 2. 修改内容

修改文件：

```text
DNN_Aggresvation7/scripts/run_pipeline.py
```

新增 `_save_edge_alpha_diagnostics(...)`，以后每个 edge-specific alpha run 都会自动保存每条边的相关系数融合权重。

新增文件：

```text
DNN_Aggresvation7/scripts/export_edge_alpha.py
```

该脚本可在不重跑训练的情况下，从已有 `model.pt` 中读取 `edge_beta` 并补导出逐边 alpha。

### 3. 补导出命令

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation7/scripts/export_edge_alpha.py
```

已对四个已完成 run 补导出。

### 4. 逐边 alpha 诊断结论

以当前最好 run：

```text
run_20260522_205438_edge_gat_loss_mask
```

为例，`edge_alpha_summary.csv` 显示：

| metric | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| pearson | 0.199840 | 0.007072 | 0.128169 | 0.200000 | 0.323880 |
| spearman | 0.199573 | 0.006239 | 0.129044 | 0.199998 | 0.248985 |
| kendall | 0.199776 | 0.004916 | 0.141051 | 0.200000 | 0.255695 |
| nmi | 0.201109 | 0.016746 | 0.128478 | 0.200000 | 0.447300 |
| distance_corr | 0.199702 | 0.005714 | 0.128363 | 0.200000 | 0.235687 |

这说明：

1. 大多数边的权重仍然非常接近均匀分布；
2. 少数边确实学出了明显偏好，例如某些边的 `nmi` 权重可达到 0.44 左右；
3. 但这种差异太稀疏，整体上没有成为稳定有效的结构改进。

四个 run 的逐边偏离情况：

| experiment | edges | mean abs deviation from 0.2 | p95 max deviation | max deviation |
|---|---:|---:|---:|---:|
| `edge_gcn_all_loss` | 816 | 0.00273 | 0.02942 | 0.17756 |
| `edge_gcn_loss_mask` | 816 | 0.00240 | 0.02616 | 0.14731 |
| `edge_gat_loss_mask` | 816 | 0.00202 | 0.02329 | 0.24730 |
| `edge_conf_gat_loss_mask` | 816 | 0.00242 | 0.02630 | 0.24948 |

核心判断：边级 alpha 不是完全没有学习，但绝大多数边没有被充分区分开。

### 5. 对效果不理想的进一步分析

当前效果不理想主要可能来自以下几点：

1. `edge_beta` 从全零初始化，softmax 后每条边一开始都是均匀权重。由于 reconstruction loss 对每条边 alpha 的梯度是间接的，150 epoch 内大多数边很难从均匀状态明显打开。
2. GAT 中的注意力分数为节点注意力分数加边权先验：

```text
logits = node_score + softplus(prior_scale) * log(edge_prior)
```

当前 `edge_gat_loss_mask` 的三层先验系数为：

```text
layer 0: 1.450981
layer 1: 1.053650
layer 2: 1.993942
```

这个系数不小，说明边权先验对注意力有较强影响。由于 `edge_prior` 本身来自几乎均匀的 alpha 融合，GAT 可能主要被原始相关图结构牵引，而不是学出更强的目标相关注意力差异。

3. confidential attention entropy 仍然偏高。例如 `congestion_price_da` 的归一化 entropy 为 0.9966，说明它的注意力几乎平均分配给有效邻居；`total_losses` 为 0.9200，`marginal_loss_price_da` 为 0.9116，也没有非常尖锐。这表示 GAT 本身没有强烈选择少数关键邻居。
4. 边级 alpha 的参数虽然不算巨大，但监督信号仍是 confidential reconstruction loss，不是直接监督“哪条边应该选哪种相关系数”。所以参数更多以后，未必更容易学，反而可能更难稳定优化。

### 6. 可尝试的超参数方向

可以尝试调参，但建议目标明确，不要盲跑：

1. 降低 GAT 边权先验系数：把 `att_prior_scale` 的初始值从 1.0 降到 0.0 或 -1.0，使 `softplus(scale)` 从约 1.31 降到约 0.69 或 0.31，让节点注意力有更大空间覆盖边权先验。
2. 降低 `edge_alpha_temperature`：例如从 1.0 改为 0.5 或 0.25，让同样的 `edge_beta` 差异经过 softmax 后更容易拉开。
3. 给 `edge_beta` 设置单独更高学习率：例如主模型 lr 仍为 0.0005，`edge_beta` 使用 0.002 或 0.005。
4. 对 edge alpha 加熵惩罚或方差奖励，让每条边不要长期停留在均匀融合。但这个要谨慎，否则可能为了“差异”而制造无意义偏置。
5. 不建议继续直接增加 GAT 层数或 hidden dim。当前问题更像是边权与注意力没有被有效打开，不是表示维度不够。

更推荐的下一轮小实验是：

```text
edge_gat_loss_mask
+ edge_alpha_temperature = 0.5
+ att_prior initial raw value = 0.0
+ edge_beta lr multiplier = 4
```

这组实验能直接回答：当前失败是否只是因为 edge alpha 和注意力先验过于保守/过强，还是这个建模方向本身收益有限。
