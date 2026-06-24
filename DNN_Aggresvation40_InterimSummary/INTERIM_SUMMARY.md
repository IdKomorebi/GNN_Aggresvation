# DNN_Aggresvation40 阶段性总结

## 1. 本阶段目的

本阶段先停止继续讨论双线注意力、LoRA 等后续结构，只固定两个基础前提：

1. 时间窗口固定为 `window_size=1`。因为当前数据切分是 shuffle，原来的时间窗口并不是连续时间窗口，窗口大于 1 反而会把随机样本拼成“伪时间序列”。
2. 修正 single/multi 的信息集不一致问题。此前 single 模式预测某一个 Confidential 字段时，会把另外 11 个 Confidential 字段当成 General 输入，这会让 single 模式看到 multi 看不到的信息，导致 single 的优势被严重高估。

修正后：

- multi：`n_general=44`，`n_confidential=12`
- single：`n_general=44`，`n_confidential=1`，其余 11 个 Confidential 字段直接从输入表中移除
- DNN probe：每个目标单独训练一个全连接 MLP，只使用原始 44 个 General 字段

## 2. 当前图模型逻辑

把每个字段看成一个节点。General 字段是可见节点，Confidential 字段是要推断的隐藏目标节点。

先用多个相关性指标构造字段间先验强度。对任意两个字段 \(i,j\)，先计算多种关系分数：

$$
r_{ij}^{(m)}, \quad m \in \{\text{Pearson}, \text{Spearman}, \text{Kendall}, \text{NMI}, \text{DistanceCorr}\}
$$

模型再学习不同指标的权重：

$$
p_{ij}=\sum_m \alpha_m r_{ij}^{(m)}
$$

其中 \(p_{ij}\) 就是边的相关性先验强度。

General-General 部分更像 GCN：用相关性先验决定邻居怎么聚合。

$$
h_i'=\sigma\left(W_0 h_i+\sum_{j\in \mathcal{N}(i)} \tilde{p}_{ij} W_g h_j\right)
$$

General-Confidential 部分更像 GAT：Confidential 节点从 General 节点接收信息，注意力分数同时看动态特征匹配和相关性先验。

$$
e_{ij}=\frac{q_i^\top k_j}{\sqrt{d}}+\lambda\log(\epsilon+p_{ij})
$$

$$
a_{ij}=\operatorname{softmax}_{j\in\mathcal{N}(i)}(e_{ij})
$$

$$
h_i'=\sum_{j\in\mathcal{N}(i)} a_{ij}v_j
$$

直观理解：General-General 用“相关图”做平滑和信息传播，General-Confidential 用“带先验的注意力”从可见字段里挑信息来推断隐藏字段。

## 3. 实验设置

输出目录：

- GNN multi：`outputs/graph_multi_window1_fixed_info/20260623_175403`
- GNN single：`outputs/graph_single_window1_fixed_info/<target>/<timestamp>`
- DNN probe：`outputs/dnn_probe_general_only/20260623_180101`

三者共同设置：

- 数据：`data/Processed/pjm_rto_hourly_2025_cleaned.csv`
- 切分：shuffle，训练 70%，测试 30%，seed=42
- 时间窗口：1
- General 字段：44 个
- Confidential 字段：12 个

DNN probe 的形式是：

$$
\hat{y}_t=f_t(x_G)
$$

其中 \(x_G\) 是 44 个 General 输入，\(f_t\) 是每个目标单独训练的 MLP。因此这个 probe 是一个很强的 single-target baseline，不是弱 baseline。

## 4. 结果

汇总文件：

- `outputs/interim_three_experiment_comparison.csv`
- `outputs/interim_three_experiment_summary.json`
- `outputs/interim_three_experiment_r2.png`
- `outputs/interim_three_experiment_r2.pdf`

均值 R2：

| 方法 | Mean R2 |
|---|---:|
| GNN multi, fixed info | 0.8096 |
| GNN single, fixed info | 0.8418 |
| Fully-connected DNN probe | 0.8937 |

逐目标看，GNN single 只在两个辅助服务字段上略高于 DNN probe：

- `da_as_total_mw_primary_reserve`：GNN single 高 0.0005
- `da_as_total_mw_synchronized_reserve`：GNN single 高 0.0009

其余目标 DNN probe 更强，尤其是价格相关字段：

- `congestion_price_da`：GNN single 0.4559，DNN probe 0.6602
- `congestion_price_rt`：GNN single 0.6084，DNN probe 0.6702
- `marginal_loss_price_da`：GNN single 0.7687，DNN probe 0.8816

结论很明确：修正信息集之后，当前图模型不能支持“相关系数构图 GNN 比全连接 DNN 更准”这个强结论。

## 5. 对结果的解释

之前 single 强，很大一部分来自信息泄漏：它可以看到另外 11 个 Confidential 字段。修正后 single 下降是合理的。

现在 DNN probe 强，是因为它有三个优势：

1. 每个目标都有一个独立 MLP，目标专属容量很足。
2. 输入是完整的 44 维 General 向量，没有经过图结构瓶颈。
3. 对许多负荷/发电类目标，General 字段已经包含非常直接的线性或非线性预测信息，MLP 很容易拟合。

当前 GNN 的优势不是准确率，而是结构解释性：它能说明“哪些字段通过相关图向目标传递信息”，还能把 General-General 和 General-Confidential 的传播机制分开看。

## 6. 文献检索结论

检索日期：2026-06-23。

相关基础工作：

- [Kipf and Welling, 2016, Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)：GCN 基础论文，说明在图结构上做邻居聚合是成熟方法。
- [Velickovic et al., 2017, Graph Attention Networks](https://arxiv.org/abs/1710.10903)：GAT 基础论文，说明用注意力给不同邻居分配不同权重是成熟方法。
- [Graph WaveNet, 2019](https://arxiv.org/abs/1906.00121)：指出固定图结构不一定反映真实依赖，并提出自适应依赖矩阵。
- [MTGNN, 2020](https://arxiv.org/abs/2005.11650)：针对多变量时间序列，在依赖未知时通过图学习模块自动学习变量间关系。
- [Leveraging Graph Neural Networks to Forecast Electricity Consumption, 2024](https://arxiv.org/abs/2408.17366)：电力负荷预测中已经有“图推断 + GNN”的方向。

针对“相关系数构图”的检索：

- arXiv 查询 `Pearson correlation` + `graph neural network` 返回多篇结果，说明“用相似度/相关系数构造图再接 GNN”不是全新的通用想法。例如 2026 年的 [DG-SA-GNN](https://arxiv.org/abs/2605.05238) 在推荐系统中构造多种用户相似图，其中包括 Pearson 类相似度，再用 attention 融合。
- arXiv 查询 `Pearson correlation` + `load forecasting` 主要找到负荷预测中用 Pearson 分析需求与外生变量关系的工作，例如 [Load Forecasting in the Era of Smart Grids, 2025](https://arxiv.org/abs/2505.18170)，但这不是字段级相关图 GNN 推断。
- arXiv 查询 `correlation` + `adjacency matrix` + `load forecasting` 没有返回直接命中的结果。这个结果不能证明没人做过，只能说明在这个关键词组合下没有很直接的 arXiv 命中。

所以创新点不能写成“第一次用相关系数构图”。更稳妥的表述是：

> 本项目把多指标相关性作为字段图先验，用于电力数据中从可见 General 字段推断 Confidential 字段；并区分 General-General 的 GCN 先验聚合与 General-Confidential 的 GAT 先验注意力，以做隐私字段推断和可解释敏感字段分析。

## 7. 建议

如果论文主张一定要写成“相关图比全连接 DNN 更准”，当前实验还不够。下一步我建议先做两个更公平的 baseline：

1. Multi-output fully-connected DNN：一个共享 MLP 同时输出 12 个目标，和 GNN multi 对齐。
2. Parameter-matched single MLP：让 DNN single 的参数量接近 GNN single，否则 DNN probe 是很强的上界。

模型上，我建议加一个直接残差分支：

$$
\hat{y}_c=\hat{y}_c^{graph}+\beta_c f_c(x_G)
$$

这里 \(f_c(x_G)\) 是一个小 MLP，\(\hat{y}_c^{graph}\) 是图分支输出。这样不会放弃相关图，但可以补上当前图传播的表达瓶颈。后续通过 ablation 比较：

- 只有 MLP
- 只有 graph
- MLP + graph residual
- graph without correlation prior
- graph with shuffled/random prior

这样才能证明相关图先验到底有没有提供增益。

另一个更可能让图模型赢的方向是做低数据/缺失输入实验。相关图先验的价值通常不一定体现在完整数据、大样本、强 MLP 的场景，而更可能体现在：

- 训练样本只有 10%/20%/50%
- General 字段随机缺失或加噪
- 某些关键 General 字段不可用
- 需要解释敏感字段依赖路径

当前最稳妥的阶段结论是：

> window=1 是正确方向；single/multi 信息集必须修正；相关图 GNN 当前具有解释优势，但准确率尚未超过强 single-target 全连接 DNN。下一步应通过公平 baseline、残差图分支、低数据/缺失输入实验，验证相关图先验是否能产生稳定增益。
