# DNN_Aggresvation9 日志

## Modify by GPT5.5: 2026-05-22 23:38:12 CST

### 1. 子项目 9 的出发点

子项目 8 证明了目标级相关系数融合权重比子项目 7 的逐边 alpha 更容易训练打开：

```text
General-General: alpha_G
Confidential-Source: alpha_c
```

但 DNN8 的最优结果：

```text
hybrid_gated_loss_mask target MSE = 0.372758
```

仍没有超过子项目 6：

```text
DNN6 target_bilinear_gat_loss_mask target MSE = 0.348861
```

因此，子项目 9 的目标是融合两条线：

1. 保留 DNN8 的 `alpha_G / alpha_c / gated prior`；
2. 引入 DNN6 的强 target-bilinear 动态注意力；
3. 恢复 DNN6 中有效的 target-specific output heads。

### 2. 模型设计

#### 2.1 General-General

General 节点之间仍使用 DNN8 的固定相关性边权 GCN：

```text
alpha_G = softmax(beta_G / temperature)
A_G(i,j) = sum_m alpha_G[m] * r_ij[m]
```

General 节点只从 General 邻居聚合。

#### 2.2 Confidential-Source target alpha

每个 confidential target `c` 学习自己的相关性指标融合权重：

```text
alpha_c = softmax(beta_c / temperature)
p(c,j) = sum_m alpha_c[m] * r_cj[m]
```

其中 `p(c,j)` 是目标专属关系先验。

#### 2.3 DNN6式 bilinear 动态项

动态 attention 分数改为 DNN6 的 target-bilinear 形式：

```text
q_c = W_q h_c + U_q e_c
k_j = W_k h_j
d(c,j) = q_c^T k_j / sqrt(d)
```

并保留一个 relation bias：

```text
b(c,j) = W_r r_cj
```

因此动态项为：

```text
d'(c,j) = d(c,j) + b(c,j)
```

#### 2.4 gated prior 融合

最终分数仍使用门控融合：

```text
lambda(c,j) = sigmoid(W_lambda[q_c || k_j || r_cj] + b)
score(c,j) = lambda(c,j) * d'(c,j) + (1 - lambda(c,j)) * p(c,j)
```

实现中使用等价拆分：

```text
Wq q_c + Wk k_j + Wr r_cj + b
```

避免构造大张量。

#### 2.5 target-specific output heads

每个 confidential target 使用独立输出头：

```text
head_c(h_c) -> y_c
```

这继承了 DNN6 中效果较好的部分。

### 3. 修改内容

核心文件：

```text
DNN_Aggresvation9/src/model.py
DNN_Aggresvation9/scripts/run_pipeline.py
DNN_Aggresvation9/scripts/run_experiments.py
DNN_Aggresvation9/scripts/compare_runs.py
```

新增两组配置：

```text
DNN_Aggresvation9/configs/experiments/hybrid_bilinear_gated_loss_mask.yaml
DNN_Aggresvation9/configs/experiments/hybrid_bilinear_gated_all_loss.yaml
```

本轮只跑两个主实验：

1. `hybrid_bilinear_gated_loss_mask`；
2. `hybrid_bilinear_gated_all_loss`。

### 4. 关键配置

```text
architecture: hybrid_bilinear_gated
hidden_dim: 64
num_layers: 3
attention_dim: 64
attention_temperature: 1.0
edge_alpha_temperature: 0.7
prior_scale_init: 0.0
gate_bias_init: -1.0
target_specific_heads: true
epochs: 300
alpha_lr_multiplier: 4.0
```

`gate_bias_init=-1.0` 的目的，是让模型初期不要过早完全依赖动态项，给关系先验 `p(c,j)` 更多参与空间。

### 5. 测试命令

语法检查：

```bash
conda run -n Pytorch310_MacBookAir python -m py_compile \
  DNN_Aggresvation9/src/model.py \
  DNN_Aggresvation9/src/train.py \
  DNN_Aggresvation9/src/sensitivity.py \
  DNN_Aggresvation9/scripts/run_pipeline.py \
  DNN_Aggresvation9/scripts/run_experiments.py \
  DNN_Aggresvation9/scripts/compare_runs.py
```

前向检查通过：

```text
output shape = [8, 12]
alpha_G shape = [5]
alpha_c shape = [12, 5]
gate shape = [12, 56]
```

正式实验：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation9/scripts/run_experiments.py
```

正式 run：

```text
DNN_Aggresvation9/outputs/run_20260522_232311_hybrid_bilinear_gated_loss_mask
DNN_Aggresvation9/outputs/run_20260522_233114_hybrid_bilinear_gated_all_loss
```

汇总文件：

```text
DNN_Aggresvation9/outputs/comparison_summary.csv
```

### 6. 实验结果

| experiment | loss mask | target MSE | all MSE | excluded MSE |
|---|---:|---:|---:|---:|
| `hybrid_bilinear_gated_loss_mask` | 是 | 0.364940 | 0.569422 | 1.591833 |
| `hybrid_bilinear_gated_all_loss` | 否 | 0.497876 | 0.497876 | - |

关键字段 R2：

| experiment | congestion_price_da | congestion_price_rt | total_losses | marginal_loss_price_da | thirty_minutes_reserve |
|---|---:|---:|---:|---:|---:|
| `hybrid_bilinear_gated_loss_mask` | 0.172712 | 0.427160 | 0.600525 | 0.483243 | 0.360360 |
| `hybrid_bilinear_gated_all_loss` | 0.169483 | 0.451175 | 0.371152 | 0.486750 | 0.353985 |

与前序最佳结果对比：

```text
DNN9 hybrid_bilinear_gated_loss_mask  = 0.364940
DNN8 hybrid_gated_loss_mask           = 0.372758
DNN7 edge_gat_loss_mask               = 0.375254
DNN4 gat_loss_mask                    = 0.362518
DNN6 target_bilinear_gat_loss_mask    = 0.348861
```

DNN9 明显优于 DNN8/DNN7，接近 DNN4，但仍未超过 DNN6。

### 7. 诊断分析

#### 7.1 alpha_G

主线 `hybrid_bilinear_gated_loss_mask` 的 General-General alpha：

| metric | alpha_G |
|---|---:|
| pearson | 0.407593 |
| spearman | 0.152062 |
| kendall | 0.144375 |
| nmi | 0.129961 |
| distance_corr | 0.166009 |

这与 DNN8 不同。DNN8 中 `alpha_G` 明显偏向 NMI，而 DNN9 中 `alpha_G` 明显偏向 Pearson。说明引入 bilinear 动态项后，General-General 部分更倾向于使用线性相关结构稳定传播。

#### 7.2 alpha_c

主线中每个 target 的 `alpha_c` 没有像 DNN8 那样明显分化，大多数仍接近 0.2。少数例外：

```text
congestion_price_rt: nmi = 0.245167
da_as_total_mw_synchronized_reserve: nmi = 0.236102
marginal_loss_price_da: pearson = 0.208012
```

这说明 DNN9 的 bilinear 动态项较强，削弱了 target alpha 的必要性。模型主要靠 bilinear dynamic score 与 target-specific heads 提升效果，而不是靠 `alpha_c` 大幅分化。

#### 7.3 gate

主线 gate lambda：

```text
count = 185
mean  = 0.670333
std   = 0.136825
min   = 0.200133
max   = 0.946018
```

相比 DNN8 的 mean lambda 约 0.756，DNN9 因为 `gate_bias_init=-1.0`，先验参与比例更高。但从 alpha_c 和 attention 结果看，先验仍没有成为主导。

不同 target 的 gate 差异明显：

```text
net_actual_interchange_mw mean lambda = 0.389575
gross_actual_interchange_mw mean lambda = 0.500591
congestion_price_rt mean lambda = 0.829107
congestion_price_da mean lambda = 0.774014
```

低可推断 interchange 字段更依赖先验，价格类字段更依赖动态 bilinear 匹配。

#### 7.4 attention entropy

DNN9 的 attention entropy 较高：

```text
congestion_price_da entropy_norm = 0.879146
congestion_price_rt entropy_norm = 0.915484
marginal_loss_price_da entropy_norm = 0.952166
total_losses entropy_norm = 0.903663
```

这与 DNN8 不同。DNN8 的 gated attention 对部分目标明显更尖锐，例如 `marginal_loss_price_da` entropy_norm 约 0.245934。DNN9 虽然 MSE 更好，但 attention 分布更平，说明预测提升更多来自 bilinear 表示能力和 target-specific heads，而不是更强的 sparse source selection。

### 8. 当前结论

DNN9 验证了：

1. 把 DNN6 的 bilinear target attention 与 DNN8 的 gated prior 结合是有效的；
2. loss mask 仍然非常关键；
3. target-specific heads 对结果有帮助；
4. DNN9 比 DNN8 更强，但还没有超过 DNN6；
5. 关系先验在 DNN9 中没有被充分利用，`alpha_c` 分化弱于 DNN8。

因此，DNN9 是一条“接近强基线但尚未突破”的融合分支。

### 9. 后续建议

如果继续在 DNN9 上调参，建议只跑 `hybrid_bilinear_gated_loss_mask`，重点解决“bilinear 动态项压过先验”的问题：

1. 将 `gate_bias_init` 从 `-1.0` 改为 `-2.0`，让先验初期权重更高。
2. 将 `attention_temperature` 从 `1.0` 提高到 `1.5` 或 `2.0`，降低 bilinear logits 的尖锐程度。
3. 对 dynamic score 做 `tanh` 或 LayerNorm，再与 prior 融合，避免尺度不匹配。
4. 将 `edge_alpha_temperature` 从 `0.7` 降到 `0.5`，鼓励 `alpha_c` 分化。
5. 尝试禁用 Confidential-Confidential source，只允许 Confidential 从 General source 读取，减少 target embedding 间互相借力。

当前综合判断：

```text
DNN6 仍是最强结果；
DNN9 证明融合方向有效，但需要进一步约束 dynamic/prior 尺度；
DNN8 提供了更好的 alpha_c 可解释性；
DNN9 提供了更好的预测性能。
```

---

## Modify by GPT5.5: 2026-05-23 01:27:00 CST

### 1. 本轮问题

用户提出一个很关键的怀疑：DNN9 中 confidential target 的相关系数融合权重 `alpha_c` 仍然太平均，导致目标条件化关系先验没有真正区分 Pearson、Spearman、Kendall、NMI、distance correlation 的作用。这个问题可能不是结构完全错误，而是训练设置太温和：

1. `alpha_c` 初始值接近全 0，经过 softmax 后天然接近均匀分布；
2. `edge_alpha_temperature` 不够低，softmax 后的 alpha 不够尖锐；
3. `alpha_lr_multiplier` 不够大，alpha 学得慢；
4. `top_k` 可能影响图密度，太密会稀释有效邻居，太稀会丢失有用 source；
5. DNN9 的 bilinear dynamic score 表达能力较强，可能压过 prior 分支，使 alpha_c 很难成为主要贡献。

因此本轮不再大范围改变模型结构，而是只对 `hybrid_bilinear_gated_loss_mask` 做聚焦调参。

### 2. 修改内容

#### 2.1 增加 alpha 随机初始化

在 `src/model.py` 中为 `InferenceDrivenGNN` 增加参数：

```text
alpha_init_std
```

当 `alpha_init_std > 0` 时，对 `beta_general` 和 `beta_confidential` 做正态随机初始化：

```text
Normal(0, alpha_init_std)
```

这样 softmax 之后的初始 alpha 不再严格接近均匀分布，给不同相关性指标一个初始差异。

#### 2.2 加强 alpha 学习与 prior 参与

新增一组 tuned config，核心设置为：

```text
attention_temperature = 1.5
edge_alpha_temperature = 0.5
alpha_init_std = 0.5
gate_bias_init = -2.0
alpha_lr_multiplier = 8.0
epochs = 300
patience = 80
seed = 43
loss exclude = net_actual_interchange_mw, gross_actual_interchange_mw
```

含义如下：

1. `attention_temperature=1.5`：降低 dynamic attention logit 的主导性；
2. `edge_alpha_temperature=0.5`：鼓励 alpha 更尖锐；
3. `alpha_init_std=0.5`：避免 alpha 从完全平均状态开始；
4. `gate_bias_init=-2.0`：让模型初期更依赖 relation prior；
5. `alpha_lr_multiplier=8.0`：让 alpha 参数更新更快；
6. 继续排除两个低可推断 interchange 字段的 loss，避免它们拖累其他 target。

#### 2.3 增加 top-k 对比

本轮共测试五组 top-k：

```text
top_k = 3, 4, 5, 8, 12
```

对应配置文件：

```text
configs/experiments/tuned_alpha_top3_loss_mask.yaml
configs/experiments/tuned_alpha_top4_loss_mask.yaml
configs/experiments/tuned_alpha_top5_loss_mask.yaml
configs/experiments/tuned_alpha_top8_loss_mask.yaml
configs/experiments/tuned_alpha_top12_loss_mask.yaml
```

并新增批量运行脚本：

```text
scripts/run_tuned_alpha_experiments.py
```

### 3. 重要修正

最初生成 top3、top4、top8、top12 配置时，误用了 `da_as_as_req_mw_*` 作为 drop columns，导致字段数变成 53，而不是公平对比所需的 56。发现后已经删除错误结果目录，修正 drop columns 为：

```text
da_as_as_mw_primary_reserve
da_as_as_mw_synchronized_reserve
da_as_as_mw_thirty_minutes_reserve
```

并重新运行全部五组 tuned 实验。因此下面结果均来自修正后的 56 字段公平实验。

### 4. 测试结果

| experiment | top_k | target MSE | all MSE | excluded MSE | congestion DA R2 | congestion RT R2 | total losses R2 | marginal loss DA R2 | 30min reserve R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original loss mask | 8 | 0.364940 | 0.569422 | 1.591833 | 0.172712 | 0.427160 | 0.600525 | 0.483243 | 0.360360 |
| tuned alpha top3 | 3 | 0.378248 | 0.580512 | 1.591833 | 0.224265 | 0.363868 | 0.595162 | 0.447083 | 0.224644 |
| tuned alpha top4 | 4 | 0.378196 | 0.580469 | 1.591833 | 0.197952 | 0.445527 | 0.565117 | 0.432777 | 0.180291 |
| tuned alpha top5 | 5 | 0.384614 | 0.585817 | 1.591833 | 0.186091 | 0.440007 | 0.531592 | 0.443684 | 0.231703 |
| tuned alpha top8 | 8 | 0.348025 | 0.555326 | 1.591833 | 0.282877 | 0.433527 | 0.610263 | 0.416366 | 0.328220 |
| tuned alpha top12 | 12 | 0.358204 | 0.563808 | 1.591833 | 0.182358 | 0.485534 | 0.660833 | 0.404931 | 0.273262 |

本轮最优是：

```text
tuned_alpha_top8_loss_mask
target MSE = 0.348025
```

它优于 DNN9 原始 loss-mask 主线：

```text
0.348025 < 0.364940
```

并且单次实验结果略好于此前 DNN6 最优结果：

```text
DNN6 best target MSE ~= 0.348861
```

不过需要注意，本轮 tuned 实验使用 `seed=43`，而此前 DNN9 原始主线使用 `seed=42`。因此这个“略好于 DNN6”的结论只能作为候选信号，不能直接作为最终论文结论。更稳妥的验证方式是对 top8 配置补跑 seed 42、43、44 的多随机种子平均。

### 5. alpha 与 attention 诊断

#### 5.1 General-General alpha 明显打开

top8 最优模型的 `alpha_general` 为：

| metric | alpha |
|---|---:|
| Pearson | 0.758161 |
| Spearman | 0.054590 |
| Kendall | 0.058284 |
| NMI | 0.059847 |
| distance correlation | 0.069119 |

这说明随机初始化、更低 alpha temperature、更高 alpha 学习率确实让 General-General 相关系数融合权重摆脱了均匀分布。模型强烈偏向 Pearson，说明在当前数据和目标下，线性相关对 general 节点稳定传播更有用。

#### 5.2 Confidential alpha_c 仍然只是轻微分化

top8 中 `alpha_c` 相比原始 DNN9 有一定变化，但多数 target 仍接近均匀。相对明显的 target 包括：

```text
total_losses: nmi = 0.293924
congestion_price_da: nmi = 0.213364, spearman = 0.204341
marginal_loss_price_da: pearson = 0.211611, distance_corr = 0.207562
da_as_total_mw_synchronized_reserve: nmi = 0.219810
```

这说明参数调整可以让 confidential alpha_c 有一点差异，但没有像 `alpha_general` 那样完全打开。也就是说，当前结构里 `alpha_c` 不是主要性能提升来源，主要提升更可能来自：

1. top8 的图密度比较合适；
2. General-General 传播变成 Pearson 主导，更稳定；
3. `gate_bias_init=-2.0` 增强了 prior 初期参与；
4. `attention_temperature=1.5` 缓和了 bilinear dynamic score 过强的问题。

#### 5.3 attention 仍有目标差异

top8 下部分 target attention entropy 仍较高：

```text
congestion_price_da entropy_norm = 0.9561
thirty_minutes_reserve entropy_norm = 0.9793
total_losses entropy_norm = 0.9194
```

但 reserve/load 类 target 更尖锐：

```text
da_as_total_mw_primary_reserve entropy_norm = 0.3137
da_as_total_mw_synchronized_reserve entropy_norm = 0.3645
metered_load_mw entropy_norm = 0.6041
total_gen entropy_norm = 0.6240
```

这说明模型不是完全平均聚合，而是不同 target 有不同 source selection 行为。问题在于 price 类和部分 loss 类 target 的 attention 仍偏分散。

### 6. 当前判断

这轮调参是有用的，但不是“只靠 alpha_c 解决问题”。

更准确地说：

```text
随机 alpha 初始化 + 更强 alpha 学习 + 更强 prior + top-k 搜索
确实能提升 DNN9；
但 confidential alpha_c 仍没有大幅分化；
真正明显打开的是 general alpha；
top8 是当前最合适的图密度。
```

top3 和 top4 并没有整体变好，说明图太稀会丢掉必要 source。top12 虽然提升了 `congestion_price_rt` 和 `total_losses`，但损害了 `thirty_minutes_reserve` 和 `marginal_loss_price_da`，说明图太密也会引入噪声。top8 在整体 MSE 上最平衡。

### 7. 下一步建议

如果继续推进，不建议继续无边界地堆结构。建议只做一个确认实验：

```text
tuned_alpha_top8_loss_mask
seed = 42, 43, 44
```

看它是否稳定优于：

```text
DNN6 best
DNN9 original loss-mask
```

如果多 seed 后 top8 仍然稳定更好，那么 DNN9 可以作为当前最强主线；如果只是 seed 43 偶然更好，则仍应把 DNN6 作为主线，DNN9 作为结构探索补充。

同时，若要继续解决 `alpha_c` 过平均的问题，下一步不应再只调 temperature，而应考虑给 `alpha_c` 加入更直接的目标监督信号，例如让每个 confidential target 的 `alpha_c` 与该 target 的全量 DNN 推断难度或 source-level R2 发生联系。否则 `alpha_c` 只是通过最终预测 loss 间接学习，梯度信号可能太弱。
