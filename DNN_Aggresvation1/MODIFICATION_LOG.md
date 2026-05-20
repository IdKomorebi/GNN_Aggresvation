# 修改日志

> 重要更正：2026-05-20 13:04:55 CST 之前的两轮实验中，General 推断能力 `q_i` 的实现不符合最终方法定义。此前实现把相关性矩阵融合分数当作推断能力，后来短暂改成了线性回归测试集 R²，但都不是用户要求的 DNN 推断能力。因此，前两轮结果只作为错误过程和对照记录保留，不能作为最终实验结论。当前有效版本从 `2026-05-20 13:04:55 CST - Modify by GPT5.5` 开始。

## 2026-05-20 05:36:55 CST - Modify by GPT5.5

### 本次修改前发现的问题

1. 原始版本中 General 字段的标签不是真实标签，而是对每一个 General 字段都用其到 Confidential 字段的 Pearson 平方相关性生成。因此模型主要是在复刻一套 Pearson 规则，而不是学习外部真实风险。
2. 输出表中的 `直接风险真值_y` 命名不准确。对 General 字段来说，它只是伪标签，不是经过外部验证的真实标签。
3. `s_init` 被直接放入节点输入特征，同时 Confidential 字段标签固定为 `1.0`。这会让模型很容易把 Confidential 字段拟合到高风险，使很低的 Confidential MAE 不再具有足够强的说明力。
4. 可学习相关性权重长期接近均匀分布。主要原因包括 beta 初始化为全 0、beta L2 正则、beta 梯度很弱、邻接矩阵行归一化会抵消尺度差异，以及 Pearson/Spearman/Kendall/dCor 矩阵之间高度冗余。
5. 训练曲线不是训练集/测试集曲线，而是在同一批图节点上的拟合误差和 MAE。
6. NMI 权重略高于其他指标的幅度很小，不足以证明 NMI 明显占主导。
7. 原实验缺少消融对照，无法判断可学习融合是否真的优于均匀融合或固定单一相关性指标。

### 本次代码修改

1. 修改 `src/data_processing.py` 中的 General 伪标签生成逻辑。
   - General 伪标签改为由“推断能力分数”和“初始敏感度传播分数”融合得到。
   - 推断能力分数使用五种相关性矩阵，而不是只使用 Pearson 平方。
   - 初始敏感度传播分数使用 Confidential 种子信号在固定图结构上传播得到。

2. 将 General 全量伪标签监督改为少量 General 锚点监督。
   - 所有 Confidential 字段仍作为监督节点，标签为 `1.0`。
   - 仅选取少量 General 字段作为伪标签监督锚点。
   - 未被选中的 General 字段导出为 `Unlabeled`，不再参与监督损失。

3. 增加标签诊断信息导出。
   - `supervision_mask`
   - `label_source`
   - `pseudo_y`
   - `inference_score`
   - `sensitivity_score`

4. 修改 `src/train.py` 中的训练损失。
   - 损失只在已监督节点上计算。
   - Confidential 和已标注 General 分别计算组内均值 MSE。
   - `w_c` 用于平衡 Confidential 组损失和 General 锚点组损失。
   - 默认关闭 beta 的 L2 正则，设置为 `lmbda: 0.0`。
   - 增加随机种子参数，便于复现实验。

5. 修改 `scripts/run_pipeline.py`。
   - 读取新的伪标签配置。
   - 将少量锚点标签参数传入数据准备流程。
   - 导出更清晰的 CSV 字段，包括监督状态、标签来源、推断能力分数、初始敏感度传播分数。

6. 修改 `configs/config.yaml`。
   - 增加 `pseudo_label` 配置。
   - 设置 `lmbda: 0.0`。
   - 增加 `seed: 42`。

### 本次测试结果

1. 主流水线可以正常运行。
2. 在 10 个 General 监督锚点、seed 42 下，最终 alpha 为：
   - Pearson: `0.203133`
   - Spearman: `0.207352`
   - Kendall: `0.201217`
   - NMI: `0.197822`
   - dCor: `0.190476`
3. 多随机种子测试显示 alpha 仍然只是在 `0.19-0.21` 附近小幅移动，平均最大最小差距约为 `0.018`。
4. 消融实验显示，可学习融合与固定均匀融合几乎打平，因此 10 锚点设置下仍不能强力证明 beta 是有效的可学习模块。
5. 10 锚点设置下，未标注 General 字段存在明显外推漂移：部分字段的全量伪风险分数不高，但模型预测值 `y_hat` 被推得很高。

## 2026-05-20 12:44:17 CST - Modify by GPT5.5

### 为什么要做这次修改

上一轮实验暴露出一个关键问题：只使用 10 个 General 监督锚点时，未标注 General 字段的约束太弱。模型可以很好拟合 Confidential 和少量 General 锚点，但对其他 General 字段的外推不稳定，出现了 `pseudo_y` 不高而 `y_hat` 很高的现象。

因此，本次实验将 General 监督锚点数从 10 提高到 20，目的是增加对 General 风险刻度的约束，观察三个问题：

1. 未标注 General 的外推漂移是否减弱。
2. beta/alpha 是否会比 10 锚点设置更明显地拉开。
3. 可学习融合是否开始稳定优于均匀融合。

### 本次代码修改

1. 修改 `configs/config.yaml`：
   - 将 `pseudo_label.general_label_count` 从 `10` 调整为 `20`。
   - 保持 `high_risk_ratio: 0.7`，即 20 个 General 锚点中约 14 个高风险锚点、6 个低风险锚点。
   - 保持 `inference_weight: 0.65`。
   - 保持 `lmbda: 0.0`，继续观察无 L2 正则时 beta 的自然分化。

2. 未修改模型结构、图结构构建方式、训练轮数、学习率和 `w_c`。

### 主流水线测试结果

主流水线在 20 个 General 锚点、seed 42 下正常运行。最终 alpha 为：

- Pearson: `0.183308`
- Spearman: `0.210707`
- Kendall: `0.218192`
- NMI: `0.192539`
- dCor: `0.195253`

相比 10 锚点设置，alpha 的最大最小差距明显变大：

- 10 锚点 seed 42：约 `0.0169`
- 20 锚点 seed 42：约 `0.0349`

主流水线输出的 General Top10 也更接近高伪风险锚点及业务直觉排序，前几位为：

1. `forecast_load_mw_latest_available`
2. `system_energy_price_da`
3. `total_pjm_rmccp_cr`
4. `forecast_load_mw_day_ahead`
5. `gross_sched_interchange_mw`

### 多随机种子测试结果

使用 seed `0-9` 重跑 10 次后，alpha 均值和标准差为：

- Pearson: mean `0.182882`, std `0.002925`
- Spearman: mean `0.206982`, std `0.004042`
- Kendall: mean `0.221157`, std `0.002467`
- NMI: mean `0.195290`, std `0.006498`
- dCor: mean `0.193689`, std `0.006425`

整体表现：

- alpha 最大最小差距均值：`0.03854`
- alpha 最大最小差距范围：`0.03479 - 0.04410`
- 未标注 General 与全量伪风险分数的相关性均值：`0.53812`
- 未标注 General 与全量伪风险分数的相关性范围：`0.30203 - 0.66906`
- Top5 Jaccard 稳定性均值：`0.63783`

与 10 锚点设置相比：

- alpha 平均差距从约 `0.018` 提升到约 `0.039`。
- 未标注 General 相关性从约 `0.24` 提升到约 `0.54`。
- Top5 稳定性从约 `0.15` 提升到约 `0.64`。

这说明增加 General 锚点数显著缓解了未标注字段外推漂移。

### 消融实验结果

使用 20 个 General 锚点、seed `0-5`，比较可学习融合、均匀融合和固定单指标融合：

| 策略 | 监督 loss | 全量 General MAE vs pseudo_y | 未标注 General MAE | 未标注相关性 | Top5 稳定性 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learnable | `6.01742e-05` | `0.1070` | `0.1783` | `0.546` | `0.570` |
| uniform | `9.01769e-05` | `0.1074` | `0.1788` | `0.545` | `0.570` |
| Pearson | `3.01603e-05` | `0.1099` | `0.1839` | `0.543` | `0.464` |
| Spearman | `2.57313e-05` | `0.1273` | `0.2134` | `0.441` | `0.381` |
| Kendall | `3.15391e-05` | `0.1234` | `0.2065` | `0.471` | `0.364` |
| NMI | `6.27582e-05` | `0.0962` | `0.1600` | `0.573` | `0.479` |
| dCor | `2.74642e-05` | `0.1317` | `0.2209` | `0.480` | `0.344` |

### 本次结论

1. 将 General 锚点数从 10 提高到 20 是有效的。它明显改善了未标注 General 的预测稳定性，减少了 `pseudo_y` 低但 `y_hat` 被推高的现象。
2. beta/alpha 的分化比 10 锚点设置明显。Kendall 和 Spearman 权重稳定偏高，Pearson 权重稳定偏低。
3. 但是，可学习融合仍然没有明显优于固定均匀融合。learnable 与 uniform 在全量 General MAE、未标注 MAE、未标注相关性和 Top5 稳定性上几乎一致。
4. 固定 NMI 在全量伪风险一致性和未标注相关性上表现最好，但 Top5 稳定性不如 learnable/uniform。这说明 NMI 可能对当前伪标签定义更友好，但不能单独证明 NMI 是全局最优传播指标。
5. 当前最稳妥的判断是：增加锚点数已经明显改善项目可信度；beta 现在能够更明显地分化，但“可学习 beta 比均匀融合更有效”仍需要进一步改目标函数或加入未标注一致性约束来证明。

## 2026-05-20 13:04:55 CST - Modify by GPT5.5

### 为什么必须再次修改

用户指出一个根本性错误：General 字段的推断风险 `q_i` 必须由 DNN 实际预测 Confidential 字段得到，而不能由相关系数、相关矩阵融合分数，或线性回归分数替代。

正确方法应为：

1. 对每个 General 字段 `x_i`。
2. 分别用它作为输入训练 DNN，预测每个 Confidential 字段 `x_c`。
3. 在测试集上计算 `R²(x_i -> x_c)`。
4. 得到 `q_{i,c} = max(0, R²(x_i -> x_c))`。
5. 再聚合：

```text
q_i = 0.7 * max_c(q_{i,c}) + 0.3 * MeanTop3_c(q_{i,c})
```

此前把相关性强度当成推断能力是不成立的，因为相关性只能说明同步变化强弱，不能说明通过一个字段能否在样本外预测另一个字段。

### 本次代码修改

1. 修改 `src/data_processing.py`。
   - 新增 `_predictive_r2_dnn_univariate`。
   - 每个 `(General字段, Confidential字段)` 对应训练一个小型 DNN 回归器。
   - DNN 输入为单个 General 字段，输出为单个 Confidential 字段。
   - 按时间顺序切分数据，前 70% 训练，后 30% 测试。
   - 在测试集上计算 `R²`，并用 `max(0, R²)` 得到 `q_{i,c}`。
   - 用 `0.7 * max + 0.3 * MeanTop3` 聚合得到 General 字段的 `q_i`。

2. 修改 `configs/config.yaml`。
   - 增加 DNN 推断标签相关参数：
     - `predictive_test_ratio: 0.3`
     - `dnn_epochs: 80`
     - `dnn_hidden_dim: 16`
     - `dnn_lr: 0.01`
     - `dnn_weight_decay: 0.0001`
     - `dnn_seed: 2026`

3. 修改 `scripts/run_pipeline.py`。
   - 读取 DNN 推断标签参数。
   - 输出当前 DNN 推断模型设置。
   - 导出 `general_to_confidential_predictive_r2.csv`，记录每个 General 字段到每个 Confidential 字段的 DNN 测试集 `q_{i,c}`。
   - 将原输出中的推断能力列改为 `预测R2推断能力_q`。

### 主流水线测试结果

20 个 General 锚点、seed 42 下，流水线正常运行。最终 alpha 为：

- Pearson: `0.178749`
- Spearman: `0.213471`
- Kendall: `0.221588`
- NMI: `0.192077`
- dCor: `0.194115`

General Top10 为：

1. `total_pjm_rmccp_cr`
2. `forecast_load_mw_latest_available`
3. `forecast_load_mw_day_ahead`
4. `system_energy_price_da`
5. `gen_fuel_gas_mw`
6. `da_as_nsr_mw_primary_reserve`
7. `gen_fuel_coal_mw`
8. `gen_fuel_other_renewables_pct`
9. `gen_fuel_nuclear_pct`
10. `gen_fuel_gas_pct`

### DNN q 矩阵检查

已导出：

```text
outputs/general_to_confidential_predictive_r2.csv
```

该文件每一行是一个 General 字段，每一列 `q_to_xxx` 是它用 DNN 预测某个 Confidential 字段时的测试集 `q_{i,c}`。

聚合 `q_i` 排名前几位：

| General字段 | 聚合推断能力 `q_i` | 最大 `q_{i,c}` |
| --- | ---: | ---: |
| `forecast_load_mw_latest_available` | `0.923848` | `0.988404` |
| `forecast_load_mw_day_ahead` | `0.871647` | `0.924595` |
| `system_energy_price_da` | `0.819270` | `0.954806` |
| `gen_fuel_gas_mw` | `0.722773` | `0.788884` |
| `gen_fuel_coal_mw` | `0.648749` | `0.685509` |
| `gen_fuel_nuclear_pct` | `0.644336` | `0.692701` |

这说明当前 `q` 已经来自 DNN 样本外预测能力，而不是相关系数。

### 多随机种子测试结果

使用 seed `0-5`，learnable 融合的 alpha 均值为：

- Pearson: `0.177162`
- Spearman: `0.207202`
- Kendall: `0.222340`
- NMI: `0.198702`
- dCor: `0.194594`

其他统计：

- alpha 最大最小差距均值：`0.04518`
- 未标注 General MAE vs pseudo_y：约 `0.1815`
- 未标注 General 相关性均值：约 `0.417`
- Top5 Jaccard 稳定性：`0.733`

### 消融实验结果

使用 DNN-q 标签、20 个 General 锚点、seed `0-5`：

| 策略 | 监督 loss | 全量 General MAE vs pseudo_y | 未标注 General MAE | 未标注相关性 | Top5 稳定性 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learnable | `6.34818e-05` | `0.1093` | `0.1815` | `0.417` | `0.733` |
| uniform | `7.46535e-05` | `0.1113` | `0.1849` | `0.418` | `0.733` |
| Pearson | `4.21597e-05` | `0.1417` | `0.2372` | `0.293` | `0.516` |
| Spearman | `3.96433e-05` | `0.1563` | `0.2618` | `0.212` | `0.447` |
| Kendall | `3.70227e-05` | `0.1525` | `0.2554` | `0.162` | `0.641` |
| NMI | `1.12978e-04` | `0.0977` | `0.1613` | `0.480` | `0.501` |
| dCor | `3.48313e-05` | `0.1580` | `0.2648` | `0.220` | `0.483` |

### 本次结论

1. 当前 `q_i` 已经按正确方法由 DNN 推断能力得到，不再由相关系数构造。
2. Kendall 和 Spearman 的 alpha 在 DNN-q 标签下仍稳定偏高，Pearson 稳定偏低。
3. learnable 融合相比 uniform 有轻微改善，但优势仍然不大；二者 Top5 稳定性相同。
4. NMI 单指标在 MAE 和未标注相关性上表现最好，但 Top5 稳定性较低，说明它更贴近当前 DNN-q 伪标签数值，但排序稳定性不如 learnable/uniform。
5. 后续如果要继续证明 beta 的必要性，应该在 DNN-q 标签基础上加入更强的未标注一致性约束，或者增加对 beta 的直接监督/温度控制，而不是再回到相关性伪标签。

## 2026-05-20 13:20:33 CST - Modify by GPT5.5

### 为什么做这次诊断

在 DNN-q 标签已经修正后，用户继续追问为什么 alpha 差别仍然不够大。为判断问题来自“锚点数量不足”“锚点选择方式不合理”“beta 学习率过小”“softmax 温度太高”还是“GNN 本体容量遮蔽 beta”，本次加入 beta 可辨识性诊断实验。

### 本次代码修改

1. 修改 `src/model.py`：
   - 增加 `alpha_temperature`。
   - `alpha = softmax(beta / alpha_temperature)`。
   - 默认 `alpha_temperature=1.0`，保持原模型行为不变。

2. 修改 `src/train.py`：
   - 增加 `beta_lr` 可选参数。
   - 当 `beta_lr` 不为空时，beta 使用单独学习率，其他 GNN 参数仍使用原学习率。
   - 默认 `beta_lr=None`，保持主流水线行为不变。

3. 新增 `scripts/run_beta_diagnostics.py`：
   - 一次性准备 DNN-q 数据。
   - 对比不同 General 锚点数量：`20 / 30 / 40 / 49`。
   - 对比两种锚点选择方式：高低两端 `extreme` 与分位数覆盖 `quantile`。
   - 对比 beta 单独学习率：`0.01 / 0.02 / 0.05`。
   - 对比 softmax temperature：`0.7 / 0.5 / 0.3`。
   - 对比较小 GNN 容量：`hidden_dim=16 / 32`。
   - 结果导出到 `outputs/beta_diagnostics.csv`。

### 关键指标解释

1. `alpha_spread_mean`：
   - 五个 alpha 中最大值减最小值。
   - 越大表示五个相关性权重越不均匀，beta 越“拉开”。

2. `unsup_corr_mean`：
   - 未参与监督的 General 字段中，`y_hat` 与 DNN-q 伪风险 `pseudo_y` 的相关系数。
   - 越高说明模型对未标注字段的外推越贴近 DNN 推断标签。

3. `top5_jaccard_mean`：
   - 不同随机种子下 General Top5 集合的平均 Jaccard 重叠度。
   - 越高说明高风险字段排名越稳定，不太依赖随机初始化。

### 诊断结果摘要

| 实验设置 | alpha spread | 未标注相关性 | Top5 稳定性 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 20锚点 extreme baseline | `0.0479` | `0.510` | `0.667` | 两端锚点，分化较大但外推一般 |
| 20锚点 quantile baseline | `0.0248` | `0.596` | `0.778` | 分位数覆盖更稳，但 alpha 分化小 |
| 30锚点 extreme baseline | `0.0568` | `0.462` | `1.000` | Top5 很稳，但未标注相关性下降 |
| 40锚点 quantile baseline | `0.0485` | `0.594` | `0.778` | 锚点多且覆盖均匀，综合较稳 |
| 20锚点 quantile + beta_lr=0.01 | `0.1255` | `0.594` | `0.778` | beta 明显拉开，外推不变 |
| 20锚点 quantile + beta_lr=0.02 | `0.2585` | `0.601` | `0.778` | 推荐候选，拉开明显且稳定性不降 |
| 20锚点 quantile + beta_lr=0.05 | `0.5895` | `0.607` | `0.587` | alpha 过度尖锐，Top5 稳定性下降 |
| 20锚点 quantile + temp=0.7 | `0.3749` | `0.601` | `0.778` | 推荐候选，拉开明显且稳定性保持 |
| 20锚点 quantile + temp=0.5 | `0.5074` | `0.606` | `0.508` | 过强，Top5 稳定性下降 |
| hidden=16 + beta_lr=0.02 + temp=0.5 | `0.7870` | `0.563` | `0.448` | 容量太小导致 Kendall 过度独大 |
| hidden=32 + beta_lr=0.02 + temp=0.5 | `0.6329` | `0.567` | `0.778` | 分化很强，但未标注相关性不如 64 hidden |

### 本次结论

1. 单纯增加锚点数量不是唯一关键。40 锚点 quantile 的综合表现不错，但 49 锚点没有未标注字段，无法继续评估外推。
2. 锚点选择方式很重要。20 锚点 quantile 的未标注相关性和 Top5 稳定性优于 20 锚点 extreme，说明覆盖完整风险刻度比只选最高/最低更合理。
3. beta 单独学习率有效。`beta_lr=0.02` 可以把 alpha spread 从 `0.0248` 提到 `0.2585`，同时未标注相关性和 Top5 稳定性基本不掉。
4. softmax temperature 也有效，但不能太低。`temp=0.7` 是较稳候选，`temp=0.5/0.3` 容易让 alpha 过度尖锐并损害稳定性。
5. 降低 GNN 容量会让 alpha 更极端，但不一定更好。hidden=16 几乎让 Kendall 独大，Top5 稳定性反而差。
6. 当前最值得进入下一轮正式实验的设置是：
   - `anchor_strategy=quantile`
   - `general_label_count=20` 或 `40`
   - `beta_lr=0.02`
   - `alpha_temperature=0.7`
   - `hidden_dim=64`

## 2026-05-20 14:47:51 CST - Modify by GPT5.5

### 本次工程整理

1. 修改 `scripts/run_pipeline.py` 的输出目录机制。
   - 每次运行不再直接写入 `outputs/` 根目录。
   - 新输出自动写入 `outputs/runYYYYMMDD_HHMMSS/`。
   - 每个 run 目录自动保存 `run_config.md`，记录本次运行使用的关键配置和输出目录。

2. 修改 `configs/config.yaml`。
   - 增加并启用 `training.beta_lr: 0.02`。
   - 增加并启用 `training.alpha_temperature: 0.7`。
   - 主学习率 `training.lr` 保持 `0.002`。

3. 整理历史输出。
   - 将原先散落在 `outputs/` 根目录下的历史文件移动到 `outputs/run2026519/`。
   - 为历史归档补充 `outputs/run2026519/run_config.md`。

4. 新增 `.gitignore`。
   - 忽略 `__pycache__/`、`.DS_Store`、虚拟环境、缓存目录和日志文件。
   - 没有忽略 `outputs/`，因此 run 结果目录会被纳入版本管理。

### 本次重跑结果

使用推荐设置：

- `beta_lr: 0.02`
- `alpha_temperature: 0.7`
- `general_label_count: 20`
- `hidden_dim: 64`
- `DNN-q` 标签

新运行目录：

```text
outputs/run20260520_144617/
```

最终 alpha：

- Pearson: `0.044825`
- Spearman: `0.164577`
- Kendall: `0.658079`
- NMI: `0.081843`
- dCor: `0.050676`

General Top10：

1. `total_pjm_rmccp_cr`
2. `forecast_load_mw_latest_available`
3. `forecast_load_mw_day_ahead`
4. `system_energy_price_da`
5. `gen_fuel_gas_mw`
6. `da_as_nsr_mw_primary_reserve`
7. `gen_fuel_coal_mw`
8. `gen_fuel_other_renewables_pct`
9. `gen_fuel_nuclear_pct`
10. `gen_fuel_gas_pct`

### Git 状态

当前目录 `/Users/haocun/Desktop/AllProjects/GNN_Aggresvation` 不是 git 仓库，没有 `.git` 目录。因此目前无法执行 `git status`、`git commit` 或 `git push`。需要先确认远程仓库地址，或把项目移动/初始化到正确的 git 仓库中。

## 2026-05-20 14:52:26 CST - Modify by GPT5.5

### DNN-q 缓存复用

1. 在 `src/data_processing.py` 中增加 DNN-q 缓存读写逻辑。
   - 默认缓存目录为 `outputs/dnn_q_cache/`。
   - 缓存包含 `dnn_q_scores.csv`、`general_to_confidential_predictive_r2.csv`、`metadata.json` 和 `README.md`。
   - 若字段列表和 DNN-q 参数签名一致，后续运行会直接读取缓存，不再重复训练 General->Confidential 推断 DNN。

2. 在 `configs/config.yaml` 中增加：
   - `pseudo_label.dnn_q_cache_dir: "outputs/dnn_q_cache"`

3. 在 `scripts/run_pipeline.py` 中接入缓存目录。
   - 每个 run 目录仍会保存一份 `general_to_confidential_predictive_r2.csv`，保证单次运行结果完整。
   - 昂贵的 DNN-q 生成步骤会优先复用 `outputs/dnn_q_cache/`。

4. 验证结果。
   - 已生成 `outputs/dnn_q_cache/`。
   - 已重新运行一次主流水线，生成 `outputs/run20260520_145148/`。
   - 第二次运行明显更快，说明 DNN-q 缓存复用生效。
