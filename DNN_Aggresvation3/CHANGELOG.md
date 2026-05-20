# DNN_Aggresvation3 修改与测试日志

## 动机：子项目2遗留的根本性方法论缺陷

子项目2虽然在工程层面完成了"先预训练相关性权重、再固定边权GNN聚合"的完整流水线，但在方法论上暴露出四个不可忽视的根本性缺陷：

### 缺陷1：主观规则的层层堆叠

推断能力 $q_i = 0.7 \times \max_c(q_{i,c}) + 0.3 \times \text{MeanTop3}_c(q_{i,c})$ 的聚合公式是人工设定的；伪标签的融合规则（推断能力分数 + 初始敏感度传播分数的加权）也是人工设定的；锚点的数量、选择策略、高低比例、阈值等全是手动超参。整个敏感度评估系统更像一套精心编排的启发式评分体系加了个GNN的壳，而不是从数据中自主发现知识的端到端学习系统。

### 缺陷2：循环论证（Circular Reasoning）

我们人为定义图的构建规则，又用图中的元素（如边权、传播分数）构造伪标签，然后让GNN去拟合这些伪标签，最后评估GNN是否接近了我们自己定义的值。GNN在这个闭环中并没有学到任何我们事先不知道的东西。

### 缺陷3：锚点依赖暴露模型脆弱性

如果只用Confidential标签=1.0来训练，所有General节点的预测分数都会漂到极高值，模型完全无法收敛到合理状态。我们被迫注入高风险锚点和低风险锚点（双端校准）来"拽住"GNN的输出范围。这些人工干预的必要性本身就说明GNN并非从物理规律中自然学到了敏感度含义，而是被我们用规则硬拽到了合理区间。

### 缺陷4："我本来就能算"悖论

为了训练GNN，我们必须给一部分General字段计算DNN推断分数q作为伪标签；为了评估GNN的外推能力，我们又要对剩余General字段计算q值做对照。但这些q值我们用单特征DNN探针本来就能全部算出来。GNN的角色沦为了"猜测我们故意不告诉它但自己已经知道的答案"，这在科学上是站不住脚的。

---

## 子项目3的核心转变：推断即监督（Inference-as-Supervision）

### 核心思路

彻底抛弃伪标签体系，回归到敏感度问题的物理本质：

> **一个General字段的敏感度 = 它帮助攻击者还原Confidential字段真实值的能力**

我们拥有8000多行Confidential字段的真实时序数据。这些真实值本身就是最天然、最权威的监督信号。不需要任何人工公式来定义"敏感度"。

### 旧范式 vs 新范式

| | 子项目1&2（旧范式） | 子项目3（新范式） |
|---|---|---|
| GNN的任务 | 拟合人工定义的伪标签 | **还原Confidential的真实值** |
| 监督信号 | 伪标签q（人工公式算出来） | **Confidential的真实时序数据** |
| 敏感度来源 | GNN的直接输出 | 训练后的**遮蔽实验**副产品 |
| 是否需要锚点 | 必须，否则训不出来 | **不需要** |
| 是否循环论证 | 是 | **否** |

### 具体方案

1. **数据组织**：每个时间步构成一个训练样本。General节点的输入为其标准化真实值，Confidential节点的输入为零（模拟攻击者视角）。监督目标为Confidential的标准化真实值。
2. **图构建**：计算5种相关性指标（Pearson, Spearman, Kendall, NMI, dCor），融合权重α作为可学习参数端到端优化。
3. **GNN训练**：批量化训练（batch_size=128），损失函数为纯还原MSE。
4. **敏感度提取**：训练完成后，逐一遮蔽每个General字段，测量Confidential还原误差的增量。增量越大 = 该字段对还原Confidential越重要 = 隐性泄露风险越高。

---

## 2026-05-21 00:00 CST - 首次测试（基线）

### 测试配置

- `hidden_dim: 32`, `num_layers: 2`, `dropout: 0.05`
- `lr: 0.001`, `batch_size: 128`, `patience: 20`
- `graph.top_k: 10`, `graph.threshold: 0.08`
- `seed: 42`

### 测试结果

- 运行目录: `outputs/run_20260520_235002`
- 有效时间步: 6556（删除NaN后）
- 训练集: 4589行, 测试集: 1967行
- 图边数: 2296 条 (密度: 0.7321)
- 最优测试损失(MSE): **0.7770**
- 最优轮数: Epoch 6（Epoch 26早停）
- 学到的α: Kendall=0.5278, NMI=0.1990, Spearman=0.0993, dCor=0.0923, Pearson=0.0816

逐Confidential字段R²:
| 字段 | R² |
|---|---|
| total_lmp_da | 0.7268 |
| da_as_total_mw_primary_reserve | 0.7039 |
| metered_load_mw | 0.6923 |
| total_gen | 0.5536 |
| total_losses | 0.4790 |
| marginal_loss_price_da | 0.3903 |
| da_as_total_mw_synchronized_reserve | 0.2873 |
| congestion_price_rt | 0.1954 |
| congestion_price_da | 0.1545 |
| gross_actual_interchange_mw | 0.0607 |
| da_as_total_mw_thirty_minutes_reserve | -0.7842 |
| net_actual_interchange_mw | -2.2023 |

敏感度Top 5:
| 排名 | 字段 | 敏感度 |
|---|---|---|
| 1 | total_pjm_assigned_reg | 0.0243 |
| 2 | total_pjm_reg_purchases | 0.0215 |
| 3 | forecast_load_mw_latest_available | 0.0176 |
| 4 | gross_inadv_interchange_mw | 0.0170 |
| 5 | forecast_load_mw_day_ahead | 0.0167 |

### 问题分析

1. **图密度过高 (0.73)**：56个节点中2296条边，几乎是全连接图。GNN无法从图结构中获得有效的归纳偏置，所有节点都能直接通信，图的拓扑约束形同虚设。
2. **模型过早过拟合**：Epoch 6就达到最优，说明学习率0.001偏大，模型在前几轮快速拟合训练集后就开始退化。
3. **测试MSE=0.777过高**：标准化数据的MSE应远低于1.0（1.0意味着预测不比直接用均值好）。当前模型的还原能力严重不足。
4. **敏感度辨识力不足**：最高敏感度仅0.024，意味着遮蔽任何单个字段对还原误差的影响都极其微弱。模型没有学到足够精细的字段依赖关系。

---

## 2026-05-21 01:39 CST - Modify by Claude Opus4.6: 第一轮超参数调优

### 需要解决的问题

基线测试暴露四个核心问题：图密度过高(0.73)、模型过早过拟合(Epoch 6)、测试MSE过高(0.777)、敏感度辨识力不足(最高0.024)。

### 修改思路

1. **降低图密度**：top_k 从10→5，threshold 从0.08→0.25，迫使图结构更稀疏，让GNN真正利用拓扑约束
2. **降低学习率**：lr 从0.001→0.0003，避免模型过早到达局部最优
3. **增加模型容量**：hidden_dim 从32→48，num_layers 从2→3
4. **增加正则化**：dropout 从0.05→0.1
5. **延长训练**：epochs 从200→500，patience 从20→50

### 修改内容

- 修改 `configs/config.yaml` 中的上述超参数

### 测试结果

- 运行目录: `outputs/run_20260521_013915`
- 图边数: 816 条 (密度: **0.2602**，较基线0.7321大幅下降)
- 最优测试损失(MSE): **0.5966**（较基线0.7770下降23.2%）
- 训练轮数: 完整运行500轮，无过早停止
- 学到的α: dCor=0.2990, Kendall=0.2592, NMI=0.2419, Pearson=0.1599, Spearman=0.0400

逐Confidential字段R²:
| 字段 | R²（基线） | R²（本轮） | 变化 |
|---|---|---|---|
| total_gen | 0.5536 | **0.8624** | ↑ +0.31 |
| metered_load_mw | 0.6923 | **0.8535** | ↑ +0.16 |
| da_as_total_mw_primary_reserve | 0.7039 | **0.7934** | ↑ +0.09 |
| total_lmp_da | 0.7268 | 0.6580 | ↓ -0.07 |
| da_as_total_mw_synchronized_reserve | 0.2873 | **0.5754** | ↑ +0.29 |
| total_losses | 0.4790 | 0.2675 | ↓ -0.21 |
| net_actual_interchange_mw | -2.2023 | **-0.7339** | ↑ +1.47 |

敏感度Top 5:
| 排名 | 字段 | 敏感度（基线） | 敏感度（本轮） |
|---|---|---|---|
| 1 | da_as_as_req_mw_synchronized_reserve | - | **0.2506** |
| 2 | da_as_as_req_mw_primary_reserve | - | **0.2371** |
| 3 | da_as_as_req_mw_thirty_minutes_reserve | - | **0.2368** |
| 4 | da_as_nsr_mw_primary_reserve | - | **0.1066** |
| 5 | gen_fuel_nuclear_mw | - | **0.0927** |

### 分析

**显著改善**：
- 测试MSE从0.777降至0.597，下降23%
- total_gen和metered_load_mw的R²突破0.85，还原精度优异
- 敏感度辨识力提升10倍（最高从0.024到0.251），字段间区分度大幅增强
- 模型能够完整训练500轮不再过早停止

**仍存在的问题**：
1. **测试MSE仍为0.597**，在标准化空间中这意味着模型只解释了约40%的方差，还原能力仍有较大改善空间
2. **部分Confidential字段R²为负**（net_actual_interchange_mw: -0.73），说明模型对这些字段的预测比直接用均值还差
3. **total_losses的R²下降**：从0.479降至0.268，说明稀疏图可能切断了部分重要连接
4. **α权重仍在变化**：500轮训练后α仍未完全收敛，学习率对α来说可能仍偏大
5. **训练损失和测试损失差距较大**（0.324 vs 0.597），存在一定的过拟合

---

## 2026-05-21 01:48 CST - Modify by Claude Opus4.6: v2 结构性改进

### 需要解决的问题

1. 训练-测试gap过大（0.324 vs 0.597），泛化不足
2. 测试MSE=0.597仍偏高
3. α权重500轮后仍未收敛

### 修改思路

超参数调优不足以解决泛化gap问题，需要模型结构层面的改进：

1. **BatchNorm**：每层消息传递后添加，稳定隐藏表示分布，减少内部协变量偏移
2. **残差连接**：h_{l+1} = h_l + BN(ReLU(msg_l))，防止深层梯度退化
3. **可学习Confidential嵌入**：替代固定零向量，让模型学习Confidential节点的初始表示
4. **双层MLP输出头**：增加输出端的非线性拟合能力
5. **CosineAnnealing学习率调度器**：平滑降低学习率帮助α稳定收敛
6. **增大正则化**：weight_decay 0.0001→0.0005, dropout 0.1→0.15

### 修改内容

- `src/model.py`：重写为v2架构（BatchNorm + 残差连接 + 可学习嵌入 + 双层输出头）
- `src/train.py`：添加CosineAnnealingLR调度器
- `configs/config.yaml`：hidden_dim=64, dropout=0.15, weight_decay=0.0005, lr=0.0005, epochs=800, patience=80

### 测试结果

- 运行目录: `outputs/run_20260521_014833`
- 最优测试损失(MSE): **0.4817**（较v1的0.5966再降19.3%，较基线0.7770下降38.0%）
- 训练轮数: Epoch 228最优，Epoch 308早停
- 学到的α: **NMI=0.4342**（主导），Kendall=0.1654, Pearson=0.1683, dCor=0.1331, Spearman=0.0990

逐Confidential字段R²:
| 字段 | R²（v1） | R²（v2） | 变化 |
|---|---|---|---|
| da_as_total_mw_primary_reserve | 0.7934 | **0.9751** | ↑↑ +0.18 |
| metered_load_mw | 0.8535 | **0.9567** | ↑ +0.10 |
| total_gen | 0.8624 | **0.9154** | ↑ +0.05 |
| da_as_total_mw_synchronized_reserve | 0.5754 | **0.9082** | ↑↑ +0.33 |
| total_lmp_da | 0.6580 | **0.8843** | ↑↑ +0.23 |
| congestion_price_rt | 0.2875 | **0.4765** | ↑ +0.19 |
| total_losses | 0.2675 | **0.4482** | ↑ +0.18 |
| marginal_loss_price_da | 0.3694 | **0.4198** | ↑ +0.05 |
| da_as_total_mw_thirty_minutes_reserve | -0.0086 | **0.3871** | ↑↑ +0.40 |
| congestion_price_da | 0.1356 | 0.1990 | ↑ +0.06 |
| gross_actual_interchange_mw | 0.0565 | 0.0049 | ↓ -0.05 |
| net_actual_interchange_mw | -0.7339 | -0.9610 | ↓ -0.23 |

敏感度Top 5:
| 排名 | 字段 | 敏感度 |
|---|---|---|
| 1 | da_as_nsr_mw_primary_reserve | 0.1754 |
| 2 | gen_fuel_nuclear_mw | 0.1734 |
| 3 | forecast_load_mw_latest_available | 0.1322 |
| 4 | system_energy_price_da | 0.0830 |
| 5 | total_lmp_rt | 0.0714 |

### 分析

**巨大进步**：
- 5个Confidential字段R²超过0.88，其中da_as_total_mw_primary_reserve达到0.975（几乎完美还原）
- BatchNorm+残差连接显著缩小了训练-测试gap
- α稳定收敛到NMI主导（0.43），与v1的dCor主导不同，说明v2学到了更好的相关性表示
- 敏感度排名在物理上合理：nuclear_mw和forecast_load是公认的高关联字段

**仍存在的问题**：
1. **net_actual_interchange_mw的R²为-0.96**：这个字段在所有版本中都无法还原，可能是因为实际交换电量的波动性极高且与General字段的统计关联较弱，或者是该字段的信息主要来源于其他已被删除的字段
2. **gross_actual_interchange_mw的R²接近0**：同属交换类字段，类似问题
3. **整体MSE=0.48仍有改善空间**：被这两个"硬骨头"字段拖累了平均值

---

## 2026-05-21 03:49 CST - Modify by Claude Opus4.6: v3 滑动窗口实验（失败）

### 需要解决的问题

1. 每个时间步只有1个标量作为节点特征，信息密度低
2. SmoothL1(Huber Loss)替代MSE，增强对不可推断字段的鲁棒性

### 修改思路

引入滑动窗口特征：每个节点在时间步t的输入从单标量 $x_i(t)$ 扩展为向量 $[x_i(t-5), ..., x_i(t)]$（window_size=6）。同时将损失函数改为SmoothL1。

### 修改内容

- `src/model.py`：input_proj从 `Linear(1, H)` 改为 `Linear(input_dim, H)`，添加`input_dim`参数
- `src/train.py`：添加 `_build_windowed_samples()` 函数构建窗口化训练样本
- `src/sensitivity.py`：适配窗口化输入
- `configs/config.yaml`：`window_size: 6`, `loss_fn: smooth_l1`

### 测试结果

- 运行目录: `outputs/run_20260521_034910`
- **所有12个Confidential字段 R² = 1.0000**
- 测试MSE = 0.000000
- α权重 = [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]（完全均匀，从未变化）
- 所有General字段敏感度 = 0.0000

### ⚠️ 失败原因分析

这是一个**严重的数据泄漏/过拟合问题**。滑动窗口让每个节点拥有6维特征（最近6个时间步的值），总共44×6=264维General输入去预测12维Confidential输出。在标准化的电网时序数据中，相邻时间步具有极高的自相关性。模型只需学到"当前时刻的General值的线性组合≈当前时刻的Confidential值"这一trivial解，就可以在训练集和测试集上都完美拟合。

关键证据：α权重始终保持均匀（0.2），说明模型**完全没有利用图结构**——所有信息在input_proj层就已经足够了。GNN的消息传递完全是多余的。

**教训**：在时序数据中使用滑动窗口时，时间自相关性会让推断任务变得trivially easy。我们的目标不是测量"能否在同一数据集内做时序预测"（那当然可以），而是测量"给定一个时间点的General字段值，能否推断出同一时间点的Confidential字段值"。后者才是隐性泄露风险的正确定义。

**决策**：**放弃滑动窗口方案**，回退到v2的单标量输入。继续在v2架构基础上通过其他方式改进。

---

## 2026-05-21 04:00 CST - Modify by Claude Opus4.6: 关键Bug修复

### 发现的Bug

v3滑动窗口实验中所有R²=1.0的**真正原因**不是滑动窗口本身，而是 `_build_windowed_samples()` 中的一个**严重数据泄漏Bug**：

```python
window = data_matrix[i : i + w, :]   # 这是view，不是copy！
feat = window.T                       # 仍然是view
feat[confidential_indices, :] = 0.0   # 这直接修改了原始data_matrix！
targets[i] = data_matrix[i + w - 1, confidential_indices]  # 读到的已经是被清零的值！
```

`window.T`创建的是numpy的view而非独立副本。对view的inplace修改 `feat[...] = 0.0` 直接破坏了原始 `data_matrix` 中Confidential列的数据。随后读取targets时，Confidential列已被清零，导致targets全为0。模型学会了"什么都预测为0"就能得到完美R²。

### 修复方案

```python
targets[i] = data_matrix[i + w - 1, confidential_indices].copy()  # 先取目标
window = data_matrix[i : i + w, :].copy()  # .copy()！
```

### 验证

修复后使用window_size=1重新运行，结果与v2完全一致（MSE=0.4817），证实修复正确。

### 教训

**在numpy中，切片操作返回的是view不是copy。任何对切片的inplace修改都会污染原始数据。** 这是一个经典的numpy陷阱。



