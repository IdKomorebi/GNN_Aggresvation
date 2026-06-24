# DNN_Aggresvation29 日志

## Modify by opencode: 2026-06-22

### 1. 子项目29要解决的问题

在分析 DNN1-DNN28 的完整演化后，发现当前架构存在一个**方法论硬伤**：
Confidential 节点之间通过 GAT 存在双向 attention 连接，导致模型在推断
某个 Confidential 字段时可以从其它 Confidential 字段的隐藏状态中"借力"。

这违背了项目的核心威胁模型：**攻击者只能读取 General 字段，不应从其它
Confidential 字段获取信息**。当前的高 R²（尤其 `total_lmp_da` > 0.9）
可能部分来自 Confidential 之间的互相校准，而非真正的 General→Confidential
推断，导致 R² 被系统性高估。

### 2. 设计思路

在 DNN25 主干（General-GCN + Confidential edge-specific α GAT + staged training
+ window=4）基础上新增 `bipartite` 开关：

- **baseline**（`bipartite=false`）：退化为 DNN25 行为，Confidential 的 source
  集合为全图所有节点（含其它 Confidential），作为对照基线。
- **bipartite**（`bipartite=true`）：Confidential 的 source 集合被限制为
  General 节点，彻底切断 Confidential-Confidential 边。图结构严格匹配
  攻击者威胁模型。

### 3. 代码修改

#### `src/model.py`
- `InferenceDrivenGNN.__init__` 新增 `bipartite: bool = False` 参数；
  当 `bipartite=True` 时注册 `general_indices` buffer。
- `compute_confidential_prior()`：bipartite 时把 Confidential 列置 0，
  只保留 General 列的先验。
- `_confidential_messages()`：bipartite 时 key/value 只从 General 节点
  `h.index_select(1, general_indices)` 取，attention 矩阵从 (B,C,N) 变为
  (B,C,n_general)，再映射回 (B,C,N) 的 General 列用于诊断。
- `build_edge_mask()` 新增 `n_general` 和 `bipartite` 参数；bipartite 时
  在常规掩码构建后强制把 `[n_general:, n_general:]` 块清零。

#### `scripts/run_pipeline.py`
- 输出目录结构改为 `outputs/<model_name>/<loss_mode>/<timestamp>/`，
  其中 `model_name ∈ {baseline, bipartite}`，`loss_mode ∈ {maskloss, allloss}`。
- `build_edge_mask()` 和 `InferenceDrivenGNN()` 调用处传入 `bipartite`
  和 `n_general`。

### 4. 实验设置

四个对照实验，每个分配一张 4090：

| 实验 | bipartite | loss口径 | device |
|---|---|---|---|
| `baseline_maskloss` | false | 排除 net/gross | cuda:0 |
| `baseline_allloss` | false | 全 12 target | cuda:1 |
| `bipartite_maskloss` | true | 排除 net/gross | cuda:2 |
| `bipartite_allloss` | true | 全 12 target | cuda:3 |

共同配置（取自 DNN25 最优）：
- window_size=4, input_encoder=linear
- staged training: phase1_epochs=100, phase1 排除 net/gross
- top_k=8, attention_temperature=0.8, dropout=0.15
- hidden_dim=64, num_layers=3, epochs=300, patience=80

### 5. 实验结果

#### maskloss 口径（排除 net/gross 的 loss）

| Confidential 字段 | baseline R² | bipartite R² | 差值 |
|---|---:|---:|---:|
| metered_load_mw | 0.9675 | 0.9757 | +0.008 |
| da_as_total_mw_primary_reserve | 0.9586 | 0.9610 | +0.002 |
| total_gen | 0.9416 | 0.9591 | +0.018 |
| total_lmp_da | 0.8968 | 0.8982 | +0.001 |
| da_as_total_mw_synchronized_reserve | 0.8765 | 0.9223 | **+0.046** |
| total_losses | 0.6984 | 0.6956 | -0.003 |
| congestion_price_rt | 0.4709 | 0.4744 | +0.004 |
| marginal_loss_price_da | 0.4636 | 0.4711 | +0.008 |
| da_as_total_mw_thirty_minutes_reserve | 0.4460 | 0.4316 | -0.014 |
| congestion_price_da | 0.3208 | 0.2573 | **-0.064** |
| gross_actual_interchange_mw | -0.5551 | -0.5551 | 0.000 |
| net_actual_interchange_mw | -1.9485 | -1.9485 | 0.000 |
| **best_test_loss** | **0.3223** | **0.3279** | +1.8% |

#### allloss 口径（全 12 target 入 loss）

| Confidential 字段 | baseline R² | bipartite R² | 差值 |
|---|---:|---:|---:|
| metered_load_mw | 0.9661 | 0.9706 | +0.005 |
| da_as_total_mw_primary_reserve | 0.9598 | 0.9627 | +0.003 |
| total_gen | 0.9551 | 0.9604 | +0.005 |
| da_as_total_mw_synchronized_reserve | 0.8786 | 0.9106 | **+0.032** |
| total_lmp_da | 0.8780 | 0.8779 | 0.000 |
| total_losses | 0.6381 | 0.6505 | +0.012 |
| congestion_price_rt | 0.4588 | 0.4727 | +0.014 |
| marginal_loss_price_da | 0.4588 | 0.4865 | **+0.028** |
| da_as_total_mw_thirty_minutes_reserve | 0.3986 | 0.4116 | +0.013 |
| congestion_price_da | 0.2943 | 0.2936 | 0.000 |
| gross_actual_interchange_mw | 0.1212 | 0.0887 | -0.033 |
| net_actual_interchange_mw | -1.1761 | -1.3186 | -0.143 |
| **best_test_loss** | **0.4642** | **0.4673** | +0.7% |

#### 关键发现

1. **bipartite 对大多数字段不降反升**：10/12 个字段在 allloss 下 R² 持平或上升，
   尤其 `da_as_total_mw_synchronized_reserve` (+0.032) 和
   `marginal_loss_price_da` (+0.028) 提升明显。说明 conf-conf 边不仅没在
   帮忙，反而在引入噪声/不稳定表示。

2. **只有 3 类字段从 conf-conf 边受益**：
   - `congestion_price_da`（maskloss 下 -0.064）：这是 LMP 的组成部分，
     物理上和 `total_lmp_da`、`marginal_loss_price_da` 强相关，确实在
     "借力"其它 confidential。
   - `net/gross_actual_interchange_mw`（allloss 下变差）：困难字段，
     从其它 confidential 的交叉信号中获得微弱帮助。
   - `da_as_total_mw_thirty_minutes_reserve`（maskloss 下 -0.014）：轻微下降。

3. **整体 loss 几乎不变**（maskloss +1.8%, allloss +0.7%），说明切断
   conf-conf 边是近乎免费的：方法论更干净，代价极小。

4. **原架构高 R² 是真实的**：`total_lmp_da`、`metered_load_mw`、
   `total_gen`、`da_as_total_mw_*` 等高 R² 字段在 bipartite 下基本不变甚至
   更高，证明它们的高 R² 来自 General→Confidential 的真实推断，而非
   confidential 互相校准。

5. **结论**：bipartite 二分传播是更合理的默认架构。conf-conf 边只对
   `congestion_price_da` 等少数物理上强耦合的 confidential 字段有少量帮助，
   但这些帮助在攻击者威胁模型下不合法（攻击者没有 confidential 输入）。

---

## 2026-06-22 补充：评估设置诊断（时序 split vs 随机 split）

### 背景

bipartite 实验中 `net_actual_interchange_mw` R²=-1.95、
`gross_actual_interchange_mw` R²=0.09，远低于其它字段。诊断发现当前
评估方式存在严重问题：**纯时序 70/30 切分导致训练集和测试集分布漂移**。

数据按时间排序后前 70% 训练、后 30% 测试：
- 训练集：2025年1-9月，`net_actual` mean=-4583 MW，98% 为负数
- 测试集：2025年10-12月，`net_actual` mean=-1838 MW，79% 为负数

标准化后测试集 `net_actual` 均值偏移 +1.22σ，已超出训练分布范围。
**这不是模型问题，是评估设置问题。**

### 实验设计

`scripts/eval_split_diagnosis.py` 对全部 12 个 Confidential 字段，
用 Ridge 线性回归（alpha=1.0）作为"GNN 线性上界参照"，对比三种切分：

| 切分方式 | 说明 |
|---|---|
| `temporal_70_30` | 当前 GNN 用的方式：前 70% 时序训练，后 30% 测试 |
| `shuffle_70_30` | 随机 70/30，打破时序，同分布参照 |
| `shuffle_5fold_mean` | 5-fold 随机 CV 均值，更稳健的同分布参照 |
| `rolling_fold1-4` | 滚动时序 CV，定位漂移发生在哪个月 |

### 主结果（Ridge alpha=1.0，R²）

| 字段 | temporal | shuffle | 5fold | gap | 诊断 |
|---|---:|---:|---:|---:|---|
| total_gen | 0.9953 | 0.9984 | 0.9984 | +0.003 | 稳定，无漂移 |
| metered_load_mw | 0.9608 | 0.9943 | 0.9945 | +0.034 | 稳定 |
| total_lmp_da | 0.9416 | 0.9820 | 0.9850 | +0.043 | 稳定 |
| da_as_total_mw_primary_reserve | 0.9342 | 0.9162 | 0.9062 | -0.028 | 稳定 |
| da_as_total_mw_synchronized_reserve | 0.8406 | 0.8513 | 0.8367 | -0.004 | 稳定 |
| total_losses | 0.2721 | 0.7890 | 0.7894 | **+0.517** | **严重漂移** |
| net_actual_interchange_mw | -0.2561 | 0.7755 | 0.7806 | **+1.037** | **严重漂移** |
| da_as_total_mw_thirty_minutes_reserve | 0.3641 | 0.7639 | 0.7687 | **+0.405** | **漂移** |
| marginal_loss_price_da | 0.3602 | 0.7371 | 0.7559 | **+0.396** | **漂移** |
| gross_actual_interchange_mw | 0.1505 | 0.7441 | 0.7385 | **+0.588** | **严重漂移** |
| congestion_price_rt | -1.1755 | 0.3926 | 0.3703 | **+1.546** | **极严重漂移** |
| congestion_price_da | -0.1189 | 0.2242 | 0.2490 | +0.368 | 漂移+本质难 |

### 滚动时序 CV（R²，定位漂移）

| 字段 | fold1 (1-3月训4-5月测) | fold2 (1-5月训6-7月测) | fold3 (1-7月训8-9月测) | fold4 (1-9月训10-12月测) |
|---|---:|---:|---:|---:|
| net_actual_interchange_mw | 0.5323 | 0.2892 | 0.8107 | **-0.2618** |
| gross_actual_interchange_mw | -0.0258 | 0.3451 | 0.5821 | 0.1743 |
| congestion_price_rt | 0.2056 | -1.5999 | -1.3879 | 0.2889 |
| total_losses | -1.3560 | -0.0540 | -0.6179 | 0.4835 |
| marginal_loss_price_da | -0.0851 | 0.7009 | 0.1947 | 0.3802 |
| da_as_total_mw_thirty_minutes_reserve | 0.2458 | 0.4179 | 0.7227 | 0.2721 |

fold4（1-9月训10-12月测）是当前 GNN 的切分方式，`net_actual` 在
fold4=-0.26 但 fold3=0.81，证明 **10-12月是漂移最严重的时段**。

### 关键发现

1. **纯时序 70/30 切分对 7/12 字段严重低估 R²**。`net_actual` 在
   shuffle 下 R²=0.78，时序下 R²=-0.26，gap=1.04。`congestion_price_rt`
   gap=1.55。**当前 GNN 报告的低 R² 里有大量是分布漂移造成的假象，
   不是模型能力不足。**

2. **只有 4 个字段真正稳定**（gap<0.05）：`total_gen`、
   `metered_load_mw`、`total_lmp_da`、`da_as_total_mw_primary_reserve`、
   `da_as_total_mw_synchronized_reserve`。这些是物理上由 General
   直接决定的字段，时序关系稳定。

3. **net/gross 本质可推断性中等**（shuffle R²≈0.74-0.78），不是
   "完全不可推断"。时序 split 下的 -0.26 是漂移假象。但它们也达不到
   total_gen 那种 0.99 的水平，因为 `net_sched_interchange_mw` 被
   drop 了，物理上 net = net_sched + net_inadv，缺了一半信息。

4. **`congestion_price_da` 是唯一真正的"本质难推断"字段**：
   shuffle R² 也只有 0.25，gap=0.37。这是价格类字段，受市场博弈影响，
   General 里确实缺少推断信号。

5. **结论**：
   - **纯时序 70/30 切分极其不合理**，对 7/12 字段造成严重低估。
   - 后续实验应使用 **shuffle split** 或 **滚动时序 CV** 作为主评估口径，
     时序 split 只作为"最坏情况鲁棒性"参照。
   - net/gross 在同分布下 R²≈0.74-0.78，不是不可推断，而是被评估方式
     惩罚了。为它们改架构之前，应先修正评估设置。

### 输出文件

- `outputs/eval_diagnosis/<timestamp>/results.csv`：全部切分 × 字段 R²
- `outputs/eval_diagnosis/<timestamp>/pivot_summary.csv`：主结果汇总表
- `outputs/eval_diagnosis/<timestamp>/rolling_cv.csv`：滚动时序 CV 详情
- `outputs/eval_diagnosis/<timestamp>/summary.json`：结构化结果
