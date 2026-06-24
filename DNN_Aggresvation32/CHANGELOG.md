# DNN_Aggresvation32 日志

## Modify by opencode: 2026-06-22

### 1. 子项目32要解决的问题

DNN31 只用了 net/gross 两个字段初步验证"多目标联合推断不损害单个目标"，
但结论不够稳健。DNN32 做一个完整的大规模对照：

- **multi**：1 个模型同时推断全部 12 个 Confidential 字段
- **single**：12 个独立模型，每次只推断 1 个 Confidential 字段
  （图结构只含 General-General + General→该 Confidential，其它 11 个
  confidential 节点及其连线全部移除）

这是对"多目标联合推断是否会降低每个目标预测质量"的严格检验。

### 2. 设计思路

- 固定 DNN31 最优配置：bipartite + shuffle split + nostaged + allloss
- multi 和 single 使用完全相同的超参数（hidden_dim, lr, epochs 等）
- single 模式下 `confidential` 列表只保留 1 个字段，`build_edge_mask`
  自动构建 General-General + General→该 Confidential 的二分图
- 相关性张量从全图缓存按列子集提取，避免重复计算

### 3. 代码修改

#### `src/data_processing.py`
- `prepare_data()` 读取 `cfg["fields"]["single_target"]`，若有则
  `confidential_columns` 只保留该字段。

#### `scripts/run_pipeline.py`
- 输出目录：`outputs/multi/<timestamp>/` 或
  `outputs/single/<target_name>/<timestamp>/`
- `_load_or_compute_metric_tensor()`：当 current_columns 是缓存
  columns 的子集时，从全图缓存按列索引提取子集。

#### `scripts/scheduler.py`
- 4 GPU 并行调度器：维护待运行队列和 GPU 空闲池，每完成一个 run
  立刻在空出来的 GPU 上启动下一个。13 个 run（1 multi + 12 single）
  分 4 批滚动执行，总耗时约 20 分钟。

### 4. 实验结果

| 字段 | multi R² | single R² | Δ | 谁更好 |
|---|---:|---:|---:|---|
| net_actual_interchange_mw | 0.7427 | **0.8198** | **+0.077** | single↑ |
| gross_actual_interchange_mw | 0.7262 | 0.7351 | +0.009 | ≈ |
| total_gen | 0.9850 | **0.9953** | +0.010 | single↑ |
| metered_load_mw | 0.9886 | 0.9946 | +0.006 | ≈ |
| total_losses | 0.7904 | **0.8268** | **+0.036** | single↑ |
| congestion_price_da | 0.3012 | **0.5171** | **+0.216** | single↑ |
| congestion_price_rt | 0.5543 | 0.5028 | **-0.052** | multi↑ |
| marginal_loss_price_da | 0.7230 | **0.7624** | **+0.039** | single↑ |
| total_lmp_da | 0.9766 | **0.9976** | +0.021 | single↑ |
| da_as_total_mw_primary_reserve | 0.9450 | **0.9921** | **+0.047** | single↑ |
| da_as_total_mw_synchronized_reserve | 0.9110 | **0.9959** | **+0.085** | single↑ |
| da_as_total_mw_thirty_minutes_reserve | 0.7972 | 0.7876 | -0.010 | ≈ |

- delta 统计：mean=+0.040, median=+0.029, min=-0.052, max=+0.216
- **single 更好：8 个**，multi 更好：1 个，持平：3 个
- multi best_test_loss = 0.2250

### 5. 关键发现

1. **多目标联合推断确实会降低大部分单个目标的预测质量**。
   8/12 字段在 single 模式下 R² 更高，平均降 0.04。这推翻了 DNN31
   "net/gross 不损害其它字段"的初步结论——DNN31 只看了 2 个字段，
   DNN32 全量对照后发现影响是普遍的。

2. **影响最大的字段是 `congestion_price_da`**（+0.216），从 0.30 升到
   0.52。这是唯一真正的本质难推断字段，在 multi 模式下被其它 11 个
   target 的梯度严重稀释。

3. **`da_as_total_mw_synchronized_reserve` 受影响也大**（+0.085），
   从 0.91 升到 0.996。这类备用容量字段在 single 模式下几乎完美推断。

4. **只有 `congestion_price_rt` 在 multi 下更好**（-0.052）。可能因为
   它和 `congestion_price_da`、`marginal_loss_price_da` 物理强耦合，
   多目标共享表示反而帮助了它。

5. **高 R² 字段（>0.97）受影响小**（total_gen +0.010, metered_load +0.006,
   total_lmp_da +0.021）。这些字段物理上由 General 直接决定，multi 和
   single 都能很好推断。

### 6. 结论

- **多目标联合推断会普遍降低单个目标 R² 约 0.04**（median），对困难
  字段（congestion_price_da）影响高达 0.22。这是共享表示容量被 12 个
  target 分摊的代价。
- **如果目标是最大化每个 target 的推断精度，应使用 single 模式**。
  但如果目标是快速评估整体风险或做字段间相对比较，multi 模式仍有价值
  （训练 1 次而非 12 次，且整体 loss 可接受）。
- **`congestion_price_da` 在 single 模式下 R²=0.52**，说明它不是完全
  不可推断，而是在 multi 模式下被梯度稀释了。

### 7. 输出目录结构

```
outputs/
├── multi/
│   └── <timestamp>/          # 1 个模型，12 个 target
├── single/
│   ├── net_actual_interchange_mw/<timestamp>/
│   ├── gross_actual_interchange_mw/<timestamp>/
│   ├── total_gen/<timestamp>/
│   ├── ...（共 12 个）
├── relationship_cache/
└── multi_vs_single.csv       # 对比汇总表
```
