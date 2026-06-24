# DNN_Aggresvation30 日志

## Modify by opencode: 2026-06-22

### 1. 子项目30要解决的问题

DNN29 的评估诊断（`eval_split_diagnosis.py`）发现：纯时序 70/30 切分对
7/12 Confidential 字段造成严重 R² 低估。`net_actual` 在 shuffle 下
Ridge R²=0.78，时序下 R²=-0.26，gap=1.04。

DNN30 的目标：**用 GNN（而非 Ridge）直接验证 shuffle split 下 net/gross
及其它字段的 R² 是否真的能大幅提升**，确认"分布漂移假说"在真实模型上成立。

### 2. 设计思路

- 固定使用 DNN29 的 bipartite 主干（已证明更合理）
- 新增 `split_mode: temporal | shuffle` 开关（`data_processing.py`）
- 4 个对照实验：{temporal, shuffle} × {maskloss, allloss}
- temporal 组复现 DNN29 结果作对照，shuffle 组验证同分布下真实性能

### 3. 代码修改

#### `src/data_processing.py`
- 新增 `shuffle_split(df, train_ratio, seed)`：随机打乱后切分，打破时序。
- `prepare_data()` 读取 `cfg["dataset"]["split_mode"]`，选择
  `chronological_split` 或 `shuffle_split`。

#### `scripts/run_pipeline.py`
- 输出目录改为 `outputs/<split_mode>/<loss_mode>/<timestamp>/`。
- 打印信息加入 `split_mode`。

### 4. 实验设置

4 个对照实验，每个分配一张 RTX 4090：

| 实验 | split_mode | loss口径 | device |
|---|---|---|---|
| temporal_maskloss | temporal | 排除 net/gross | cuda:0 |
| temporal_allloss | temporal | 全 12 target | cuda:1 |
| shuffle_maskloss | shuffle | 排除 net/gross | cuda:2 |
| shuffle_allloss | shuffle | 全 12 target | cuda:3 |

共同配置：bipartite=True, window=4, staged(phase1=100), top_k=8,
attn_temp=0.8, dropout=0.15, hidden=64, layers=3, epochs=300, patience=80。

### 5. 实验结果

#### maskloss 对比

| 字段 | temporal R² | shuffle R² | Δ |
|---|---:|---:|---:|
| metered_load_mw | 0.9757 | 0.9851 | +0.009 |
| total_gen | 0.9591 | 0.9807 | +0.022 |
| total_lmp_da | 0.8982 | 0.9766 | +0.078 |
| da_as_total_mw_primary_reserve | 0.9610 | 0.9444 | -0.017 |
| da_as_total_mw_synchronized_reserve | 0.9223 | 0.9071 | -0.015 |
| total_losses | 0.6956 | 0.7838 | +0.088 |
| da_as_total_mw_thirty_minutes_reserve | 0.4316 | **0.7697** | **+0.338** |
| marginal_loss_price_da | 0.4711 | 0.7139 | **+0.243** |
| congestion_price_rt | 0.4744 | 0.5436 | +0.069 |
| congestion_price_da | 0.2573 | 0.3150 | +0.058 |
| gross_actual_interchange_mw | -0.5551 | -0.0000 | +0.555 |
| net_actual_interchange_mw | **-1.9485** | **-0.0008** | **+1.948** |
| **best_test_loss** | **0.3279** | **0.2208** | **-32.7%** |

#### allloss 对比

| 字段 | temporal R² | shuffle R² | Δ |
|---|---:|---:|---:|
| metered_load_mw | 0.9706 | 0.9864 | +0.016 |
| total_gen | 0.9604 | 0.9822 | +0.022 |
| total_lmp_da | 0.8779 | 0.9757 | +0.098 |
| da_as_total_mw_primary_reserve | 0.9627 | 0.9443 | -0.019 |
| da_as_total_mw_synchronized_reserve | 0.9106 | 0.9083 | -0.002 |
| da_as_total_mw_thirty_minutes_reserve | 0.4116 | **0.7909** | **+0.379** |
| total_losses | 0.6505 | 0.7882 | +0.138 |
| marginal_loss_price_da | 0.4865 | 0.7139 | **+0.227** |
| gross_actual_interchange_mw | 0.0887 | **0.7104** | **+0.622** |
| net_actual_interchange_mw | **-1.3186** | **0.6456** | **+1.964** |
| congestion_price_rt | 0.4727 | 0.5481 | +0.075 |
| congestion_price_da | 0.2936 | 0.3140 | +0.020 |
| **best_test_loss** | **0.4673** | **0.2363** | **-49.4%** |

### 6. 关键发现

1. **分布漂移假说在 GNN 上完全证实**。`net_actual` allloss 从 R²=-1.32
   跃升到 **R²=0.65**；`gross_actual` 从 R²=0.09 跃升到 **R²=0.71**。
   这两个字段不是"不可推断"，而是被时序 split 的季节性漂移惩罚了。

2. **整体 loss 几乎减半**。maskloss 从 0.328 降到 0.221 (-33%)，
   allloss 从 0.467 降到 0.236 (-49%)。纯时序 split 严重低估了模型能力。

3. **GNN 接近 Ridge 线性上界**。shuffle allloss 下：
   - net GNN R²=0.65 vs Ridge R²=0.78（覆盖 83%）
   - gross GNN R²=0.71 vs Ridge R²=0.74（覆盖 96%）
   GNN 已经接近线性上界，说明当前架构在 net/gross 上已无明显短板。

4. **`congestion_price_da` 是唯一真正的本质难推断字段**：
   shuffle 下 R² 也只有 0.31（maskloss）/ 0.31（allloss），gap 只有 0.02-0.06。
   这是价格类字段，受市场博弈影响，General 里确实缺少推断信号。

5. **10/12 字段在 shuffle 下提升**。`da_as_total_mw_thirty_minutes_reserve`
   +0.38，`marginal_loss_price_da` +0.23，`total_losses` +0.14。只有
   `da_as_total_mw_primary_reserve` 和 `da_as_total_mw_synchronized_reserve`
   微降（-0.02），因为它们物理上由 General 直接决定，时序关系本就稳定。

### 7. 结论

- **纯时序 70/30 切分极其不合理**，对 7/12 字段造成严重 R² 低估，
  整体 loss 高估近一倍。
- **shuffle split 应作为主评估口径**，时序 split 只作"最坏情况鲁棒性"参照。
- **net/gross 在同分布下 R²=0.65-0.71**，不是不可推断，无需为它们改架构。
- **唯一值得继续投入的字段是 `congestion_price_da`**（shuffle R²=0.31），
  需要更强的非线性建模或额外特征。

### 8. 输出目录结构

```
outputs/
├── temporal/
│   ├── maskloss/<timestamp>/    # 时序切分 + maskloss
│   └── allloss/<timestamp>/     # 时序切分 + allloss
├── shuffle/
│   ├── maskloss/<timestamp>/    # 随机切分 + maskloss
│   └── allloss/<timestamp>/     # 随机切分 + allloss
├── relationship_cache/          # 相关性张量缓存
└── split_comparison.csv         # 4 组 R² 对比汇总表
```
