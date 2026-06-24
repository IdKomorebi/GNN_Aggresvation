# DNN_Aggresvation30 走读报告

## 子项目定位

DNN30 验证 DNN29 评估诊断的发现：**纯时序 70/30 切分是 net/gross 低 R²
的主因，而非模型能力不足**。在 DNN29 bipartite 主干上新增 shuffle split，
用 GNN 直接对比 temporal vs shuffle 的 R²。

---

## 改动

相对 DNN29 只改了两个文件：

1. `src/data_processing.py`：新增 `shuffle_split()`，通过
   `cfg["dataset"]["split_mode"]` 选择 `temporal` 或 `shuffle`。
2. `scripts/run_pipeline.py`：输出目录改为
   `outputs/<split_mode>/<loss_mode>/<timestamp>/`。

模型结构、训练逻辑、bipartite 设置完全沿用 DNN29。

---

## 实验设置

4 个对照，各分配一张 RTX 4090：

| 实验 | split_mode | loss口径 |
|---|---|---|
| temporal_maskloss | temporal | 排除 net/gross |
| temporal_allloss | temporal | 全 12 target |
| shuffle_maskloss | shuffle | 排除 net/gross |
| shuffle_allloss | shuffle | 全 12 target |

---

## 核心结果

### net/gross 的 R² 跃升

| 字段 | temporal R² | shuffle R² | Δ | Ridge上界 |
|---|---:|---:|---:|---:|
| net_actual (allloss) | **-1.32** | **0.65** | **+1.96** | 0.78 |
| gross_actual (allloss) | **0.09** | **0.71** | **+0.62** | 0.74 |

shuffle split 下 GNN 接近 Ridge 线性上界，证明 net/gross **不是不可推断**，
而是被时序漂移惩罚了。

### 整体 loss

| 口径 | temporal | shuffle | 改善 |
|---|---:|---:|---:|
| maskloss | 0.328 | 0.221 | **-33%** |
| allloss | 0.467 | 0.236 | **-49%** |

### 各字段 R²（allloss，按 shuffle R² 降序）

| 字段 | temporal | shuffle |
|---|---:|---:|
| metered_load_mw | 0.97 | 0.99 |
| total_gen | 0.96 | 0.98 |
| total_lmp_da | 0.88 | 0.98 |
| da_as_total_mw_primary_reserve | 0.96 | 0.94 |
| da_as_total_mw_synchronized_reserve | 0.91 | 0.91 |
| da_as_total_mw_thirty_minutes_reserve | 0.41 | 0.79 |
| total_losses | 0.65 | 0.79 |
| marginal_loss_price_da | 0.49 | 0.71 |
| gross_actual_interchange_mw | 0.09 | 0.71 |
| net_actual_interchange_mw | -1.32 | 0.65 |
| congestion_price_rt | 0.47 | 0.55 |
| congestion_price_da | 0.29 | 0.31 |

---

## 结论

1. **纯时序 70/30 切分极其不合理**，整体 loss 高估近一倍，net/gross
   R² 低估 1.0-2.0。
2. **shuffle split 应作为主评估口径**。
3. **net/gross 无需改架构**：shuffle 下 R²=0.65-0.71，已接近 Ridge 上界。
4. **唯一本质难推断字段是 `congestion_price_da`**（shuffle R²=0.31）。
