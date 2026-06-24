# DNN_Aggresvation32 走读报告

## 子项目定位

DNN32 对"多目标联合推断 vs 单目标独立推断"做完整 12 字段对照。
固定 DNN31 最优配置（bipartite + shuffle + nostaged + allloss）。

---

## 实验设置

- **multi**：1 个模型同时推断 12 个 Confidential（56 节点图）
- **single**：12 个独立模型，每次只推断 1 个 Confidential（45 节点图，
  只含 General-General + General→该 Confidential）
- 共 13 个 run，4 GPU 滚动调度，总耗时约 20 分钟

---

## 核心结果

| 字段 | multi R² | single R² | Δ |
|---|---:|---:|---:|
| net_actual_interchange_mw | 0.74 | **0.82** | +0.08 |
| gross_actual_interchange_mw | 0.73 | 0.74 | +0.01 |
| total_gen | 0.99 | **1.00** | +0.01 |
| metered_load_mw | 0.99 | 0.99 | +0.01 |
| total_losses | 0.79 | **0.83** | +0.04 |
| **congestion_price_da** | **0.30** | **0.52** | **+0.22** |
| congestion_price_rt | 0.55 | 0.50 | -0.05 |
| marginal_loss_price_da | 0.72 | **0.76** | +0.04 |
| total_lmp_da | 0.98 | **1.00** | +0.02 |
| da_as_total_mw_primary_reserve | 0.95 | **0.99** | +0.05 |
| da_as_total_mw_synchronized_reserve | 0.91 | **1.00** | +0.08 |
| da_as_total_mw_thirty_minutes_reserve | 0.80 | 0.79 | -0.01 |

- **8/12 字段 single 更好**，1 个 multi 更好，3 个持平
- median delta = +0.029（single 平均比 multi 高 0.03 R²）

---

## 结论

1. **多目标联合推断会普遍降低单个目标 R²**，median 降 0.03，困难字段
   降 0.22。DNN31 "net/gross 不损害其它字段"的结论是抽样偏差——
   全量对照后影响是普遍的。

2. **`congestion_price_da` 在 single 下 R²=0.52**（vs multi 0.30），
   说明它不是完全不可推断，而是被多目标梯度稀释。

3. **如果目标是最大化每个 target 的推断精度，应使用 single 模式**。
   multi 模式适合快速整体风险评估或字段间相对比较。
