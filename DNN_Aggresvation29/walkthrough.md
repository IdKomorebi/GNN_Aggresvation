# DNN_Aggresvation29 走读报告

## 子项目定位

DNN29 是在 DNN25 主干上验证 **bipartite 二分传播** 的实验：彻底切断
Confidential-Confidential 边，使图结构严格匹配攻击者威胁模型。

---

## DNN13 与 DNN25 的区别（背景回顾）

太多子项目容易混淆，这里简要说明这两个主干的关系：

| 维度 | DNN13 | DNN25 |
|---|---|---|
| **图主干** | General-GCN + Confidential edge-specific α GAT | 相同 |
| **window_size** | 1（单时刻） | 4（最近 4 时刻序列输入） |
| **input_encoder** | 固定 Linear | 可选 linear / mlp |
| **staged_training** | 无 | 有（phase1 排除 net/gross，phase2 全目标） |
| **edge_mask 选择** | 固定 top_k | 支持 adaptive_threshold_topk（min_k/max_k） |
| **loss 加权** | 均匀 | 支持 uncertainty weighting + difficulty weighting |
| **定位** | 确立 General-GCN + Conf-edge-α-GAT 主干 | 在主干上叠加时序窗口+staged training，是最强 allloss 基线 |

简言之：**DNN13 确立了图结构，DNN25 在其基础上加了时间维度和训练策略。**
DNN29 继承 DNN25 的全部能力，只新增 `bipartite` 开关。

---

## 代码改动

核心改动在 `src/model.py`，新增 `bipartite: bool` 参数：

1. `build_edge_mask(..., n_general, bipartite)`：bipartite 时把
   `[n_general:, n_general:]` 块清零，从图结构层面切断 conf-conf。
2. `compute_confidential_prior()`：bipartite 时把 Confidential 列置 0，
   只保留 General 列的先验。
3. `_confidential_messages()`：bipartite 时 key/value 只从 General 节点取
   (`h.index_select(1, general_indices)`)，attention 矩阵从 (B,C,N) 变为
   (B,C,n_general)。

`scripts/run_pipeline.py` 改动：输出目录从 `outputs/run_<timestamp>_<name>/`
改为 `outputs/<model_name>/<loss_mode>/<timestamp>/`，其中
`model_name ∈ {baseline, bipartite}`，`loss_mode ∈ {maskloss, allloss}`。

---

## 实验设置

4 个对照实验，每个分配一张 RTX 4090：

| 实验 | bipartite | loss口径 | device |
|---|---|---|---|
| baseline_maskloss | false | 排除 net/gross | cuda:0 |
| baseline_allloss | false | 全 12 target | cuda:1 |
| bipartite_maskloss | true | 排除 net/gross | cuda:2 |
| bipartite_allloss | true | 全 12 target | cuda:3 |

共同配置：window=4, staged(phase1=100), top_k=8, attn_temp=0.8, dropout=0.15,
hidden=64, layers=3, epochs=300, patience=80。

---

## 实验结果

### maskloss 对比

| 字段 | baseline | bipartite | Δ |
|---|---:|---:|---:|
| metered_load_mw | 0.9675 | 0.9757 | +0.008 |
| da_as_total_mw_primary_reserve | 0.9586 | 0.9610 | +0.002 |
| total_gen | 0.9416 | 0.9591 | +0.018 |
| total_lmp_da | 0.8968 | 0.8982 | +0.001 |
| da_as_total_mw_synchronized_reserve | 0.8765 | **0.9223** | **+0.046** |
| total_losses | 0.6984 | 0.6956 | -0.003 |
| congestion_price_rt | 0.4709 | 0.4744 | +0.004 |
| marginal_loss_price_da | 0.4636 | 0.4711 | +0.008 |
| da_as_total_mw_thirty_minutes_reserve | 0.4460 | 0.4316 | -0.014 |
| congestion_price_da | 0.3208 | 0.2573 | **-0.064** |
| **best_test_loss** | **0.3223** | **0.3279** | +1.8% |

### allloss 对比

| 字段 | baseline | bipartite | Δ |
|---|---:|---:|---:|
| metered_load_mw | 0.9661 | 0.9706 | +0.005 |
| da_as_total_mw_primary_reserve | 0.9598 | 0.9627 | +0.003 |
| total_gen | 0.9551 | 0.9604 | +0.005 |
| da_as_total_mw_synchronized_reserve | 0.8786 | **0.9106** | **+0.032** |
| total_lmp_da | 0.8780 | 0.8779 | 0.000 |
| total_losses | 0.6381 | 0.6505 | +0.012 |
| congestion_price_rt | 0.4588 | 0.4727 | +0.014 |
| marginal_loss_price_da | 0.4588 | **0.4865** | **+0.028** |
| da_as_total_mw_thirty_minutes_reserve | 0.3986 | 0.4116 | +0.013 |
| congestion_price_da | 0.2943 | 0.2936 | 0.000 |
| gross_actual_interchange_mw | 0.1212 | 0.0887 | -0.033 |
| net_actual_interchange_mw | -1.1761 | -1.3186 | -0.143 |
| **best_test_loss** | **0.4642** | **0.4673** | +0.7% |

---

## 结论

1. **切断 conf-conf 边对大多数字段不降反升**（10/12 字段持平或上升）。
   conf-conf 边不仅没在帮忙，反而在引入噪声。

2. **只有 `congestion_price_da` 和 net/gross interchange 从 conf-conf 边
   受益**。这些字段在物理上和其它 confidential 强耦合（LMP 组成部分、
   交换功率），确实在"借力"。但这种借力在攻击者威胁模型下不合法。

3. **整体 loss 代价极小**（+1.8% maskloss, +0.7% allloss）。

4. **原架构高 R² 是真实的**：`total_lmp_da` (0.88)、`metered_load_mw`
   (0.97)、`total_gen` (0.96) 等在 bipartite 下不变甚至更高，证明它们
   来自 General→Confidential 的真实推断。

5. **建议将 bipartite 作为默认架构**：方法论更干净，代价极小，且
   α 权重解释性更清晰（只描述 General→Confidential 路径）。
