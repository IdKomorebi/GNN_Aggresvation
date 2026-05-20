# 修改日志

## 2026-05-20

### 新建子项目

- 创建 `DNN_Aggresvation2` 子项目，用于实现“先预训练相关性指标权重，再固定边权进行 GNN 聚合”的风险聚合流程。
- 默认使用 `NEW_PROJECT` 中的 V2 处理后数据：
  `NEW_PROJECT/data/processed/data2025_Processed_V2/pjm_rto_hourly_2025_aligned_processed_one_header.csv`。
- 项目结构包括：
  - `configs/data2025_v2_fixed_edge.yaml`
  - `scripts/run_fixed_edge_pipeline.py`
  - `src/correlation_metrics.py`
  - `src/models.py`
  - `src/node_features.py`
  - `src/pipeline.py`
  - `src/utils.py`

### 预训练相关性指标权重

- 参考 `NEW_PROJECT` 中的改进门控网络思路，实现 correlation-gated DNN。
- 每个 confidential 目标单独训练一组相关性指标权重：
  \[
  \alpha_c=\mathrm{softmax}(\beta_c)
  \]
- 每个 general 字段的 gate 由其与目标字段的相关性向量生成：
  \[
  z_{i,c}=\sum_k \alpha_{c,k}r_{i,c}^{(k)},\quad
  g_{i,c}=\sigma(a_c z_{i,c}+b_c)
  \]
- 对所有 confidential 目标的权重按验证集 \(R^2\) 加权平均，得到全局相关性权重。

### 固定边权 GNN

- 使用预训练得到的全局相关性权重构造固定边权。
- GNN 阶段只训练模型参数，不更新边权。
- 默认设置：
  - `general_anchor_count: 10`
  - `label_mode: sensitivity`
  - `include_initial_sensitivity: true`
  - `graph.learn_edge_weights: false`

### 运行调整

- 根据测试情况从默认相关性指标中移除 `hsic`，避免全字段关系矩阵计算过慢。
- 增加 `fields.drop_constant_columns: true`，默认删除全零/常量字段。
- 该调整修复了首次测试中未监督节点被常量字段异常顶高的问题。

### 测试记录

- 首次测试 run：
  `outputs/run_20260520_172339`
  - 问题：未监督榜首出现全零常量列，例如 `da_as_ircmwt2_*` 和 `gen_fuel_storage_pct`。
  - 结论：该结果不能作为有效敏感度判断依据。

- 修复后测试 run：
  `outputs/run_20260520_173235`
  - 字段数：56
  - confidential 字段数：12
  - general 字段数：44
  - GNN 监督节点数：22，其中 12 个 confidential，10 个 general anchor。
  - 最终 GNN loss：约 `0.0002405`。

### 修复后未监督 general 排名前列

修复后未监督 general 字段的高分项主要包括：

- `net_inadv_interchange_mw`
- `gen_fuel_storage_mw`
- `total_pjm_rmpcp_cr`
- `total_pjm_assigned_reg`
- `rmpcp`
- `total_pjm_self_sched_reg`
- `da_as_nsr_mw_primary_reserve`
- `total_pjm_reg_purchases`
- `total_pjm_loc_credit`

这些字段大多属于 interchange deviation、fuel component、settlement 或 ancillary service 相关字段，
从业务语义上看比第一次测试中的全零常量列更合理。

### 当前观察

- confidential 节点基本被稳定推到 0.99 以上。
- general anchor 的预测分数与测得的 \(q\) 标签大体一致。
- 未监督 general 节点中出现了一批高分字段，说明固定边权 GNN 能把风险从 confidential 和 anchor 节点传播到未直接监督的字段。
	- 但未监督 general 的整体分数仍偏高，后续建议增加 held-out anchor 验证：从 10 个 anchor 中留出 2-3 个不监督，只用于检查 GNN 对未监督节点的校准能力。

### 2026-05-20 后续校准修改

- 增加 held-out anchor 验证脚本：
  `scripts/validate_heldout_anchors.py`。
- 验证结果显示：GNN 对 held-out anchor 的 MAE 约为 `0.0787`，但未监督 general 的整体分数仍偏高，非 anchor general 的平均分约 `0.6966`，90 分位约 `0.9423`。
- 将 anchor 机制从单端 top high-risk 选择改为双端选择：
  - `high_anchor_count: 10`
  - `low_anchor_count: 5`
- 低风险 anchor 不再直接设为 0，而是先运行同样的 single-feature probe。
  只有同时满足：
  - 固定图上的 `anchor_score <= low_anchor_max_graph_score`
  - probe 生成的 `q_label <= low_anchor_max_probe_q`
  才作为低风险校准 anchor 进入 GNN 监督。
- `general_anchor_labels.csv` 新增字段：
  - `anchor_role`
  - `anchor_score`
  - `raw_probe_q_label`
  - `use_in_gnn`
  - `label_source`
- 默认数据路径更新为仓库内已有的：
  `../data/Processed/pjm_rto_hourly_2025_cleaned.csv`。
- 新增复用脚本：
  `scripts/run_gnn_from_pretrained.py`。
  当相关性张量和全局相关权重已经存在时，可以跳过 DNN 预训练，只重跑固定图、双端 anchor、probe 标签与 GNN。
- `compute_anchor_labels` 支持复用旧 run 的 `general_anchor_target_r2.csv`。
  因此已有 high anchors 的 probe 结果可以直接复用，只为新增 low anchors 补算 probe。
