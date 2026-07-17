# DNN_Aggresvation45 日志

## 目标

验证 DNN43 graph-combo 中看起来“有效但不可解释”的部分到底是否真的贡献精度：

- `baseline_g11`: 复现 DNN43 最优 multi graph-combo。
- `uniform_alpha`: 固定 5 个相关性指标权重为均匀 `0.2`，验证 learned alpha 是否必要。
- `dynamic_only`: 去掉 prior 分数，只保留动态 attention，验证 prior 是否主要只是 mask。
- `prior_only`: 去掉动态 attention，只用相关性 prior，验证静态相关性先验单独是否足够。

## 初始判断

DNN43 的诊断显示 learned prior 与简单平均 prior 几乎等价，且 gate 明显偏动态 attention。
因此 DNN45 优先做消融验证，而不是继续调多头或强行让 alpha 更稀疏。

## 结果：2026-06-26 multi 消融验证

四个配置使用同一份 DNN43 `g11_graphcombo` 基线设置，4 GPU 并行完成：

| 配置 | 说明 | mean R² | best MSE | 结论 |
|---|---|---:|---:|---|
| `ab00_baseline_g11` | learned alpha + gated dynamic/prior | **0.8663** | **0.1410** | 仍是最好 |
| `ab02_dynamic_only` | 只用动态 attention，prior 只作 mask | 0.8625 | 0.1449 | 非常接近 baseline |
| `ab01_uniform_alpha` | alpha 固定均匀，但保留 gated prior | 0.8538 | 0.1540 | 明显下降 |
| `ab03_prior_only` | 只用相关性 prior，不用动态 attention | 0.8538 | 0.1541 | 明显下降 |

关键解释：

1. **learned alpha 本身仍然不可解释**：baseline 的 `alpha_general` 最终仍接近均匀
   `[0.217, 0.195, 0.199, 0.190, 0.199]`。
2. **动态 attention 是主力**：dynamic-only 只比 baseline 低 0.0038 mean R²，说明大部分
   推断能力来自节点表示和动态 q-k 注意力。
3. **prior 分支有小但真实的收益**：baseline 比 dynamic-only 略高，主要差异在
   `total_losses`、`gross_actual_interchange_mw`、reserve 类等目标；但 prior 单独不够。
4. **固定均匀 alpha 会伤价格类**：`uniform_alpha` 对 `congestion_price_da` 比 baseline
   低 0.071，对 `marginal_loss_price_da` 低 0.038。这说明 alpha 不适合拿来解释，但 learned
   alpha/可学习 prior 通道仍提供了一点优化自由度。

结论：DNN43 的“相关性权重解释”确实站不住；但 gated prior 通道不是完全没用，它提供
小幅精度收益。下一步若继续提精度，应把边关系特征直接送进 message/value，例如
edge-conditioned value MLP，而不是继续强行让 alpha 变稀疏。

汇总文件：

- `outputs/_comparison/ablation_summary_multi.csv`
- `outputs/_comparison/ablation_per_target_multi.csv`
