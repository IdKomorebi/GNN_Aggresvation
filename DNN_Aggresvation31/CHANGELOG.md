# DNN_Aggresvation31 日志

## Modify by opencode: 2026-06-22

### 1. 子项目31要解决的问题

DNN30 证明 shuffle split 下 net/gross R² 大幅提升（0.65/0.71），时序漂移
是低 R² 主因。现在需要回答两个问题：

1. **staged training 是否还有必要？** 它是为时序 split 下 net/gross 早期
   大误差淹没其它 target 而设计的。shuffle split 下 net/gross 不再有
   分布漂移，staged 可能多余甚至有害。
2. **加入 net/gross 到 loss 是否损害其它 10 个字段？** 这同时回答一个
   更深的问题：多目标联合推断是否会降低每个目标的预测质量。

### 2. 设计思路

2×2 因子实验，固定 bipartite=True + shuffle split：

| | maskloss（排除 net/gross） | allloss（全 12 target） |
|---|---|---|
| staged | staged_maskloss | staged_allloss |
| nostaged | nostaged_maskloss | nostaged_allloss |

### 3. 代码修改

沿用 DNN30 全部 src（bipartite + shuffle split）。仅改
`scripts/run_pipeline.py` 输出目录为
`outputs/<staging_mode>/<loss_mode>/<timestamp>/`。

### 4. 实验结果

#### 表1: staged vs nostaged（allloss，shuffle split）

| 字段 | staged R² | nostaged R² | Δ |
|---|---:|---:|---:|
| metered_load_mw | 0.9864 | 0.9886 | +0.002 |
| total_gen | 0.9822 | 0.9850 | +0.003 |
| total_lmp_da | 0.9757 | 0.9766 | +0.001 |
| da_as_total_mw_primary_reserve | 0.9443 | 0.9450 | +0.001 |
| da_as_total_mw_synchronized_reserve | 0.9083 | 0.9110 | +0.003 |
| da_as_total_mw_thirty_minutes_reserve | 0.7909 | 0.7972 | +0.006 |
| total_losses | 0.7882 | 0.7904 | +0.002 |
| net_actual_interchange_mw | 0.6456 | **0.7427** | **+0.097** |
| gross_actual_interchange_mw | 0.7104 | 0.7262 | +0.016 |
| marginal_loss_price_da | 0.7139 | 0.7230 | +0.009 |
| congestion_price_rt | 0.5481 | 0.5543 | +0.006 |
| congestion_price_da | 0.3140 | 0.3012 | -0.013 |
| **best_test_loss** | **0.2363** | **0.2250** | **-4.8%** |

#### 表2: maskloss vs allloss（nostaged，shuffle split）

看加入 net/gross 后其它 10 个稳定字段是否受损：

| 字段 | maskloss R² | allloss R² | Δ |
|---|---:|---:|---:|
| metered_load_mw | 0.9851 | 0.9886 | +0.004 |
| total_gen | 0.9807 | 0.9850 | +0.004 |
| total_lmp_da | 0.9766 | 0.9766 | 0.000 |
| da_as_total_mw_primary_reserve | 0.9444 | 0.9450 | +0.001 |
| da_as_total_mw_synchronized_reserve | 0.9071 | 0.9110 | +0.004 |
| total_losses | 0.7838 | 0.7904 | +0.007 |
| da_as_total_mw_thirty_minutes_reserve | 0.7697 | 0.7972 | +0.028 |
| marginal_loss_price_da | 0.7139 | 0.7230 | +0.009 |
| congestion_price_rt | 0.5436 | 0.5543 | +0.011 |
| congestion_price_da | 0.3150 | 0.3012 | -0.014 |
| **10字段均值** | **0.7920** | **0.7972** | **+0.005** |
| net_actual_interchange_mw | -0.001 | 0.7427 | — |
| gross_actual_interchange_mw | -0.000 | 0.7262 | — |

#### 表3: 2×2 汇总（best_test_loss）

| | maskloss | allloss |
|---|---:|---:|
| staged | 0.2208 | 0.2363 |
| nostaged | 0.2208 | **0.2250** |

### 5. 关键发现

1. **staged training 应取消**。nostaged 在 11/12 字段上持平或更好，
   尤其 `net_actual` 从 0.65 升到 **0.74**（+0.097）。staged 把 net/gross
   延迟 100 epoch 引入，在 shuffle split 下 net/gross 不再有漂移大误差，
   反而少了 100 epoch 训练量。**staged 是为时序 split 设计的，shuffle split
   下完全多余且有害。**

2. **加入 net/gross 不损害其它字段**。10 个稳定字段均值 R² 从 0.7920
   微升到 0.7972（+0.005）。最大单项变化是
   `da_as_total_mw_thirty_minutes_reserve` +0.028（反而提升）。
   只有 `congestion_price_da` 微降 -0.014。
   **多目标联合推断不会降低单个目标的预测质量。**

3. **nostaged allloss 是最优配置**（best_test_loss=0.2250），比
   staged allloss（0.2363）好 4.8%，比 staged maskloss（0.2208）
   在 net/gross 上有巨大提升（0→0.74/0.73）。

4. **staged maskloss = nostaged maskloss**（0.2208）。因为 maskloss 下
   staged 的 phase1 排除 net/gross，phase2 也排除 net/gross（loss口径
   不变），staged 等于没做任何事，所以两者完全相同。

### 6. 结论

- **取消 staged training**：shuffle split 下无分布漂移，staged 多余且
  对 net/gross 有害（-0.097 R²）。
- **allloss 无副作用**：加入 net/gross 不损害其它 10 个字段，反而微升。
  多目标联合推断不会降低单个推断质量。
- **推荐默认配置**：bipartite + shuffle split + nostaged + allloss。
