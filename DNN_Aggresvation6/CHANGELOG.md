# DNN_Aggresvation6 调优日志

## 2026-05-21 17:10 - Modify by GPT5.5: 创建子项目6，修正弱target-conditioned attention

### 子项目5暴露的问题

子项目5验证了“不同Confidential目标应该使用不同邻居权重”的方向，但第一次实现的
`target_gat` 太弱：`condition_score(target_embedding)` 对同一个目标行里的所有source都是
同一个常数，进入softmax后会被抵消。因此它看起来是target-conditioned，实际主要仍然是
`source_score + target_score + relation_prior`，没有真正表达“这个目标应该关注哪个源节点”。

子项目5测试结果也支持这个判断：

- `target_gat_loss_mask` 的10个有效目标 MSE 为 `0.384630`，弱于DNN4最佳 `gat_loss_mask` 的 `0.362518`。
- `target_gat` 全目标 MSE 为 `0.485694`，也弱于DNN4普通GAT `0.472676`。
- 部分字段如 `congestion_price_da` 有提升，但整体目标被拖累，说明方向可能有效，具体打分形式不够强。

### 修改思路

本子项目不覆盖DNN4/DNN5，而是新增架构 `target_bilinear_gat`：

1. General节点仍然使用GCN传播，保留稳定的信息通路。
2. Confidential节点使用强target-conditioned attention：
   - target端生成 query：`query = W_q_hidden h_target + W_q_id e_target`
   - source端生成 key/value：`key = W_k h_source`, `value = W_v h_source`
   - attention logit 使用双线性交互：`query · key / sqrt(d)`
   - 再加入关系特征bias和图先验bias。
3. 输出层增加 `target_specific_heads`，让每个Confidential字段拥有独立输出头，避免所有目标共用同一回归头。

### 修改内容

- `src/model.py`
  - 新增 `target_bilinear_gat` 架构。
  - 新增 source-key / target-query / value projection。
  - 新增 relation feature bias。
  - 新增 target-specific output heads。
  - `get_mean_attention()` 支持导出新架构注意力诊断。
- `scripts/run_pipeline.py`
  - 支持读取 `attention_dim` 和 `target_specific_heads`。
  - 注意力诊断纳入 `target_bilinear_gat`。
- `configs/config.yaml`
  - 默认改为 `target_bilinear_gat_loss_mask`。
- `configs/target_bilinear_gat.yaml`
  - 全12个Confidential目标参与loss。
- `configs/target_bilinear_gat_loss_mask.yaml`
  - 排除 `net_actual_interchange_mw` 与 `gross_actual_interchange_mw` 的loss贡献。
- `scripts/run_experiments.py`
  - 改为顺序运行DNN6两组核心实验。

### 测试记录

#### 语法检查

```bash
conda run -n Pytorch310_MacBookAir python -m py_compile \
  DNN_Aggresvation6/src/model.py \
  DNN_Aggresvation6/src/train.py \
  DNN_Aggresvation6/src/sensitivity.py \
  DNN_Aggresvation6/scripts/run_pipeline.py \
  DNN_Aggresvation6/scripts/run_experiments.py \
  DNN_Aggresvation6/scripts/compare_runs.py
```

结果：通过。

#### 实验1：`target_bilinear_gat_loss_mask`

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation6/scripts/run_pipeline.py --config DNN_Aggresvation6/configs/target_bilinear_gat_loss_mask.yaml
```

运行目录：

```text
DNN_Aggresvation6/outputs/run_20260521_152412_target_bilinear_gat_loss_mask
```

核心结果：

| 指标 | 数值 |
| --- | ---: |
| best/final test MSE（10个参与loss目标） | 0.348861 |
| final test MSE（12个全部目标） | 0.556023 |
| 被排除2目标 MSE | 1.591833 |

逐目标R²：

| Confidential字段 | R² | 是否参与loss |
| --- | ---: | --- |
| `da_as_total_mw_primary_reserve` | 0.9763 | 是 |
| `metered_load_mw` | 0.9755 | 是 |
| `total_gen` | 0.9635 | 是 |
| `da_as_total_mw_synchronized_reserve` | 0.9464 | 是 |
| `total_lmp_da` | 0.8737 | 是 |
| `total_losses` | 0.5327 | 是 |
| `da_as_total_mw_thirty_minutes_reserve` | 0.4988 | 是 |
| `congestion_price_rt` | 0.4980 | 是 |
| `marginal_loss_price_da` | 0.4251 | 是 |
| `congestion_price_da` | 0.1602 | 是 |
| `gross_actual_interchange_mw` | -0.5567 | 否 |
| `net_actual_interchange_mw` | -1.9254 | 否 |

敏感度Top 10：

| 排名 | General字段 | sensitivity |
| ---: | --- | ---: |
| 1 | `da_as_nsr_mw_primary_reserve` | 0.3318 |
| 2 | `gen_fuel_nuclear_mw` | 0.1577 |
| 3 | `system_energy_price_da` | 0.1517 |
| 4 | `marginal_loss_price_rt` | 0.1312 |
| 5 | `total_lmp_rt` | 0.0937 |
| 6 | `forecast_load_mw_latest_available` | 0.0766 |
| 7 | `rmccp` | 0.0483 |
| 8 | `forecast_load_mw_day_ahead` | 0.0445 |
| 9 | `total_pjm_rmccp_cr` | 0.0207 |
| 10 | `gen_fuel_coal_pct` | 0.0173 |

注意力诊断：

- `total_losses` 的Top attention开始集中到燃料结构、负荷、发电和价格相关字段：
  `gen_fuel_coal_pct`、`forecast_load_mw_latest_available`、`gen_fuel_nuclear_mw`、
  `gen_fuel_coal_mw`、`metered_load_mw`、`total_gen`、`system_energy_price_da`。
- `congestion_price_da` 的Top attention集中在 `system_energy_price_da`、
  `forecast_load_mw_latest_available`、`marginal_loss_price_rt`、`total_lmp_da`、
  `marginal_loss_price_da`。
- `congestion_price_rt` 的Top attention集中在 `gen_fuel_multiple_fuels_mw`、
  `system_energy_price_da`、`total_lmp_rt`、`marginal_loss_price_rt`、
  `total_pjm_rmccp_cr`、`rmccp`。
- attention entropy normalized 多数仍在 `0.90` 左右，说明模型虽然已经学出更合理的排序，
  但并不是极尖锐选择；后续如果要继续调优，可以考虑温度退火、sparsemax/entmax或attention entropy正则。

#### 实验2：`target_bilinear_gat`

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation6/scripts/run_pipeline.py --config DNN_Aggresvation6/configs/target_bilinear_gat.yaml
```

运行目录：

```text
DNN_Aggresvation6/outputs/run_20260521_153002_target_bilinear_gat
```

核心结果：

| 指标 | 数值 |
| --- | ---: |
| best/final test MSE（12个全部目标） | 0.479325 |

逐目标R²：

| Confidential字段 | R² |
| --- | ---: |
| `metered_load_mw` | 0.9795 |
| `da_as_total_mw_primary_reserve` | 0.9731 |
| `da_as_total_mw_synchronized_reserve` | 0.9591 |
| `total_gen` | 0.9345 |
| `total_lmp_da` | 0.9309 |
| `congestion_price_rt` | 0.4916 |
| `marginal_loss_price_da` | 0.4305 |
| `total_losses` | 0.3674 |
| `da_as_total_mw_thirty_minutes_reserve` | 0.3550 |
| `congestion_price_da` | 0.1588 |
| `gross_actual_interchange_mw` | -0.1763 |
| `net_actual_interchange_mw` | -0.7204 |

### 与DNN4/DNN5关键对比

| 实验 | 训练目标MSE | 全量MSE | `total_losses` R² | `congestion_price_rt` R² | `thirty_minutes_reserve` R² | `congestion_price_da` R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DNN4 `gat_loss_mask` | 0.362518 | 0.557638 | 0.4671 | 0.4630 | 0.2920 | 0.2137 |
| DNN5 `target_gat_loss_mask` | 0.384630 | 0.753616 | 0.3218 | 0.4261 | 0.3415 | 0.1926 |
| DNN6 `target_bilinear_gat_loss_mask` | 0.348861 | 0.556023 | 0.5327 | 0.4980 | 0.4988 | 0.1602 |
| DNN5 `target_gat` | 0.485694 | 0.485694 | 0.3529 | 0.4495 | 0.3017 | 0.1502 |
| DNN6 `target_bilinear_gat` | 0.479325 | 0.479325 | 0.3674 | 0.4916 | 0.3550 | 0.1588 |

### 结果分析

1. `target_bilinear_gat_loss_mask` 是目前最好的GNN方案：
   - 10个有效训练目标MSE从DNN4最佳 `0.362518` 降到 `0.348861`。
   - 相比DNN5弱target-conditioned `0.384630`，提升更明显。
   - 说明“target-conditioned attention方向本身可行”，但必须用source-target交互，不能只把target embedding作为行常数加进softmax。

2. 中等R²字段有明显改善：
   - `total_losses`: DNN4 `0.4671` -> DNN6 `0.5327`
   - `congestion_price_rt`: DNN4 `0.4630` -> DNN6 `0.4980`
   - `da_as_total_mw_thirty_minutes_reserve`: DNN4 `0.2920` -> DNN6 `0.4988`
   这正是引入强target-conditioned attention最想解决的部分。

3. `congestion_price_da` 没有改善：
   - DNN4 `gat_loss_mask` 为 `0.2137`
   - DNN6 `target_bilinear_gat_loss_mask` 为 `0.1602`
   attention里它只连接到5个source，且主要是价格/负荷字段。可能问题不是attention表达能力，
   而是当前图构建的邻居候选太窄、DA拥塞字段本身噪声更强，或需要时序/分区/LMP组件信息。

4. 两个interchange字段仍不适合作为主要训练目标：
   - loss-mask版本里它们被排除后，主目标表现最好。
   - 全目标版本里它们虽然负R²有所缓和，但整体MSE和多个中等目标表现下降。
   - 结论仍然是：保留节点可提供图上下文，但不要让它们参与主loss；如果要预测它们，应单独建模。

5. 全目标版本不优于loss-mask版本：
   - `target_bilinear_gat`: MSE `0.479325`
   - `target_bilinear_gat_loss_mask`: 训练目标MSE `0.348861`
   全目标训练会把容量分给低信号interchange字段，损害主要目标。

### 当前建议

短期推荐把DNN6的 `target_bilinear_gat_loss_mask` 作为新的主线结果。它是目前GNN路线下最稳的版本。

如果继续调优，建议优先尝试三件事：

1. 为 `congestion_price_da` 放宽或重建邻居候选，例如提高 `top_k`、降低threshold，或给价格类字段建立专门候选池。
2. 对强attention加温度/稀疏度调优，例如 `attention_temperature: 0.7`、`0.5`，或增加attention entropy正则。
3. 做“目标分组模型”：价格类、reserve类、load/generation类分别训练子模型，避免一个图同时兼顾完全不同的推断关系。
