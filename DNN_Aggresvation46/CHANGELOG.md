# DNN_Aggresvation46 日志

## 目标

验证 edge-conditioned value MLP 是否能让边关系特征直接进入 message/value，从而比 DNN43/DNN45
只把关系特征用于 attention score 更有效。

## 设计：边关系如何进入 value

DNN43 的 Confidential 聚合是：

```
v_j = W_v h_j
m_c = sum_j a_cj v_j
```

DNN46 保持 attention score 不变，只改 value 内容。对每条 Confidential target-source 边
构造静态边特征：

```
r_cj = [corr_pearson, corr_spearman, corr_kendall, corr_nmi, corr_distance_corr, prior_cj, log(prior_cj)]
```

每层用一个 edge-value MLP 生成 FiLM 参数：

```
[delta_gamma_cj, delta_bias_cj] = MLP_edge_value_l(r_cj)
```

然后把原 value 改为边条件 value：

```
v_tilde_cj = v_j * (1 + s * tanh(delta_gamma_cj)) + s * tanh(delta_bias_cj)
m_c = sum_j a_cj v_tilde_cj
```

其中 `s=edge_value_scale`，默认 0.25。最后一层 MLP 零初始化，所以训练开始时严格退化为
原 baseline 的 `v_tilde_cj = v_j`，后续再学习边条件修正。

重要边界：

- 只改 Confidential 聚合的 value/message 内容。
- 不改 General-General 聚合。
- 不改 gated dynamic/prior attention score。
- 不引入 MLP 旁路；输出仍然来自图消息传播。

## 实验配置

| 配置 | 说明 |
|---|---|
| `ev00_baseline_g11` | DNN43/DNN45 baseline，`value_mode=source` |
| `ev01_edge_film` | baseline + `value_mode=edge_film` |
| `ev02_edge_film_dynamic_only` | dynamic-only attention score + edge-conditioned value |
| `ev03_edge_film_uniform_alpha` | 固定均匀 alpha + edge-conditioned value |

主要判断：

1. `ev01` 是否超过 `ev00`：edge-conditioned value 是否直接提精度。
2. `ev02` 是否接近或超过 `ev00`：边条件 value 是否能替代 prior score 的小收益。
3. `ev03` 是否修复 DNN45 中 uniform alpha 掉分：raw edge metrics 进 value 后，是否还需要
   learned alpha。

## 结果：2026-06-26 multi 验证

四个配置 4 GPU 并行完成：

| 配置 | 说明 | mean R² | best MSE | 结论 |
|---|---|---:|---:|---|
| `ev00_baseline_g11` | 原 gated dynamic/prior + source value | **0.8648** | **0.1425** | 最好 |
| `ev01_edge_film` | baseline + edge-conditioned value | 0.8584 | 0.1493 | 下降 |
| `ev03_edge_film_uniform_alpha` | 固定均匀 alpha + edge-conditioned value | 0.8548 | 0.1530 | 接近 DNN45 uniform，但没修复 |
| `ev02_edge_film_dynamic_only` | dynamic-only score + edge-conditioned value | 0.8501 | 0.1578 | 最差 |

逐目标看，`ev01_edge_film` 有小幅改善：

- `total_lmp_da`: +0.0053 R²
- `total_gen`: +0.0013 R²
- `metered_load_mw`: +0.0008 R²
- `total_losses`: +0.0013 R²

但关键困难目标掉分更多：

- `congestion_price_da`: −0.0290 R²
- `congestion_price_rt`: −0.0203 R²
- `gross_actual_interchange_mw`: −0.0196 R²
- `marginal_loss_price_da`: −0.0092 R²

结论：

1. 这个 edge-conditioned value FiLM 版本没有提高精度，反而整体降低。
2. 边关系直接调制 value 会帮一部分强物理量字段，但对价格类/交互类引入了噪声。
3. `dynamic_only + edge_film` 没有替代 gated prior，说明 prior score 的小收益不能简单转移到 value。
4. `uniform_alpha + edge_film` 没有修复 DNN45 中 uniform alpha 掉分，raw edge metrics 直接进 value
   不足以替代 learned alpha/gated prior 通道。

这说明“更强边条件 message”不是不能做，但当前 FiLM 方式太直接。若继续走这条线，更合理的下一步
是只对价格类单独做小 adapter，或给 edge_film 加更强正则/更小 scale，而不是全目标共享打开。

汇总文件：

- `outputs/_comparison/edge_value_summary_multi.csv`
- `outputs/_comparison/edge_value_per_target_multi.csv`
