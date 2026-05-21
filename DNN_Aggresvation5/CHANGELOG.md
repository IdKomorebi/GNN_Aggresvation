# DNN_Aggresvation5 修改与测试日志

## 背景：为什么从子项目4进入方案B

子项目4完成了四组GNN/GAT实验和Route A单目标MLP诊断。结论是：图结构本身不是无效，但“一张共享全局图 + 多目标同时还原”不适合所有Confidential目标。

Route A显示：

- `total_losses` 单目标MLP R²=0.6541，远高于共享GNN/GAT，说明共享图拖累明显。
- `congestion_price_da` 单目标MLP R²=0.0476，反而低于GNN/GAT，说明它需要图结构或间接关系。
- `net/gross_actual_interchange_mw` 单目标MLP约0.25，说明不是完全无信号，而是共享图结构很不适配。

因此，子项目5测试方案B：**target-conditioned GAT**。目标是让边权/attention显式依赖当前要推断的Confidential目标，而不是让所有目标共享同一套边含义。

## 解决思路

普通GAT的问题是：虽然每个目标节点会产生不同attention，但attention logits主要来自节点隐藏状态，目标字段“身份”没有被显式注入。子项目5加入目标字段embedding作为attention条件：

```text
attention(source -> target | target_confidential)
```

实现上采用更保守的混合结构：

1. General节点仍使用GCN聚合，保持全图表示稳定。
2. Confidential节点读取邻居时，使用 target-conditioned attention。
3. 相关性融合邻接矩阵仍作为attention prior。
4. 默认仍排除 `net_actual_interchange_mw`、`gross_actual_interchange_mw` 的训练损失，但保留它们在图中。

---

## 2026-05-21 - Modify by GPT5.5: 初始化target-conditioned GAT

### 修改前问题分析

子项目4的GAT已经能把price类目标的注意力指向更合理的邻居，但提升有限，且注意力熵偏高。核心怀疑是：普通GAT仍然不够“目标条件化”，不能明确表达“同一条边对不同Confidential目标有不同含义”。

### 修改思路

在 `DNN_Aggresvation5` 中复用子项目4代码骨架，新增 `architecture: target_gat`：

- General节点消息：继续使用GCN的行归一化邻接聚合。
- Confidential节点消息：用目标字段embedding参与attention logits。
- Attention prior：继续使用5种相关性指标的可学习alpha融合结果。
- 输出和训练流程沿用子项目4，便于横向对比。

### 修改内容

- 新建 `DNN_Aggresvation5`
- 修改 `src/model.py`
  - 新增 `target_gat` 架构
  - 新增 `attention_temperature`
  - 新增 target-conditioned attention：
    `target_score + condition_score(target_embedding) + source_score + relation_prior`
  - 只替换Confidential节点消息，General节点保持GCN
- 修改 `scripts/run_pipeline.py`
  - 支持 `target_gat`
  - 注意力诊断兼容 `gat` 与 `target_gat`
- 新增配置：
  - `configs/target_gat.yaml`
  - `configs/target_gat_loss_mask.yaml`
  - 默认 `configs/config.yaml` 指向 `target_gat_loss_mask`
- 复用 `DNN_Aggresvation4/outputs/relationship_cache` 作为相关性缓存

### 测试结果

#### 试运行1：发现并修复edge mask失效bug

初次运行：

| 实验 | 运行目录 | 状态 |
|---|---|---|
| `target_gat_loss_mask` | `outputs/run_20260521_123850_target_gat_loss_mask` | 无效 |
| `target_gat` | `outputs/run_20260521_144756_target_gat` | 无效 |

问题：`_target_conditioned_messages()` 中先对 `adjacency_prior` 做了 `clamp(min=1e-12)`，再使用 `prior > 0` 判断有效边，导致所有56个节点都被视为有效邻居，edge mask被绕开。注意力诊断中所有Confidential节点degree都显示为56，这是异常证据。

修复：

```python
prior_raw = adjacency_prior[self.conf_indices]
valid_edges = prior_raw > 0
prior = prior_raw.clamp(min=1e-12)
```

修复后重新运行以下有效实验。

#### 有效实验结果

| 实验 | 运行目录 | loss目标 | 目标MSE | 全目标MSE | 被排除目标MSE |
|---|---|---:|---:|---:|---:|
| `target_gat_loss_mask` | `outputs/run_20260521_145254_target_gat_loss_mask` | 10 | 0.384630 | 0.753616 | 2.598547 |
| `target_gat` | `outputs/run_20260521_150325_target_gat` | 12 | 0.485694 | 0.485694 | N/A |

与子项目4关键结果对比：

| 实验 | 结构 | 10个可推断目标MSE | 全12目标MSE |
|---|---|---:|---:|
| DNN4 `baseline_gcn` | GCN | 0.367448 | 0.481692 |
| DNN4 `gat` | 普通GAT | 0.372783 | **0.472676** |
| DNN4 `gat_loss_mask` | 普通GAT + loss-mask | **0.362518** | 0.557638 |
| DNN5 `target_gat` | target-conditioned GAT | 约0.3679 | 0.485694 |
| DNN5 `target_gat_loss_mask` | target-conditioned GAT + loss-mask | 0.384630 | 0.753616 |

逐字段R²对比：

| 字段 | DNN4 baseline_gcn | DNN4 gat_loss_mask | DNN5 target_gat | DNN5 target_gat_loss_mask |
|---|---:|---:|---:|---:|
| `congestion_price_da` | 0.1990 | **0.2137** | 0.1502 | 0.1926 |
| `congestion_price_rt` | **0.4765** | 0.4630 | 0.4495 | 0.4261 |
| `total_losses` | 0.4482 | **0.4671** | 0.3529 | 0.3218 |
| `marginal_loss_price_da` | 0.4198 | **0.5011** | 0.4837 | 0.4510 |
| `da_as_total_mw_thirty_minutes_reserve` | **0.3871** | 0.2920 | 0.3017 | 0.3415 |
| `total_lmp_da` | 0.8843 | **0.9234** | 0.9177 | 0.9179 |
| `total_gen` | 0.9154 | **0.9379** | 0.9124 | 0.9338 |
| `metered_load_mw` | 0.9567 | 0.9602 | **0.9586** | 0.9524 |
| `da_as_total_mw_primary_reserve` | 0.9751 | 0.9735 | 0.9547 | **0.9756** |
| `da_as_total_mw_synchronized_reserve` | 0.9082 | **0.9403** | 0.8806 | 0.9394 |
| `net_actual_interchange_mw` | -0.9610 | -1.8833 | **-0.7988** | -4.1125 |
| `gross_actual_interchange_mw` | 0.0049 | -0.4121 | **0.1768** | -1.0924 |

### 结果分析

1. **当前target-conditioned GAT没有超过子项目4普通GAT**  
   `target_gat_loss_mask` 的10目标MSE为0.384630，明显差于 DNN4 `gat_loss_mask` 的0.362518，也差于 DNN4 `baseline_gcn` 同口径0.367448。  
   `target_gat` 的全目标MSE为0.485694，略差于 DNN4 `baseline_gcn` 的0.481692，更差于 DNN4 `gat` 的0.472676。

2. **目标条件化方向合理，但当前注入方式无效**  
   当前实现把 target embedding 作为一个额外标量项加到attention logits：

   ```text
   target_score + condition_score(target_embedding) + source_score + relation_prior
   ```

   这个形式太弱。因为 `condition_score(target_embedding)` 对同一个目标的所有source都是同一个常数，它不会改变该目标内部不同source的相对排序；softmax对同一行加常数几乎没有效果。也就是说：虽然代码上“注入了目标身份”，但它没有真正学习 `source-target` 交互。

   更正确的target-conditioned attention应该是：

   ```text
   score(source, target) = MLP([h_source, h_target, e_target, relation_features])
   ```

   或至少使用双线性/点积交互：

   ```text
   score(source, target) = (W_s h_source)^T (W_t e_target)
   ```

3. **attention邻居语义合理，但性能不够**  
   修复后attention degree恢复正常，例如 `congestion_price_da` degree=5，`congestion_price_rt` degree=10。Top邻居也符合业务语义：
   - `congestion_price_da`: `system_energy_price_da`、`forecast_load_mw_latest_available`、`marginal_loss_price_rt`、`total_lmp_da`
   - `congestion_price_rt`: `total_lmp_rt`、`marginal_loss_price_rt`、`system_energy_price_da`
   - `total_losses`: `forecast_load`、`gen_fuel_nuclear_mw`、`gen_fuel_coal_pct`

   但R²没有改善，说明“看起来合理的邻居排序”不等于有效预测。当前attention对数值拟合帮助不足。

4. **loss-mask在target_gat里更不稳定**  
   `target_gat_loss_mask` 让被排除的 `net/gross_actual_interchange_mw` 预测彻底崩坏：被排除目标MSE=2.598547，`net_actual_interchange_mw` R²=-4.1125，`gross_actual_interchange_mw` R²=-1.0924。  
   这说明在当前target_gat结构下，排除loss会让这些节点的表示漂移，进而可能干扰图中其他节点。

### 阶段结论

方案B的“目标条件化”概念仍然值得保留，但当前DNN5第一版实现没有成功。失败原因不是数据，而是目标条件注入方式太弱：给每个目标加一个source无关的condition标量，不能真正改变边的相对权重。

当前结论：

- 不建议把 DNN5 `target_gat` 作为下一步主线。
- DNN4 `gat_loss_mask` 仍是目前最好的GAT方向。
- Route A 单目标MLP仍然是最清楚的诊断参照。

### 下一步建议

如果继续做方案B，应该改成真正的source-target交互，而不是当前的加性常数条件：

1. **双线性target-conditioned attention**
   ```text
   q_target = W_q e_target
   k_source = W_k h_source
   score = q_target · k_source + relation_bias
   ```

2. **MLP边打分器**
   ```text
   score = MLP([h_source, h_target, e_target, r_source_target])
   ```

3. **目标专属输出头**
   当前所有Confidential共用同一个 `output_head`，这也会造成目标间冲突。下一版应尝试每个目标一个小head，或按业务组共享head。

4. **优先做Route C分组模型**
   从工程投入/收益看，Route C可能比继续改DNN5更稳：price/loss、load/gen、reserve、interchange分组后，每组内部关系更一致，目标冲突更小。
