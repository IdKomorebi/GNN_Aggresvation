# DNN_Aggresvation4 修改与测试日志

## 子项目3遗留问题

子项目3完成了从“伪标签敏感度评分”到“推断即监督”的关键转变：GNN不再拟合人工规则，而是直接用General字段还原Confidential字段真实值，再通过遮蔽实验得到General字段敏感度。这个方法论比子项目2更干净，也已经在 `run_20260521_040031` 中形成可复现基线。

但子项目3仍有三个主要问题：

1. **消息传递仍是静态均匀聚合**：当前模型只学习5种相关性指标的全局融合权重 alpha，再对固定边掩码内的邻居做行归一化加权。它不能针对不同Confidential目标学习不同邻居的重要性。例如推断 `congestion_price_da` 时，模型无法显式学会更关注price类General邻居。
2. **不可推断字段仍参与训练损失**：`net_actual_interchange_mw` 和 `gross_actual_interchange_mw` 在子项目3中R²长期接近0或为负，但训练代码仍然把它们纳入MSE。这会让模型把容量浪费在不可完成任务上，并拖累中等R²字段。
3. **滑动窗口路线已证明不适合当前目标**：v3窗口实验暴露过numpy view inplace bug；修复后虽然回到正常结果，但时序窗口容易把任务变成时序预测/自相关利用，不适合作为“同一时刻General推断Confidential”的隐性泄露评估主线。

## 解决思路

子项目4不继续污染子项目3，而是把子项目3的v2/v4有效模型作为基线，独立测试两类结构性改进：

1. **loss-mask**：把不可推断的 `net_actual_interchange_mw`、`gross_actual_interchange_mw` 从训练损失和遮蔽敏感度损失口径中排除，但仍保留在图中作为节点参与消息传递和相关性结构。
2. **GAT化注意力聚合**：在固定边掩码内学习目标节点到邻居节点的注意力权重，让模型针对不同Confidential节点自适应选择重要邻居。

为了判断改进来源，子项目4固定四组实验：

| 实验 | 架构 | loss排除 | 目的 |
|---|---|---|---|
| `baseline_gcn` | GCN | 否 | 复现子项目3当前有效基线 |
| `loss_mask_gcn` | GCN | 是 | 单独测试排除不可推断字段的收益 |
| `gat` | GAT | 否 | 单独测试注意力机制的收益 |
| `gat_loss_mask` | GAT | 是 | 测试两者组合是否提升中等R²字段 |

---

## 2026-05-21 - Modify by GPT5.5: 初始化子项目4与实验框架

### 修改前问题分析

子项目3的配置注释曾写到“排除不可推断字段”，但源码中没有实际的loss mask；训练仍然在12个Confidential字段上平均计算MSE。根据 `run_20260521_040031` 的结果估算，整体MSE为0.4817；如果仅统计排除 `net/gross_actual_interchange_mw` 后的10个目标，等价MSE约为0.3674，而两个不可推断目标本身平均MSE约为1.0529。这说明它们确实严重拖累训练口径。

同时，子项目3的 `compute_adjacency()` 只提供一个全局静态邻接矩阵，消息传递阶段无法学习每个目标节点自己的邻居偏好。这对 `congestion_price_da`、`congestion_price_rt`、`total_losses` 等中等R²字段可能是不够的。

### 修改思路

1. 复制 `DNN_Aggresvation3` 的源码骨架到 `DNN_Aggresvation4`，保留原始GCN基线。
2. 在训练模块增加 `loss_exclude_confidential` 配置项，训练和早停按未排除目标计算MSE，同时仍报告全目标MSE和被排除目标MSE。
3. 在模型模块新增 `architecture: gat` 分支，在固定边掩码内学习注意力权重，并保留相关性融合alpha作为注意力先验。
4. 让遮蔽敏感度使用同一目标损失口径，避免不可推断字段继续主导敏感度排序。
5. 增加四套实验配置和结果对比脚本。

### 修改内容

- 新建 `DNN_Aggresvation4`
- 修改 `src/model.py`
  - 支持 `architecture: gcn | gat`
  - GAT模式下每层学习 target/source attention logits
  - 使用相关性融合邻接作为attention prior
  - 保存最后一层平均注意力诊断
- 修改 `src/train.py`
  - 增加 `loss_exclude_confidential`
  - 记录 `final_test_mse_targets`、`final_test_mse_all`、`final_test_mse_excluded`
  - 输出每个Confidential字段的R²、MSE、是否参与loss
- 修改 `src/sensitivity.py`
  - 遮蔽敏感度支持只对训练目标集合计算MSE
- 修改 `scripts/run_pipeline.py`
  - 保存 `config_used.yaml`
  - 支持相关性张量缓存
  - 支持GAT注意力诊断导出
  - summary中写入实验名、架构、loss目标、逐目标MSE/R²
- 新增四套实验配置：
  - `configs/experiments/baseline_gcn.yaml`
  - `configs/experiments/loss_mask_gcn.yaml`
  - `configs/experiments/gat.yaml`
  - `configs/experiments/gat_loss_mask.yaml`
- 新增 `scripts/run_experiments.py` 和 `scripts/compare_runs.py`

### 测试结果

四组实验已完成，运行环境为 `conda env: Pytorch310_MacBookAir`，设备为CPU。相关性张量在第一组实验中重新计算，后续实验复用 `outputs/relationship_cache`。

运行目录：

| 实验 | 运行目录 |
|---|---|
| `baseline_gcn` | `outputs/run_20260521_051829_baseline_gcn` |
| `loss_mask_gcn` | `outputs/run_20260521_052210_loss_mask_gcn` |
| `gat` | `outputs/run_20260521_052604_gat` |
| `gat_loss_mask` | `outputs/run_20260521_053113_gat_loss_mask` |

核心指标：

| 实验 | 架构 | loss目标数 | 10个可推断目标MSE | 全12目标MSE | 被排除2目标MSE | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `baseline_gcn` | GCN | 12 | 0.367448 | 0.481692 | 1.052911 | 成功复现子项目3最佳基线 |
| `loss_mask_gcn` | GCN | 10 | 0.377320 | 0.574803 | 1.562220 | 只排除loss后并未改善10目标整体 |
| `gat` | GAT | 12 | 0.372783 | **0.472676** | 0.972139 | 全12目标MSE小幅优于baseline |
| `gat_loss_mask` | GAT | 10 | **0.362518** | 0.557638 | 1.533239 | 10目标口径最佳，但提升幅度较小 |

关键字段R²对比：

| 字段 | baseline_gcn | loss_mask_gcn | gat | gat_loss_mask | 观察 |
|---|---:|---:|---:|---:|---|
| `congestion_price_da` | 0.1990 | 0.1501 | 0.2037 | **0.2137** | GAT+mask小幅提升，但仍是最弱可推断字段 |
| `congestion_price_rt` | **0.4765** | 0.4479 | 0.4671 | 0.4630 | 未提升，baseline最好 |
| `total_losses` | 0.4482 | **0.4829** | 0.3922 | 0.4671 | loss-mask有效，GAT单独变差 |
| `marginal_loss_price_da` | 0.4198 | 0.4592 | 0.4851 | **0.5011** | GAT+mask显著提升 |
| `da_as_total_mw_thirty_minutes_reserve` | **0.3871** | 0.3175 | 0.2886 | 0.2920 | 新方案均变差 |
| `total_lmp_da` | 0.8843 | 0.8885 | 0.9200 | **0.9234** | GAT明显提升 |
| `total_gen` | 0.9154 | 0.9029 | 0.9027 | **0.9379** | GAT+mask提升 |
| `metered_load_mw` | 0.9567 | 0.9580 | 0.9522 | **0.9602** | 小幅提升 |
| `da_as_total_mw_synchronized_reserve` | 0.9082 | 0.9406 | **0.9462** | 0.9403 | GAT/loss-mask均改善 |
| `net_actual_interchange_mw` | -0.9610 | -2.0651 | **-0.8385** | -1.8833 | 不可推断性质未改变 |
| `gross_actual_interchange_mw` | 0.0049 | -0.2692 | **0.1185** | -0.4121 | 不可推断性质未改变 |

### 结果分析

1. **baseline复现成功**  
   `baseline_gcn` 的全目标MSE、逐字段R²、alpha权重和子项目3 `run_20260521_040031` 一致，说明子项目4的复刻没有破坏原始基线。

2. **只做loss-mask不是稳定收益**  
   `loss_mask_gcn` 的训练目标MSE为0.377320，看起来低于baseline全目标MSE 0.481692，但这是口径变化造成的。按同样10个可推断字段重算，baseline为0.367448，反而略好于 `loss_mask_gcn`。  
   这说明：单纯把 `net/gross_actual_interchange_mw` 从loss排除，并不会自动释放出更好的共享表示；甚至可能让模型丢掉一部分对整体图结构有帮助的误差信号。

3. **GAT有正向信号，但不是压倒性提升**  
   `gat` 的全目标MSE从0.481692降到0.472676，说明注意力机制确实有一点收益。它明显改善了 `marginal_loss_price_da`、`total_lmp_da`、`gross_actual_interchange_mw`，但同时拉低了 `total_losses` 和 `da_as_total_mw_thirty_minutes_reserve`。  
   因此，当前GAT实现不是“全面更好”，而是让模型重新分配容量，部分price类字段获益，部分reserve/loss字段受损。

4. **GAT + loss-mask 是当前四组中最适合继续推进的方向**  
   `gat_loss_mask` 在10个可推断目标上取得最低MSE 0.362518，较baseline同口径0.367448下降约1.34%。字段层面，它把：
   - `congestion_price_da` 从0.1990提升到0.2137
   - `marginal_loss_price_da` 从0.4198提升到0.5011
   - `total_lmp_da` 从0.8843提升到0.9234
   - `total_gen` 从0.9154提升到0.9379

   但它也让 `congestion_price_rt`、`total_losses`、`da_as_total_mw_thirty_minutes_reserve` 相比baseline下降，所以当前结论只能说“组合方向有价值”，不能说“已经全面解决中等R²字段问题”。

5. **注意力诊断支持GAT化思路，但注意力仍偏平**  
   在 `gat_loss_mask` 中：
   - `congestion_price_da` 的最高注意力邻居包括 `forecast_load_mw_latest_available`、`system_energy_price_da`、`marginal_loss_price_rt`、`total_lmp_da`、`marginal_loss_price_da`
   - `congestion_price_rt` 的最高注意力邻居包括 `total_lmp_rt`、`marginal_loss_price_rt`、`system_energy_price_da`
   - `total_lmp_da` 的最高注意力邻居包括 `system_energy_price_da`、`marginal_loss_price_rt`、`forecast_load`

   这些邻居在业务语义上合理，说明GAT确实在“看对方向”。但注意力熵整体偏高，例如 `congestion_price_da` 的归一化熵约0.99，意味着它只有5个邻居但权重仍接近平均分配；注意力还没有形成非常尖锐的字段选择。

6. **敏感度排序发生了合理变化**  
   `gat_loss_mask` 的Top敏感字段为：`da_as_nsr_mw_primary_reserve`、`gen_fuel_nuclear_mw`、`system_energy_price_da`、`total_lmp_rt`、`forecast_load_mw_latest_available`、`marginal_loss_price_rt`。  
   相比baseline，price相关字段在排序中更靠前，符合“推断价格类Confidential时更多关注price类General邻居”的目标。

### 阶段结论

子项目4验证了两个判断：

1. Claude提出的“GAT化”方向是有根据的，但当前实现带来的提升还偏温和，需要继续调结构和正则。
2. “排除不可推断字段”不能单独作为提升手段，它更像是和GAT搭配时的目标口径校正。单独使用loss-mask会让部分字段变好，但总体10目标MSE不如原baseline。

当前推荐保留 `gat_loss_mask` 作为下一轮实验起点，但不能替代 `baseline_gcn` 作为最终结论。下一步应围绕GAT本身继续优化，而不是继续堆loss-mask。

### 下一步建议

1. **降低注意力熵**：尝试 attention temperature、entropy penalty，或减少edge_mask邻居数（例如 `top_k=3`）让GAT更敢于选择关键邻居。
2. **目标分组训练/多头输出**：price类、load/gen类、reserve类的依赖结构不同，当前一个共享GNN可能存在任务冲突。
3. **只在Confidential目标聚合时使用GAT**：General节点之间仍用GCN，Confidential读取General时用attention，减少全图注意力带来的不稳定。
4. **单独保护退化字段**：`da_as_total_mw_thirty_minutes_reserve` 在所有新方案中下降，应检查其邻接边是否被GAT错误稀释。
5. **多seed复跑**：当前只跑seed=42，提升幅度只有1%左右，必须用多seed确认不是随机波动。

---

## 2026-05-21 - Modify by GPT5.5: Route A 单目标MLP上限诊断

### 修改前问题分析

四组GNN/GAT实验后仍有一个关键疑问没有回答：

> 低R²到底是因为字段本身不可从General推断，还是因为“一张共享图同时推断多个Confidential”造成多任务冲突？

当前GNN/GAT模型存在共享结构假设：所有Confidential目标共用同一套节点表示、同一张edge mask、同一组alpha相关性融合权重，只是在GAT中允许不同节点有局部attention。真实业务关系很可能不是这样：

- `total_gen` 更依赖fuel/load类字段
- `congestion_price_*` 更依赖price/LMP/marginal loss类字段
- `reserve` 字段更依赖ancillary service字段
- `interchange` 字段可能依赖外部交换/调度信息

如果每个目标单独建模能显著超过共享GNN，就说明共享图/多任务结构确实在拖累；如果单目标也低，则说明该字段在当前General集合下确实难推断。

### 修改思路

新增一个诊断实验，不改变GNN主线：

1. 对每个Confidential字段单独训练一个MLP。
2. 输入只使用44个General字段，不使用Confidential真实值。
3. 每个目标独立早停、独立保存训练曲线和模型。
4. 输出单目标probe R²，并与 `baseline_gcn`、`gat_loss_mask` 对比。

注意：这个MLP probe不是理论上限，只是一个“非图、非共享、多目标冲突较少”的诊断参照。如果probe高于GNN，说明目标有直接General可推断信号；如果probe低于GNN，说明图结构/间接关系对该目标有帮助。

### 修改内容

- 新增 `scripts/run_target_probe.py`
  - `TargetMLP`: `[128, 64]` 两层MLP + BatchNorm + ReLU + Dropout
  - 每个Confidential字段单独训练
  - 保存：
    - `target_probe_results.csv`
    - `summary.json`
    - 每个target的训练曲线 `histories/*.csv`
    - 每个target的模型权重 `models/*.pt`

### 测试结果

运行命令：

```bash
conda run -n Pytorch310_MacBookAir python DNN_Aggresvation4/scripts/run_target_probe.py
```

运行目录：

`outputs/target_probe_20260521_113054`

结果对比：

| Confidential字段 | 单目标MLP R² | baseline_gcn R² | gat_loss_mask R² | 诊断结论 |
|---|---:|---:|---:|---|
| `total_gen` | **0.9906** | 0.9154 | 0.9379 | 单目标显著更高，共享图有拖累 |
| `metered_load_mw` | **0.9762** | 0.9567 | 0.9602 | 单目标略高 |
| `da_as_total_mw_primary_reserve` | **0.9752** | 0.9751 | 0.9735 | 三者几乎一致，已接近上限 |
| `da_as_total_mw_synchronized_reserve` | **0.9418** | 0.9082 | 0.9403 | GAT+mask接近单目标 |
| `total_lmp_da` | **0.9299** | 0.8843 | 0.9234 | GAT+mask接近单目标 |
| `total_losses` | **0.6541** | 0.4482 | 0.4671 | 单目标大幅更高，共享图明显拖累 |
| `da_as_total_mw_thirty_minutes_reserve` | **0.4569** | 0.3871 | 0.2920 | 单目标更高，GAT当前伤害该字段 |
| `congestion_price_rt` | 0.4562 | **0.4765** | 0.4630 | 单目标不占优，图结构有帮助 |
| `marginal_loss_price_da` | 0.4407 | 0.4198 | **0.5011** | GAT+mask最好，图注意力有帮助 |
| `net_actual_interchange_mw` | **0.2482** | -0.9610 | -1.8833 | 不是完全不可推断，GNN结构严重不适配 |
| `gross_actual_interchange_mw` | **0.2461** | 0.0049 | -0.4121 | 单目标有弱信号，共享图/多任务不适配 |
| `congestion_price_da` | 0.0476 | 0.1990 | **0.2137** | 单目标MLP很弱，图结构/间接关系必要 |

整体：

- 12目标平均单目标MLP R²：0.6136
- 10个非interchange目标平均单目标MLP R²：0.6869
- `baseline_gcn` 的12目标平均R²：0.4679
- `gat_loss_mask` 的12目标平均R²：0.3647（因为被排除字段预测更差）

### 结果分析

1. **“一张共享图推断所有节点”确实会拖累部分字段**  
   `total_losses` 是最强证据：单目标MLP达到0.6541，而baseline GCN只有0.4482，`gat_loss_mask`也只有0.4671。这说明该字段不是数据不可推断，而是共享GNN/GAT结构没有学好它的专属关系。

   `total_gen`、`total_lmp_da`、`da_as_total_mw_thirty_minutes_reserve` 也有类似现象，只是程度不同。

2. **某些字段确实需要图结构/间接关系**  
   `congestion_price_da` 单目标MLP只有0.0476，远低于GNN/GAT的0.1990/0.2137，而且best_epoch=3，说明直接从44个General字段做黑箱MLP很快过拟合或找不到稳定信号。  
   这类字段可能不是“单目标DNN即可解决”，而是需要图结构、相关性先验、或更明确的price类子图。

   `marginal_loss_price_da` 也是GAT+mask最好，说明注意力机制确实对price/loss类目标有价值。

3. **interchange字段不是绝对不可推断，但不适合当前共享图**  
   之前 `net_actual_interchange_mw` 在GNN里是负R²，很容易被解释为“完全不可推断”。但单目标MLP达到0.2482，`gross_actual_interchange_mw` 达到0.2461，说明General字段里存在弱到中等的直接信号。  
   问题更可能是：interchange字段和其他目标的机制差异太大，在共享图/共享loss里被其他任务扰乱；把它们简单作为“安全字段”可以作为当前GNN结论，但从数据角度不能说完全没有泄露信号。

4. **Route A支持“target-specific”路线**  
   单目标MLP明显高于共享GNN的字段包括：
   - `total_losses`: +0.2059
   - `net_actual_interchange_mw`: +1.2092
   - `gross_actual_interchange_mw`: +0.2412
   - `total_gen`: +0.0752
   - `da_as_total_mw_thirty_minutes_reserve`: +0.0698
   - `total_lmp_da`: +0.0456

   这说明后续模型不应继续坚持“一张图一套参数覆盖所有目标”。更合理的是目标条件化或分组模型。

### 阶段结论

Route A回答了前面的问题：图结构本身不是无效，但“共享全局图 + 多目标同时还原”确实不适合所有Confidential目标。不同目标需要不同推断路径：

- **直接General信号强**：`total_gen`、`metered_load_mw`、`total_losses`
- **图/price子图有帮助**：`congestion_price_da`、`marginal_loss_price_da`、`total_lmp_da`
- **当前GAT伤害，需要单独保护**：`da_as_total_mw_thirty_minutes_reserve`
- **机制特殊，需独立建模**：`net/gross_actual_interchange_mw`

下一步不建议继续只调全局GAT。更优先的方向是：

1. **Route C：按业务组分模型**
   - price/loss组：`congestion_price_da/rt`、`marginal_loss_price_da`、`total_lmp_da`、`total_losses`
   - load/gen组：`total_gen`、`metered_load_mw`
   - reserve组：`da_as_total_mw_*`
   - interchange组：`net/gross_actual_interchange_mw`

2. **Route B：target-conditioned GAT**
   - 让attention显式依赖目标字段embedding
   - 学习 `edge_weight(source -> target | target_confidential)`，而不是共享一套边含义

3. **保留单目标MLP作为每个字段的参照上限**
   - 后续任何GNN/GAT改动，都应该和 `target_probe_20260521_113054` 的单目标结果对比，判断是否真的利用了图结构，而不是被多任务共享拖累。
