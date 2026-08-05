# DNN_Aggresvation85 日志

## 2026-07-24：定位

84 号由 Claude 使用，已完成“单调约束 + S1–S2 分歧级联”；本号使用下一个空闲编号 85，
不修改 84。

D83 已证明逐集合 `poly2/arithmetic ridge` 能显著恢复强协同，但仍有两个问题：

1. 当前实现对每个集合重复扫描全部 4,589 行训练数据；
2. standalone ridge 对强算术协同很好，但整体排序不如 K25/K50。

本号测试两个互补改进：

### A. 缓存充分统计量

固定 poly2 字典后预计算 fit/validation/train/test 的：

`G=ΦᵀΦ, H=ΦᵀY, YᵀY`

查询集合时只抽取与该集合有关的 raw/square/product 列，解最高 9 维的小矩阵，并直接通过
二次型计算 test SSE。扫描阶段不再触碰原始样本行。

验收：

- 与 D83 非缓存 poly2 的每个 R² 最大差；
- 冷启动 cache 构建时间；
- 热缓存全 946 个 pair + 13,244 个 triple 的时间与 query/s。

### B. K0 + 结构化残差 ridge

对每个集合先得到 uniform oracle K0 的逐样本预测 `p0`，再闭式拟合：

`Y - p0 = ridge(可见字段的 linear/poly2/arithmetic 字典)`

最终预测为 `p0 + residual`。这相当于只让小模型补 K0 缺失的关系，不做任何神经网络反向
传播。先在 2,197 个真值三元组及其全部 pair 上检验；若成功再扩全空间。

## 结果

### 1. 缓存充分统计量：成功，而且是精确加速

预先构造全局 `raw + square + pair-product` 字典，并缓存训练内
fit/validation、全训练和 test 的 Gram/cross 矩阵。查询集合时只抽取该集合对应的小矩阵。

| 项目 | 结果 |
|---|---:|
| 冷启动 cache 构建 | 0.605 s |
| 热 cache 读取 | 0.035 s |
| 全 946 pair + 13,244 triple 扫描 | **4.96 s（CPU）** |
| 查询吞吐 | 2,862 sets/s |
| 与 D83 原逐集合 poly2 最大 R² 差 | **1.16e-10** |

因此这不是近似代码替换，而是把同一个岭回归的重复样本扫描改写成充分统计量查询。
D83 的两个结构化模型扫描约 131.5 s；本号单个 poly2 热扫描约 5 s。

### 2. K0 + 查询集合残差修正：绝对值更准，但不是最优主筛

对每个集合先做通用 MLP oracle 的 K0 逐样本预测 `p0`，再闭式拟合
`Y-p0`。全空间结果：

| 方法 | parent MAE | syn3 MAE | S1 Spearman | 全空间 Top-30% recall | 极强 S1 均值 | 极强负值 | 实测时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| universal K0 | 0.0718 | 0.0196 | 0.6824 | 0.8424 | 0.0045 | 11/15 | 3.73 s |
| universal K25 | 0.0414 | 0.0121 | 0.8401 | 0.8989 | 0.0611 | 1/15 | — |
| universal K50 | **0.0327** | **0.0101** | **0.8727** | 0.9110 | 0.1304 | 0/15 | — |
| cached standalone poly2 | 0.0613 | 0.0154 | 0.7845 | **0.9630** | 0.2773 | 0/15 | **4.99 s** |
| K0 + poly2 residual | 0.0451 | 0.0128 | 0.8302 | 0.9581 | 0.2675 | 0/15 | 60.85 s |
| K0 + arithmetic residual | 0.0377 | 0.0113 | 0.8310 | 0.9605 | **0.2920** | 0/15 | 94.16 s |

解释：

- 若目标是**绝对敏感度尽量贴近 DNN 重训练**，arithmetic residual 很有价值；
- 若目标是**从全空间找三阶协同**，约 5 s 的 cached poly2 已比 K50 的召回高；
- cached poly2 的 parent MAE 并非最好，却能把强协同差值恢复出来。原因是 parent 和直接
  pair 的共同低估会在作差时抵消，而显式乘积列又专门补了当前强协同所需的关系；
- residual arithmetic 的绝对值最好，但多出的 89 s 只换来相近的 Top-30% 召回，
  所以不应默认给所有候选运行。

D79 原 K 网格从 K0 同轨迹跑到 K50，全 4 GPU 墙钟约 510 s（含各中间 K）。
新的 cached poly2 是数量级更便宜的低阶主扫描通道。

### 3. 不是被挑选样本造成的假提升

2197 个真值三元组分成两部分：1800 个无偏随机三元组与 397 个专门认证三元组；再按
D79 固定 tune/test 划分。全空间先排序、只保留 Top-30%，最后在 test 真值点打分：

| 方法 | 加权总体 recall | 随机 test（46 个强） | 定向 test（84 个强） |
|---|---:|---:|---:|
| K0 | 0.8424 | 0.8478 | 0.8214 |
| K25 | 0.8989 | 0.8913 | 0.9286 |
| K50 | 0.9110 | 0.8913 | 0.9881 |
| cached poly2 | **0.9630** | **0.9565** | 0.9881 |
| residual poly2 | 0.9581 | 0.9565 | 0.9643 |
| residual arithmetic | 0.9605 | 0.9565 | 0.9762 |
| residual/poly2 无真值 rank 融合 | **0.9654** | **0.9565** | **1.0000** |

这说明提升同时存在于无偏随机池和定向难例，不只是把原先挑出的候选背下来。
但最强 `syn3>0.20` 的独立 test 只有 5 个，且都来自定向池，所以“极强召回=1”只能作
正面信号，不能当精确总体比例。

### 4. S2 救援的角色发生变化

用 tune 集在每个候选预算选择 S2 保护通道占比：

- 原 K0 主筛在 Top-30% 仍会选择 10% 的 S2 lane；
- residual arithmetic 主筛在 5/10/20/30% 全部选择 **0% S2 lane**；
- residual + poly2 除 Top-10% 外也都选择 0%。

所以 D81/D84 的 S2 不是错误，而是 **K0-S1 失明时的补救信号**。一旦换成能恢复乘法/除法
关系的 S1，绝对泄露 S2 不再提供稳定增量。当前更干净的协议是直接把结构化 S1 当低阶
主筛，而不是永久固定“S1+S2”。

不同保留预算下没有一个估计器处处最优：

- Top-5%：residual poly2 recall 0.5655；
- Top-10%：residual arithmetic 0.7777；
- Top-20%：residual/poly2 rank-max 0.9259；
- Top-30%：rank 融合 0.9654，cached poly2 单独已 0.9630。

因此默认推荐以 **cached poly2（5 s）**做宽筛；只有候选池必须压到 5–10% 时，再运行
residual 通道并融合。

### 5. 扩到任意大小集合：稀疏二阶字典可用，但枚举仍爆炸

先用不接触真值的二阶估计分数选出 Top-50/100/200 pair edges。对更大集合只保留：

`可见字段 raw + square + 该集合内部命中的已选 pair-product`

在 D63 的 105 个任意大小重训练真值点上：

| 方法 | overall MAE | bias | Spearman | 50 个随机点 MAE / rho |
|---|---:|---:|---:|---:|
| D63 universal MLP K0 | 0.0796 | -0.0796 | 0.9763 | **0.0870 / 0.9846** |
| sparse poly2 Top-200 | **0.0779** | -0.0779 | **0.9785** | 0.0911 / **0.9882** |

按 size 的 MAE：

| size | MLP K0 | sparse Top-200 |
|---|---:|---:|
| 1–4 | 0.0564 | **0.0544** |
| 5–8 | 0.1215 | **0.1157** |
| 9–16 | 0.1156 | **0.1062** |
| 17–32 | **0.0701** | 0.0704 |
| 33–44 | **0.0348** | 0.0469 |

结论应写成“低成本第二攻击通道与 MLP K0 整体相当、排序略强”，不能写成全面替代：
在真正随机 50 点的绝对 MAE 略差，而且 33–44 大集合仍由 MLP 更准。

Top-200 稀疏查询从 size=4 的约 0.35 ms 增长到 size=44 的约 14.4 ms。但全枚举外推：

| 阶数 | 组合数 | 单机全枚举约耗时 |
|---:|---:|---:|
| 4 | 135,751 | 约 47 s |
| 5 | 1,086,008 | 约 6.8 min |
| 6 | 7,059,052 | 约 45 min |
| 8 | 177,232,627 | 约 23 h |
| 10 | 2,481,256,778 | 约 357 h |

这钉死了两个层次：

1. 缓存/稀疏字典解决“**一次查询太贵**”；
2. 四阶以上仍必须用 beam、剥离/极小充分集或分支定界解决“**候选数量太多**”。

不能因为一次查询达到毫秒级，就声称高阶组合爆炸已经解决。

## 当前推荐协议

1. universal MLP K0：保留为覆盖任意 mask 的免费基线；
2. 二/三阶：cached poly2 全扫作为默认主筛；
3. 需要更小候选池或更准绝对值时：只对候选增加 residual poly2/arithmetic；
4. 最终候选：K25/K50 或完整重训练认证；
5. 四阶以上：sparse Top-200 作为快速 `v(S)` 查询器，但另配非枚举搜索，不直接全扫。

这比原来的“全部 K=25 微调”多了一条真正廉价且能看见强协同的通道，也解释了为什么
原 K0-S1 失明、S2 能救，而结构化 S1 本身又不再依赖 S2。

## 验证与边界

`outputs/validation.json` 全部通过：

- cached poly2 与 D83 原实现最大差 `1.16e-10`；
- 新 K0 与 D79 的 170,280 个 R² 最大差 `2.80e-6`；
- truth/full 两次 residual 运行最大差 `<2.5e-8`；
- 全 13,244 三元组无缺失；
- alpha 只用训练内部验证，保护通道只用 tune 选，test 只作最终报告；
- 高阶 pair edge 选择只用估计器自身分数，不用重训练真值。

必须保留四条边界：

1. 随机 test 中 `syn3>0.10` 只有 46 个，`>0.20` 为 0；
2. `>0.20` 的 test 召回只由 5 个定向样本支撑；
3. D63 任意大小基准只有 105 点，其中 50 随机、55 为 top-k；
4. poly2/arithmetic 是合法攻击者下界，不是数学上的无条件 `sup_f`。

## 主要产出

```text
scripts/cached_poly2.py              缓存充分统计量的全空间低阶扫描
scripts/residual_ridge.py            K0 + 查询集合闭式残差修正
scripts/analyze_search.py            随机/定向分层、全空间 recall 与无真值融合
scripts/eval_high_order.py           任意大小真值、稀疏边和枚举成本
scripts/make_summary.py              统一表格与图
scripts/validate_outputs.py          数值链与隔离验证
outputs/low_order_method_comparison.csv
outputs/global_recall_curve.csv
outputs/source_stratified_metrics.csv
outputs/high_order_fidelity_summary.csv
outputs/high_order_enumeration_benchmark.csv
outputs/validation.json
figures/d85_search_and_fidelity.png
```
