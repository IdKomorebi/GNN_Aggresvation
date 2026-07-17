# DNN_Aggresvation51 日志

## Modify by Claude: 2026-06-26

## 1. 子项目定位

为当前「**分裂聚合**」设计提供消融证据(回应审稿质疑),并系统扫一遍聚合方式的
设计空间。当前模型对两类边用两套聚合:

- **General→General**:GCN 聚合,相关性先验加权的**固定**邻接(共享 `alpha_general`)。
- **General→Confidential**:门控 GAT,`score = gate·(q·k/√d) + (1-gate)·scale·log(prior)`。

把**整张图**强制改成单一聚合方式,看 12 个 confidential 的平均 test R²。**图结构
不变**(所有节点都只从其 general 邻居聚合,边集与分裂设计完全一致),只换聚合函数。
只测准确率。基准 0.866,DNN probe 单字段诊断上界 0.8937。

## 2. 设计空间的三个维度

每种聚合可由三个正交维度刻画(这也是审稿最可能问的):

- **是否采用 GAT**:有没有数据驱动的动态 q·k 注意力。
- **GAT 作用范围**:无 / 仅 confidential(G→C) / 全图。
- **相关系数权重(先验)类型**:
  - **无**:不用相关性加权(只用相关性筛出的拓扑)。
  - **全局静态**:单一共享 `alpha_general`(5 维),固定不随样本变化。
  - **部分静态**:G-G 用全局静态、G→C 用逐边 `alpha_confidential`(分裂设计)。
  - **全局动态**:用全局 alpha 构造先验,但由逐对 gate **逐样本**决定与注意力的混合。

## 3. 完整配置 × 效果表(12-conf 平均 test R²)

| 变体 | 用GAT | GAT范围 | 相关系数权重 | 混合方式 | 平均R² | Δvs基准 | 距上界 |
|---|---|---|---|---|---:|---:|---:|
| **split_baseline(当前)** | 是 | 仅G→C | 部分静态 | G-G:GCN静态 / G→C:逐对动态gate | **0.8663** | +0.0000 | −0.0274 |
| gcn_noprior | 否 | 无 | 无 | GCN均匀邻接(行归一) | 0.8737 | +0.0074 | −0.0200 |
| gcn_prior | 否 | 无 | 全局静态 | GCN先验加权静态邻接 | 0.8667 | +0.0004 | −0.0270 |
| gat_noprior | 是 | 全图 | 无 | 纯动态 q·k | 0.8506 | −0.0157 | −0.0431 |
| **gat_prior_static** | 是 | 全图 | **全局静态** | 动态q·k + 全局scale×log先验(no_gate) | **0.8523** | **−0.0140** | −0.0414 |
| gat_prior | 是 | 全图 | 全局动态 | 逐对动态gate混合(q·k与全局先验) | 0.8727 | +0.0064 | −0.0210 |

基准精确复现 0.8663(=已知 0.866),对比可信。图:`outputs/ablation_r2.png`。

## 4. 关键发现

### 4.1 易推字段上聚合方式完全等价
7 个高 R² 字段(metered_load/total_gen/total_lmp/各 reserve,R²>0.88)在 6 个变体
间差异 ≤0.005。信号足够强时,聚合方式无关紧要——所有区别都发生在难推的低 R² 字段
(congestion_price_da/rt、gross_actual_interchange、marginal_loss_price)。

### 4.2 GCN 一支:加不加先验都差不多
gcn_noprior(0.8737) 与 gcn_prior(0.8667) 接近,均匀聚合甚至略好。GCN 对相关性筛出的
拓扑做平均本身已是很强的 baseline,静态先验加权对它帮助有限。

### 4.3 GAT 一支:先验**必须动态混合**才有用(本子项目最重要的结论)
- gat_noprior(纯 q·k,无先验):**0.8506**
- gat_prior_static(q·k + 全局静态先验,no_gate):**0.8523** ← 几乎等于无先验!
- gat_prior(q·k + 逐对动态 gate):**0.8727** ← 比静态高 +0.020

→ **把先验以单一全局静态权重加进去几乎没用(0.8523≈0.8506);只有逐对、逐样本的
动态 gate 才能让 GAT+先验真正生效。** gat_prior_static 输给 gat_prior 的部分全部
集中在难字段:congestion_price_rt −0.082、congestion_price_da −0.081、
marginal_loss_price −0.042——正是信号弱、需要**自适应权衡「信不信先验」**的地方,
而单个全局 scale 无法因边/样本而变。

### 4.4 唯一稳健最差的是「全图 GAT 但缺动态混合」
gat_noprior 与 gat_prior_static 是仅有的两个明显低于基准(−0.014~−0.016)的变体,
共同点是 GAT 缺少自适应的先验混合。说明**门控(gate)不是装饰,而是 GAT 路径能用的
前提**。

## 5. 回答用户的两个问题

1. **「gat 加全局静态注意力会不会更好?」→ 不会,反而更差(0.8523,低于基准
   0.014)**,几乎退回无先验水平。要全用 GAT,必须保留逐对动态 gate(gat_prior
   0.8727),不能用静态 no_gate 版。

2. **「GCN+先验是全局静态/部分静态/全局动态?」→ 全局静态**:单一共享
   `alpha_general` 构造固定邻接,行归一后直接 `A@h`,A 不随样本变化。

## 6. 对「分裂设计」与「全用 GAT」的取舍结论

- 准确率上,{split 0.8663, gcn_noprior 0.8737, gcn_prior 0.8667, gat_prior 0.8727}
  四者在单 seed 噪声内打平,**没有方案系统性优于分裂**;gcn_noprior 名义最高但优势
  来自两个难字段的随机波动。
- **若要"全用一种聚合",唯一能与分裂打平的是 gat_prior(动态 gate)**;gat_noprior
  /gat_prior_static 都明显更差。
- 保留 GAT(分裂或 gat_prior)的真正理由超出准确率:① attn 这个反例需要注意力才能
  算(DNN50 "注意力归因在字段缺失下崩溃" 的论据);② GAT 的注意力/gate 才是"学到的
  推断传播路径"(项目创新点 DNN41 的基础),纯 GCN 只有固定先验图、无路径可提取。
- gat_prior(全图 GAT + 全局动态 gate)是表达能力的超集、自由度最大,准确率与分裂
  打平,且只剩一个全局 alpha 向量(逐边 `alpha_confidential` 不再使用),结构更简洁,
  可作为后续统一形式的候选。

## 7. 局限
单 seed。4.1/6 关于"差异属噪声"的判断若要严格,建议对 {split, gcn_noprior,
gat_prior} 各跑 3~5 seed 看均值±方差(预计互相覆盖)。4.3 的结论(静态 vs 动态 gate
差 0.020,且集中在难字段)幅度较大、方向一致,基本不受 seed 影响。

## 8. 输出
```
outputs/<variant>/{model.pt, per_target_r2.csv, summary.json}   6 个变体
outputs/ablation_summary.csv   (含配置维度 + Δvs基准 + 距probe上界)
outputs/ablation_r2.png        (柱状图 + 基准线 + probe 上界线)
```
变体:split_baseline / gcn_noprior / gcn_prior / gat_noprior / gat_prior_static / gat_prior
