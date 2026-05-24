# DNN_Aggresvation15 日志

## Modify by GPT5.5: 2026-05-24

### 1. 子项目15要解决的问题

DNN14 中固定 `top_k=8` 的 sharp attention maskloss 是当前整体最优配置，但固定邻居数仍有一个问题：

```text
如果一个节点客观上和很多节点都有较高相关性，固定top_k可能漏掉有效source；
如果一个节点客观上只有少量高相关邻居，固定top_k又会强行引入噪声source。
```

因此 DNN15 测试自适应邻居选择：

```text
最低保留 4 个邻居；
最高保留 12 个邻居；
中间由 threshold=0.25 决定。
```

即：超过阈值的邻居有多少就保留多少，但不低于 4、不高于 12。

### 2. 修改内容

在 `src/model.py` 的 `build_edge_mask()` 中新增：

```text
selection: adaptive_threshold_topk
min_k: 4
max_k: 12
```

自适应规则为：

```text
above = number of neighbors whose mean correlation >= threshold
keep_count = min(max(above, min_k), max_k)
```

然后按平均相关性排序保留前 `keep_count` 个邻居。

如果开启 `symmetrize=true`，对称化后可能让某些节点超过 `max_k`，因此 DNN15 会再按行剪枝一次，保证每个 target 的入邻居数量仍然保持在 `[min_k, max_k]` 附近，并避免固定 top-k 的刚性约束。

### 3. 实验设置

DNN15 只运行两组实验，都采用 DNN14 的最优 attention 温度：

```text
attention_temperature = 1.0
```

两组配置为：

```text
adaptive_sharp_maskloss
adaptive_sharp_allloss
```

其中：

```text
adaptive_sharp_maskloss:
  排除 net_actual_interchange_mw 和 gross_actual_interchange_mw 的 loss

adaptive_sharp_allloss:
  12 个 confidential target 全部进入 loss
```

输出目录为：

```text
DNN_Aggresvation15/outputs_adaptive/
```

### 4. 待观察

重点观察：

```text
1. adaptive maskloss 是否能超过 DNN14 sharp_attention_maskloss 的 target_MSE=0.331766；
2. thirty_minutes_reserve 是否能接近 single-target probe R2=0.4569；
3. congestion_price_da 是否因更多/更少邻居而改善；
4. allloss 是否仍然被两个不可推断 interchange target 拖累；
5. 实际边数和节点邻居数是否符合 min=4, max=12 的设定。
```

### 5. 测试结果

两组实验已完成。汇总文件：

```text
DNN_Aggresvation15/outputs_adaptive/comparison_summary.csv
```

实际构图结果为：

```text
边数: 523
节点入邻居数: min=4, median=12.0, mean=9.34, max=12
```

这说明自适应规则生效了，但相比 DNN14 sharp attention 的固定构图：

```text
DNN14 sharp_attention_maskloss: 908 条边
DNN15 adaptive_sharp_maskloss:  523 条边
```

DNN15 的图明显更稀疏。

核心结果如下：

```text
experiment                target_MSE  all_MSE   cp_da   cp_rt   losses  mlp_da  30min_reserve
adaptive_sharp_maskloss   0.366659    0.570855  0.1680  0.4432  0.5330  0.4613  0.4142
adaptive_sharp_allloss    0.480862    0.480862  0.0690  0.4535  0.5279  0.4164  0.3762
```

对比 DNN14 最优的 `sharp_attention_maskloss`：

```text
DNN14 sharp_attention_maskloss:
  target_MSE = 0.331766
  all_MSE    = 0.541777
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926
```

### 6. 结果分析

#### 6.1 自适应构图没有超过 DNN14 整体最优

DNN15 maskloss 的整体误差：

```text
0.366659
```

明显高于 DNN14 sharp attention：

```text
0.331766
```

主要原因不是自适应思想一定错误，而是本轮 `threshold=0.25 + max_k=12 + 对称后再剪枝` 让图从 908 条边降到 523 条边，信息通路被砍得比较多。

这对依赖较多 source 的 target 伤害明显：

```text
total_losses:          0.6768 -> 0.5330
congestion_price_da:   0.2678 -> 0.1680
congestion_price_rt:   0.4796 -> 0.4432
```

其中 `congestion_price_da` 在 DNN15 中只有 6 个邻居，注意力 entropy normalized 仍高达：

```text
0.9897
```

也就是说，它既少了邻居，又没有形成强选择注意力，因此效果下降比较合理。

#### 6.2 thirty_minutes_reserve 反而提升

DNN15 maskloss 对 `da_as_total_mw_thirty_minutes_reserve` 有提升：

```text
DNN14 sharp: 0.3926
DNN15 mask:  0.4142
```

这支持一个判断：该字段不一定需要更多边，反而可能受噪声邻居影响。自适应剪枝后，它保留 12 个邻居，效果更好。

但它仍没有达到 DNN4 single-target probe：

```text
probe R2 = 0.4569
```

说明还有一部分 target-specific 信息没有被多目标 GNN 充分利用。

#### 6.3 allloss 仍不建议作为主线

allloss 让两个 interchange target 进入训练后，整体 all_MSE 数值看起来变低：

```text
adaptive_sharp_allloss all_MSE = 0.480862
```

但这是评价口径改变后的结果，不代表主线更好。它明显损害了关键弱 target：

```text
congestion_price_da:           0.1680 -> 0.0690
marginal_loss_price_da:        0.4613 -> 0.4164
thirty_minutes_reserve:        0.4142 -> 0.3762
```

并且 `net_actual_interchange_mw` 仍然是负 R2：

```text
net_actual_interchange_mw: -0.8672
```

因此 allloss 仍然不适合作为主方案。

### 7. 当前结论

DNN15 证明了：

```text
1. 固定top_k确实不是唯一合理选择；
2. 自适应剪枝能改善 thirty_minutes_reserve；
3. 但本轮自适应图过稀疏，整体不如 DNN14 sharp attention；
4. congestion_price_da 对邻居数量和边筛选非常敏感；
5. allloss 仍然会被不可推断字段干扰。
```

如果继续改，建议不要完全回到固定 top_k，也不要继续使用当前 523 条边的稀疏图。更合理的下一步是：

```text
保留自适应思想，但放宽上限或降低阈值，例如：
  min_k=6, max_k=16, threshold=0.20
或：
  min_k=8, max_k=16, threshold=0.25
```

这样既允许低相关节点少连边，也避免把 `total_losses`、`marginal_loss_price_da` 这类需要较多 source 的 target 截得太狠。

## Modify by GPT5.5: 2026-05-24 追加回调实验

### 8. 回调内容

根据上一轮结论，在同一个子项目15中将自适应构图从：

```text
threshold = 0.25
min_k = 4
max_k = 12
```

放宽为：

```text
threshold = 0.20
min_k = 6
max_k = 16
```

并重新运行两组：

```text
adaptive_relaxed_maskloss
adaptive_relaxed_allloss
```

为了不覆盖上一轮结果，本次实验名从 `adaptive_sharp_*` 改为 `adaptive_relaxed_*`。

### 9. 放宽版构图结果

实际构图结果为：

```text
边数: 753
节点入邻居数: min=6, median=16.0, mean=13.45, max=16
```

对比：

```text
DNN15 第一轮 adaptive_sharp: 523 条边
DNN15 回调 adaptive_relaxed: 753 条边
DNN14 sharp_attention:        908 条边
```

因此本轮确实处在“比第一轮自适应更密，但仍比 DNN14 稍稀疏”的中间位置。

### 10. 回调测试结果

```text
experiment                  target_MSE  all_MSE   cp_da   cp_rt   losses  mlp_da  30min_reserve
adaptive_sharp_maskloss     0.366659    0.570855  0.1680  0.4432  0.5330  0.4613  0.4142
adaptive_relaxed_maskloss   0.378450    0.580680  0.0798  0.4687  0.5061  0.4757  0.4008
adaptive_sharp_allloss      0.480862    0.480862  0.0690  0.4535  0.5279  0.4164  0.3762
adaptive_relaxed_allloss    0.452866    0.452866  0.1086  0.4829  0.5588  0.4726  0.3797
```

对比 DNN14 最优：

```text
DNN14 sharp_attention_maskloss:
  target_MSE = 0.331766
  all_MSE    = 0.541777
  cp_da      = 0.2678
  cp_rt      = 0.4796
  losses     = 0.6768
  mlp_da     = 0.4832
  30min      = 0.3926
```

### 11. 回调结果分析

#### 11.1 放宽版 maskloss 没有改善主线

放宽版 maskloss 的整体结果比第一轮自适应更差：

```text
adaptive_sharp_maskloss:
  target_MSE = 0.366659

adaptive_relaxed_maskloss:
  target_MSE = 0.378450
```

虽然边数从 523 增加到 753，但整体没有接近 DNN14 sharp 的：

```text
0.331766
```

这说明问题不是简单的“上一轮边太少”。自适应边选择改变了图结构分布后，模型可能更难稳定利用 price 和 loss 类目标所需的 source。

具体看：

```text
congestion_price_da:    0.1680 -> 0.0798
total_losses:           0.5330 -> 0.5061
30min_reserve:          0.4142 -> 0.4008
```

这些都下降。只有：

```text
congestion_price_rt:    0.4432 -> 0.4687
marginal_loss_price_da: 0.4613 -> 0.4757
```

略有提升。

因此对 maskloss 主线来说，自适应构图目前不如 DNN14 固定 top_k + threshold 的组合。

#### 11.2 放宽版 allloss 有明显改善，但仍不适合作为主线

allloss 在放宽后有明显改善：

```text
adaptive_sharp_allloss:
  all_MSE = 0.480862

adaptive_relaxed_allloss:
  all_MSE = 0.452866
```

多个字段也提升：

```text
congestion_price_rt:    0.4535 -> 0.4829
total_losses:           0.5279 -> 0.5588
marginal_loss_price_da: 0.4164 -> 0.4726
congestion_price_da:    0.0690 -> 0.1086
```

并且两个 interchange target 相比之前 allloss 有所改善：

```text
gross_actual_interchange_mw: 0.0117
net_actual_interchange_mw:  -0.5379
```

但 `net_actual_interchange_mw` 仍然是负 R2，且关键字段 `congestion_price_da` 仍明显低于 maskloss/DNN14。因此 allloss 可以作为诊断对照，但仍不建议作为主线。

#### 11.3 注意力仍然偏平均

放宽版 maskloss 的注意力 entropy normalized：

```text
congestion_price_da:    0.9569
congestion_price_rt:    0.9302
total_losses:           0.9532
marginal_loss_price_da: 0.9471
30min_reserve:          0.9643
```

放宽版 allloss：

```text
congestion_price_da:    0.9670
congestion_price_rt:    0.9382
total_losses:           0.9437
marginal_loss_price_da: 0.9533
30min_reserve:          0.9667
```

邻居数增加到 16 后，attention 仍然比较平均，尤其是 `congestion_price_da` 和 `30min_reserve`。这解释了为什么增加边数并没有带来稳定收益：模型看到了更多 source，但没有足够强地选择其中真正有用的 source。

### 12. 子项目15最终结论

DNN15 的两个自适应构图版本都没有超过 DNN14 sharp attention maskloss。

当前最可靠主线仍然是：

```text
DNN14 sharp_attention_maskloss
```

也就是：

```text
fixed top_k=8 + threshold=0.25
attention_temperature=1.0
maskloss
```

DNN15 的价值主要是证明：

```text
1. 自适应边数可以改变个别 target 的表现；
2. thirty_minutes_reserve 对剪枝较敏感，稀疏图反而有一定帮助；
3. congestion_price_da 对图结构非常敏感，但自适应边数没有解决它；
4. 简单增加 max_k 或降低 threshold 并不能稳定提升；
5. 后续重点不应继续调边数，而应考虑 target-specific head、target-specific loss 或更强的 source selection 机制。
```

## Modify by GPT5.5: 2026-05-24 追加局部补边实验

### 13. 实验动机

全局自适应边数没有超过 DNN14 sharp attention，但仍然可以验证一个更局部的假设：

```text
不要改变全图结构；
基本保持 DNN14 sharp_attention_maskloss；
只给少数推断困难、且相关邻居差距不明显的 target 增加入边。
```

本轮选择：

```text
net_actual_interchange_mw:   至少保留 top16 入边
gross_actual_interchange_mw: 至少保留 top16 入边
total_losses:                至少保留 top12 入边
```

解释是：这些 target 附近的相关性候选 source 差别可能不大，强行只取 top8 可能过早截断信息。

### 14. 修改方式

在 `scripts/run_pipeline.py` 中新增：

```text
graph.target_top_k_overrides
```

该配置不改变基础构图规则。基础图仍然是 DNN14 sharp：

```text
selection = threshold_or_topk
top_k = 8
threshold = 0.25
symmetrize = true
attention_temperature = 1.0
```

然后只对指定 target 的入边做追加：

```text
target_top_k_overrides:
  net_actual_interchange_mw: 16
  gross_actual_interchange_mw: 16
  total_losses: 12
```

注意：这是“至少保留前 N 个 source”，不是强制删到 N 个。如果某个 target 因为 threshold 或 symmetrize 已经超过 N 条边，则不删边。

### 15. 实际补边结果

本轮实际补边如下：

```text
target_field                  target_top_k  degree_before  degree_after  added_edges
net_actual_interchange_mw      16            9              16            7
gross_actual_interchange_mw    16            11             17            6
total_losses                   12            19             19            0
```

因此，`total_losses` 在 DNN14 基础图里本来就已经有 19 条入边，top12 override 没有产生新增边。真正新增的是两个 interchange target。

整体边数：

```text
DNN14 sharp_attention:        908 条边
DNN15 target_extra_edges:     921 条边
```

这说明本轮是非常局部的小改动，而不是像自适应构图那样大幅改变图结构。

### 16. 局部补边测试结果

```text
experiment                  target_MSE  all_MSE   cp_da   cp_rt   losses  mlp_da  30min_reserve
DNN14 sharp_attention       0.331766    0.541777  0.2678  0.4796  0.6768  0.4832  0.3926
target_extra_edges_maskloss 0.335582    0.544957  0.2690  0.4947  0.6082  0.4905  0.3562
target_extra_edges_allloss  0.439477    0.439477  0.2407  0.4817  0.6443  0.4624  0.3666
```

### 17. 结果分析

#### 17.1 局部补边明显优于全局自适应，但仍未超过 DNN14 sharp 主线

`target_extra_edges_maskloss` 的结果：

```text
target_MSE = 0.335582
```

非常接近 DNN14 sharp：

```text
target_MSE = 0.331766
```

明显好于 DNN15 的全局自适应版本：

```text
adaptive_sharp_maskloss:   0.366659
adaptive_relaxed_maskloss: 0.378450
```

这说明用户提出的“只给个别 target 增边”方向比全局改边数更稳，因为它没有破坏大部分 target 的原始图结构。

#### 17.2 对 price/loss 类字段有局部收益

相比 DNN14 sharp，maskloss 局部补边后：

```text
congestion_price_da:    0.2678 -> 0.2690
congestion_price_rt:    0.4796 -> 0.4947
marginal_loss_price_da: 0.4832 -> 0.4905
```

这三个 price/loss 相关字段均有小幅提升。尤其 `congestion_price_rt` 和 `marginal_loss_price_da` 提升比较明确。

但是：

```text
total_losses:           0.6768 -> 0.6082
30min_reserve:          0.3926 -> 0.3562
```

下降明显，导致整体 target_MSE 没有超过 DNN14。

这说明补两个 interchange target 的边会改变它们的隐藏表示，并通过 confidential-source attention 间接影响其它 target。它不是完全无副作用的小改动。

#### 17.3 allloss 在局部补边下改善明显，但仍不是主线

`target_extra_edges_allloss` 比前面所有 allloss 版本都更好：

```text
adaptive_sharp_allloss:    0.480862
adaptive_relaxed_allloss:  0.452866
target_extra_edges_allloss:0.439477
```

并且：

```text
gross_actual_interchange_mw: 0.2367
net_actual_interchange_mw:  -0.8812
```

`gross_actual_interchange_mw` 确实从局部补边中受益，但 `net_actual_interchange_mw` 仍为负 R2。这再次说明两个 interchange 字段并不是同一种难度：gross 可以被部分拉起来，net 仍然很难。

尽管 allloss 的 all_MSE 最低，但它的目标口径和 maskloss 不同，而且 `net_actual_interchange_mw` 仍然不可推断，因此仍不建议替代 maskloss 主线。

### 18. 当前判断

局部补边实验的结论比全局自适应更积极：

```text
1. 只给少数 target 增边是可解释且相对稳定的；
2. 它能改善 congestion_price_rt、marginal_loss_price_da 和 allloss 下的 gross_actual_interchange_mw；
3. 但它会损害 total_losses 和 30min_reserve；
4. total_losses 实际上没有被新增边，因为它在原图已有 19 条入边；
5. net_actual_interchange_mw 即使从 9 条入边加到 16 条，仍然负 R2，说明问题不只是邻居数量不足。
```

因此，目前仍建议以 DNN14 sharp attention maskloss 作为主线。如果继续在 DNN15 中做最后一轮验证，更值得试的是：

```text
只给 gross_actual_interchange_mw 增边；
不要给 net_actual_interchange_mw 增边；
不要动 total_losses；
并仍然使用 maskloss。
```

因为本轮显示 gross 有受益迹象，而 net 的补边可能引入不稳定信息。

## Modify by GPT5.5: 2026-05-24 追加 allloss 专用补边实验

### 19. 实验动机

根据前一轮结果，后续可以把两条线分开：

```text
maskloss:
  使用 DNN14 sharp 主线，即 fixed top_k=8 + threshold=0.25。

allloss:
  允许为少数 all-loss 下困难 target 做局部补边。
```

本轮只测试 allloss，不再重复 maskloss。

用户提出的目标为：

```text
net_actual_interchange_mw
total_losses
total_lmp_da
```

先检查 DNN14 sharp 基础图中的原始入边数：

```text
net_actual_interchange_mw: 9
total_losses:              19
total_lmp_da:              32
```

因此本轮设置：

```text
net_actual_interchange_mw: top24
total_losses:              top24
total_lmp_da:              top24
```

其中 `total_lmp_da` 原本已经有 32 条入边，所以 top24 不会删边，实际不会改变。

### 20. 实际补边结果

```text
target_field                  target_top_k  degree_before  degree_after  added_edges
net_actual_interchange_mw      24            9              24            15
total_losses                   24            19             24            5
total_lmp_da                   24            32             32            0
```

整体边数：

```text
DNN14 sharp_attention:                 908
target_extra_edges_allloss:            921
target_extra_edges_allloss_net24...:   928
```

因此本轮仍是局部补边，不是全局改图。

### 21. all-loss 补边测试结果

```text
experiment                                all_MSE   cp_da   cp_rt   losses  mlp_da  30min   gross   net
target_extra_edges_allloss                0.439477  0.2407  0.4817  0.6443  0.4624  0.3666  0.2367 -0.8812
target_extra_edges_allloss_net24_losses24 0.452614  0.2822  0.4937  0.6615  0.4787  0.3356  0.1598 -1.1041
```

完整实验名为：

```text
target_extra_edges_allloss_net24_losses24_lmp24
```

### 22. 结果分析

将 `net_actual_interchange_mw` 从 9 条入边补到 24 条，并没有改善 net，反而更差：

```text
net_actual_interchange_mw:
  -0.8812 -> -1.1041
```

这说明 net 的低 R2 不是简单的“边不够”。补入更多相关性相近的 source 后，模型可能获得了更多噪声路径，反而更难拟合。

`total_losses` 从 19 条补到 24 条后有所提升：

```text
total_losses:
  0.6443 -> 0.6615
```

price/loss 类字段也普遍提升：

```text
congestion_price_da:
  0.2407 -> 0.2822

congestion_price_rt:
  0.4817 -> 0.4937

marginal_loss_price_da:
  0.4624 -> 0.4787
```

但是 `gross_actual_interchange_mw` 和 `30min_reserve` 下降：

```text
gross_actual_interchange_mw:
  0.2367 -> 0.1598

da_as_total_mw_thirty_minutes_reserve:
  0.3666 -> 0.3356
```

总体 all_MSE 也变差：

```text
0.439477 -> 0.452614
```

因此这次更像是“把模型能力推向 price/loss 类字段，但牺牲 interchange 和 reserve 字段”。

### 23. 当前判断

本轮验证后，allloss 线的判断更清晰：

```text
1. net_actual_interchange_mw 不应该盲目补到 24；
2. total_losses 补到 24 对 total_losses 和 price/loss 类字段有帮助；
3. total_lmp_da 原本已有 32 条入边，不需要补；
4. 如果目标是 all_MSE，上一轮 target_extra_edges_allloss 更好；
5. 如果目标是 price/loss 类字段，net24 + losses24 有局部收益。
```

因此更合理的下一步不是继续给 net 加边，而是测试：

```text
allloss:
  只给 total_losses 补到 24；
  不补 net_actual_interchange_mw；
  不补 total_lmp_da。
```

或者：

```text
allloss:
  gross_actual_interchange_mw 补到 16；
  total_losses 补到 24；
  不补 net_actual_interchange_mw。
```

因为目前证据显示：`gross` 可被补边帮助，`total_losses` 可被补边帮助，但 `net` 补边会伤害整体。

## Modify by GPT5.5: 2026-05-24 追加 allloss 折中补边实验

### 24. 实验设置

本轮继续只测试 allloss，基础图仍然是：

```text
top_k = 8
threshold = 0.25
attention_temperature = 1.0
loss_exclude_confidential = []
```

只对三个 target 做局部补边：

```text
net_actual_interchange_mw:   top16
gross_actual_interchange_mw: top17
total_losses:                top24
```

实际补边结果：

```text
target_field                  target_top_k  degree_before  degree_after  added_edges
net_actual_interchange_mw      16            9              16            7
gross_actual_interchange_mw    17            11             18            7
total_losses                   24            19             24            5
```

`gross_actual_interchange_mw` 设置为 top17 后实际变为 18，是因为基础 threshold/symmetrize 图中已有额外入边，override 只加边、不删边。

整体边数：

```text
DNN14 sharp_attention:                  908
target_extra_edges_allloss:             921
net24_losses24_lmp24:                   928
net16_gross17_losses24:                 927
```

### 25. 折中补边结果

```text
experiment                         all_MSE   cp_da   cp_rt   losses  mlp_da  30min   gross   net
target_extra_edges_allloss         0.439477  0.2407  0.4817  0.6443  0.4624  0.3666  0.2367 -0.8812
net24_losses24_lmp24               0.452614  0.2822  0.4937  0.6615  0.4787  0.3356  0.1598 -1.1041
net16_gross17_losses24             0.443358  0.2165  0.4606  0.6605  0.4633  0.3440  0.3826 -0.9185
```

完整实验名：

```text
target_extra_edges_allloss_net16_gross17_losses24
```

### 26. 折中结果分析

相比 `net24_losses24_lmp24`，把 net 从 top24 收回到 top16，并给 gross 明确补到 top17 后：

```text
all_MSE:
  0.452614 -> 0.443358

gross_actual_interchange_mw:
  0.1598 -> 0.3826

net_actual_interchange_mw:
  -1.1041 -> -0.9185
```

说明：

```text
1. net 补到 24 确实太多；
2. gross 补边非常有效；
3. total_losses 补到 24 可以维持 total_losses 较高 R2。
```

但相比上一轮 `target_extra_edges_allloss`：

```text
all_MSE:
  0.439477 -> 0.443358

congestion_price_da:
  0.2407 -> 0.2165

congestion_price_rt:
  0.4817 -> 0.4606

30min_reserve:
  0.3666 -> 0.3440
```

整体仍略差。它的优势主要集中在：

```text
gross_actual_interchange_mw:
  0.2367 -> 0.3826

total_losses:
  0.6443 -> 0.6605
```

因此这组是一个“gross 和 total_losses 友好”的方案，但不是 all_MSE 最优方案。

### 27. 当前 allloss 线结论

当前 allloss 补边实验中：

```text
all_MSE 最优:
  target_extra_edges_allloss
  即 net top16, gross top16/17, total_losses 不额外补边

gross 最优:
  net16_gross17_losses24

price/loss 类更强:
  net24_losses24_lmp24 对 cp_da/cp_rt/losses/mlp_da 更高，但 net 和 all_MSE 更差
```

如果目标是论文主线稳定性，仍不建议让 allloss 替代 maskloss。  
如果目标是保留一个 allloss 对照，当前推荐：

```text
target_extra_edges_allloss
```

因为它 all_MSE 最低，且不会像 net24 那样严重拉坏 net。

## Modify by GPT5.5: 2026-05-24 追加 allloss 最后折中实验

### 28. 实验设置

本轮继续只测试 allloss，基础设置不变：

```text
top_k = 8
threshold = 0.25
attention_temperature = 1.0
loss_exclude_confidential = []
```

局部补边设置为：

```text
net_actual_interchange_mw:   top16
gross_actual_interchange_mw: top16
total_losses:                top22
```

实际补边结果：

```text
target_field                  target_top_k  degree_before  degree_after  added_edges
net_actual_interchange_mw      16            9              16            7
gross_actual_interchange_mw    16            11             17            6
total_losses                   22            19             23            4
```

`total_losses` 设置 top22 后实际到 23，同样是因为基础图中已有额外 threshold/symmetrize 入边，override 只加边不删边。

### 29. 最后折中实验结果

```text
experiment                         all_MSE   cp_da   cp_rt   losses  mlp_da  30min   gross   net
target_extra_edges_allloss         0.439477  0.2407  0.4817  0.6443  0.4624  0.3666  0.2367 -0.8812
net16_gross16_losses22             0.446135  0.2411  0.4717  0.6292  0.4414  0.3523  0.2081 -0.8884
net16_gross17_losses24             0.443358  0.2165  0.4606  0.6605  0.4633  0.3440  0.3826 -0.9185
net24_losses24_lmp24               0.452614  0.2822  0.4937  0.6615  0.4787  0.3356  0.1598 -1.1041
```

完整实验名：

```text
target_extra_edges_allloss_net16_gross16_losses22
```

### 30. 最终 allloss 补边判断

这组 `net16_gross16_losses22` 没有超过当前最佳 allloss：

```text
target_extra_edges_allloss:
  all_MSE = 0.439477

net16_gross16_losses22:
  all_MSE = 0.446135
```

它也没有保住 `gross_actual_interchange_mw` 的高值：

```text
gross:
  target_extra_edges_allloss = 0.2367
  net16_gross16_losses22     = 0.2081
  net16_gross17_losses24     = 0.3826
```

说明 gross 的提升更依赖“补到 17/18”这一档，而不是 top16。

`total_losses` 补到 22/23 也没有补到 24 好：

```text
total_losses:
  target_extra_edges_allloss = 0.6443
  net16_gross16_losses22     = 0.6292
  net16_gross17_losses24     = 0.6605
  net24_losses24_lmp24       = 0.6615
```

因此最后判断：

```text
1. all_MSE 最优仍然是 target_extra_edges_allloss；
2. gross 最优是 net16_gross17_losses24；
3. total_losses/price-loss 类最优更接近 net24_losses24_lmp24；
4. net 无论补到16还是24都仍是负R2，不能靠补边解决；
5. net16_gross16_losses22 不作为推荐配置。
```

如果后续需要保留一个 allloss baseline，推荐：

```text
target_extra_edges_allloss
```

如果论文中想展示“局部补边可以服务特定 target”，可以把：

```text
net16_gross17_losses24
```

作为说明 gross/total_losses 的辅助实验，但不作为最终主结果。
