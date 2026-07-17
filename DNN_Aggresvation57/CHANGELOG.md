# DNN_Aggresvation57 日志

## Modify by Codex: 2026-07-07

## 1. 子项目定位

验证 **DNN(44->12) 与 GCN 三个变体在推断准确率和 Shapley 敏感度排名上的一致性**。

核心问题不是继续追求一个新的最高 R2,而是检查:

1. 不同推断模型学到的单字段敏感度排序是否稳定;
2. GCN 的图先验/动态先验是否会给出明显不同于 DNN 的敏感度结构;
3. 如果排序高度一致,说明当前单字段归因口径已经接近饱和。

模型设置:

| 模型 | 来源/说明 |
|---|---|
| `dnn` | 现训 MLP,输入 44 个 general 字段,输出 12 个 confidential |
| `gcn_noprior` | 复用已训 GCN,不使用相关先验 |
| `gcn_static` | 复用已训 GCN,静态相关先验 |
| `gcn_dynamic` | 复用已训 GCN,动态先验门控 |

敏感度计算使用蒙特卡洛 Shapley,`value(S)` 为只给 general 子集 S 时的 12-conf 加权 R2。

## 2. 结果

### 2.1 推断准确率

| 模型 | 12-conf 平均 test R2 |
|---|---:|
| `dnn` | 0.8645 |
| `gcn_noprior` | 0.8737 |
| `gcn_static` | 0.8664 |
| `gcn_dynamic` | **0.8746** |
| DNN single-target probe 参考 | 0.8937 |

`gcn_dynamic` 仍是四个模型中最高,但和 single-target DNN probe 还有差距。说明在当前固定字段表、全字段输入、单系统回归设定下,GNN 没有明显预测优势。

### 2.2 Shapley 排名一致性

Spearman 相关:

|  | dnn | gcn_noprior | gcn_static | gcn_dynamic |
|---|---:|---:|---:|---:|
| dnn | 1.000 | 0.937 | 0.902 | 0.888 |
| gcn_noprior | 0.937 | 1.000 | 0.940 | 0.952 |
| gcn_static | 0.902 | 0.940 | 1.000 | 0.912 |
| gcn_dynamic | 0.888 | 0.952 | 0.912 | 1.000 |

top-10 敏感字段重合率:

|  | dnn | gcn_noprior | gcn_static | gcn_dynamic |
|---|---:|---:|---:|---:|
| dnn | 1.00 | 0.80 | 0.80 | 0.80 |
| gcn_noprior | 0.80 | 1.00 | 0.90 | 0.90 |
| gcn_static | 0.80 | 0.90 | 1.00 | 1.00 |
| gcn_dynamic | 0.80 | 0.90 | 1.00 | 1.00 |

## 3. 关键结论

1. **不同模型的单字段 Shapley 排名高度一致。** 即使 DNN 和 GCN 架构不同,最终找到的高风险字段仍大体相同。
2. **图结构没有在单字段敏感度排名上制造新信息。** 当前口径下,GNN 的优势没有体现在最终排序上。
3. **这支持"单字段归因已饱和"的判断。** 后续如果继续只比较 top-k 排名,很难拉开方法差距。

## 4. 意义与下一步

DNN57 的价值是把问题从"哪个模型更会预测"推进到"单字段敏感度是否还有区分度"。

结果表明,如果研究目标仍是一个 44 维敏感度向量,模型结构带来的差异会被强边际字段主导,难以形成清晰贡献。后续需要转向:

- 子集泄露函数 `v(S)`;
- 字段组合协同/冗余;
- 重训口径下的 top-k 忠实性;
- 反事实最小防护集。

## 5. 输出

```
outputs/summary.json                 准确率、Spearman、一致性汇总
outputs/shapley_by_model.csv         4 个模型的字段 Shapley 分数
outputs/accuracy.png                 推断准确率柱状图
outputs/consistency_heatmap.png      Shapley 排名一致性热图
outputs/_probe_reference/            single-target DNN probe 参考结果
scripts/run_all.py                   主实验脚本
gcn_models/*.pt                      复用的 GCN 模型
```
