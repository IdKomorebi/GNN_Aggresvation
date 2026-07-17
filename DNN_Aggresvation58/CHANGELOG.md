# DNN_Aggresvation58 日志

## Modify by Codex: 2026-07-07

## 1. 子项目定位

验证 **随机遮蔽训练的 `gcn_dynamic` 是否能作为更忠实的子集泄露近似 oracle**。

旧口径的问题是:训练好的 full-input 模型只在全字段分布上见过数据,评估 `v(S)` 时把 S 之外字段置 0,测到的可能是分布外行为,不一定代表"攻击者只有 S 时能推断多少"。

本子项目训练一个遮蔽版 `gcn_dynamic(masked)`:

```text
每个训练 batch:
  q ~ Uniform(0, 1)
  对每个 general 字段做 Bernoulli(q) 保留
  confidential 始终置 0
```

这样模型在训练中见过大量字段子集,前向查询 `v(S)` 更接近"攻击者只有 S"的口径。

## 2. 主要结果

### 2.1 全字段准确率与小子集能力

旧版 `gcn_dynamic(zero-mask)` 与新版 `gcn_dynamic(masked)` 的对比:

| 指标 | zero-mask 旧版 | masked 新版 |
|---|---:|---:|
| unweighted 全字段平均 R2 | 0.8746 | 0.8301 |
| weighted/full `v(S)` 上界 | 0.8874 | 0.8520 |
| Shapley Spearman(old vs new) | \- | 0.9335 |
| top-10 重合率 | \- | 0.80 |

遮蔽训练牺牲了一部分全字段准确率,但明显改善了小子集 `v(S)`:

| k | zero-mask 旧版 `v(S)` | masked 新版 `v(S)` |
|---:|---:|---:|
| 1 | 0.0077 | 0.0737 |
| 2 | 0.0171 | 0.2019 |
| 4 | 0.0542 | 0.3654 |
| 8 | 0.1594 | 0.5328 |
| 16 | 0.2233 | 0.6401 |
| 24 | 0.4248 | 0.7547 |
| 32 | 0.5968 | 0.8028 |
| 44 | 0.8874 | 0.8520 |

### 2.2 top-k 敏感字段曲线

遮蔽版在自己的 `v(S)` 口径下,top-k 小子集明显更高:

| 方法 | 前 10 个 k 的平均 top-k R2 |
|---|---:|
| single | 0.4978 |
| mask | 0.5354 |
| shapley | **0.6478** |
| ig | 0.6161 |
| lrp | 0.6367 |
| gate | 0.5122 |
| random | 0.3341 |

五模型对比中,`gcn_dynamic(masked)` 的小 k 曲线最高,但 full-input 上界最低:

| 模型 | full 上界 | 前 10 个 k 平均 |
|---|---:|---:|
| `DNN(44->12)` | 0.8819 | 0.4388 |
| `gcn_noprior` | 0.8861 | 0.4184 |
| `gcn_static` | 0.8807 | 0.4400 |
| `gcn_dynamic(zero-mask)` | **0.8874** | 0.3990 |
| `gcn_dynamic(masked)` | 0.8520 | **0.6478** |

## 3. 关键结论

1. **遮蔽训练确实让小子集查询更合理。** 旧 zero-mask 在小 k 时几乎塌陷,说明它不适合直接当 `v(S)`。
2. **遮蔽训练不是免费午餐。** 它提高了任意子集鲁棒性,但降低了全字段推断准确率。
3. **Shapley 排名没有发生根本变化。** old/new Spearman 仍有 0.9335,top-10 重合 0.80,说明单字段高风险字段仍被同一批强边际特征主导。
4. **DNN58 只能说明近似 oracle 更像样,不能替代重训真值。** 要判断 `v(S)` 的忠实性,仍需要对照逐子集重训结果。

## 4. 方法论教训

`zero-mask` 和 `masked-trained` 测的不是同一件事:

```text
zero-mask:
  固定 full-input 模型,把缺失字段置 0
  容易测到分布外行为

masked-trained:
  训练时见过随机字段子集
  更接近攻击者只有 S 时的摊销推断模型

retrain:
  每个 S 单独训练攻击者
  最忠实,但最贵
```

因此 DNN58 的自然下一步是 DNN59:用重训曲线校验各种近似和排名。

## 5. 输出

```
trained/gcn_dynamic_masked.pt         随机遮蔽训练版模型
outputs/summary.json                  old/new 准确率、v(S) 曲线、Shapley 一致性
outputs/old_vs_masked.csv             old/new Shapley 与 single-field leakage
outputs/topk_summary.json             遮蔽版多方法 top-k 汇总
outputs/topk_5models_summary.json     五模型 top-k 汇总
outputs/topk_*.csv/.png               top-k 曲线
outputs/retrain/retrain_curves.csv    初版重训对照曲线
scripts/train_masked.py               遮蔽训练脚本
scripts/compare_faithful.py           old vs masked 忠实性对比
scripts/topk_*.py                     top-k 分析脚本
```
